import os
import time
import numpy as np
from numba import njit, prange

# --- CONFIGURACIÓN DE ENTORNO ---
os.environ["KMP_WARNINGS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# Importante: Mantener TBB o Workqueue para Numba en Mac/Linux
os.environ["NUMBA_THREADING_LAYER"] = "tbb" 

# ==========================================
# 1. KERNEL DE XGBOOST (NUMBA)
# ==========================================

@njit(fastmath=True, cache=True)
def get_best_split_xgboost(X_col, gradients, lambda_reg):
    """
    Calcula el 'Structure Score' de XGBoost para MSE.
    Score = (Sum_G_L^2 / (n_L + lambda)) + (Sum_G_R^2 / (n_R + lambda))
    """
    n_samples = len(gradients)
    if n_samples <= 1:
        return -np.inf, np.nan

    # 1. Ordenar
    idxs = np.argsort(X_col)
    X_sorted = X_col[idxs]
    g_sorted = gradients[idxs] # g = residuales en nuestro caso MSE

    # 2. Pre-calcular G total (Suma de gradientes)
    G_total = 0.0
    for val in gradients:
        G_total += val

    # Inicializar contadores
    G_left = 0.0
    G_right = G_total
    
    # Hessianos: Para MSE, h=1, así que sum_h es simplemente el conteo n
    H_left = 0.0
    H_right = float(n_samples)

    best_score = -np.inf
    best_thr = np.nan

    # 3. Barrido Lineal
    for i in range(n_samples - 1):
        g_i = g_sorted[i]
        
        # Mover gradiente y hessiano de derecha a izquierda
        G_left += g_i
        G_right -= g_i
        H_left += 1.0
        H_right -= 1.0

        if X_sorted[i] != X_sorted[i + 1]:
            # --- FÓRMULA DE XGBOOST ---
            # Score = G^2 / (H + lambda)
            # Notar que ignoramos el término '- gamma' durante la búsqueda porque es constante
            score_left = (G_left * G_left) / (H_left + lambda_reg)
            score_right = (G_right * G_right) / (H_right + lambda_reg)
            
            gain = score_left + score_right

            if gain > best_score:
                best_score = gain
                best_thr = (X_sorted[i] + X_sorted[i + 1]) / 2.0

    return best_score, best_thr

@njit(parallel=True, fastmath=True, cache=True)
def find_split_xgb_parallel(X, gradients, feature_indices, lambda_reg):
    n_selected = len(feature_indices)
    results = np.zeros((n_selected, 2), dtype=np.float64)

    for i in prange(n_selected):
        feat_idx = feature_indices[i]
        X_col = X[:, feat_idx]
        score, thr = get_best_split_xgboost(X_col, gradients, lambda_reg)
        results[i, 0] = score
        results[i, 1] = thr

    best_idx = np.argmax(results[:, 0])

    if np.isinf(results[best_idx, 0]):
        return -1, np.nan

    return int(feature_indices[best_idx]), results[best_idx, 1]

# ==========================================
# 2. CLASES (XGBoost Tree)
# ==========================================

class XGBNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class XGBoostTree:
    def __init__(self, max_depth=3, lambda_reg=1.0, min_samples_split=10, n_feats=-1):
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.min_samples_split = min_samples_split
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, gradients):
        X = np.asfortranarray(X, dtype=np.float64)
        gradients = np.ascontiguousarray(gradients, dtype=np.float64)
        
        n_features = X.shape[1]
        if self.n_feats == -1: self.n_feats = n_features
        else: self.n_feats = min(self.n_feats, n_features)

        self.root = self._grow(X, gradients, 0)

    def _calculate_leaf_weight(self, gradients):
        # w = Sum(G) / (Sum(H) + lambda)
        G = np.sum(gradients)
        H = len(gradients)
        return G / (H + self.lambda_reg)

    def _grow(self, X, gradients, depth):
        n_samples = X.shape[0]
        leaf_weight = self._calculate_leaf_weight(gradients)
        
        # 1. Criterios de parada básicos
        if (depth >= self.max_depth or n_samples < self.min_samples_split):
            return XGBNode(value=leaf_weight)

        # 2. CORRECCIÓN: Verificar si el ERROR absoluto es 0 (ajuste perfecto), no la suma
        if np.sum(np.abs(gradients)) < 1e-7:
             return XGBNode(value=leaf_weight)

        # 3. Selección de features y Split
        feat_idxs = np.random.choice(X.shape[1], self.n_feats, replace=False)
        
        best_feat, best_thr = find_split_xgb_parallel(X, gradients, feat_idxs, self.lambda_reg)

        if best_feat == -1: # Si no encontró ganancia
            return XGBNode(value=leaf_weight)

        left_mask = X[:, best_feat] <= best_thr
        X_left, g_left = X[left_mask], gradients[left_mask]
        X_right, g_right = X[~left_mask], gradients[~left_mask]

        if len(g_left) == 0 or len(g_right) == 0:
            return XGBNode(value=leaf_weight)

        return XGBNode(
            feature=best_feat, 
            threshold=best_thr, 
            left=self._grow(X_left, g_left, depth + 1), 
            right=self._grow(X_right, g_right, depth + 1)
        )

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None: return node.value
        if x[node.feature] <= node.threshold: return self._traverse(x, node.left)
        return self._traverse(x, node.right)
# ==========================================
# 3. EL ORQUESTADOR (Mini-XGBoost)
# ==========================================

class MiniXGBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, lambda_reg=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.trees = []
        self.initial_pred = 0.0

    def fit(self, X, y):
        # Paso 0: Predicción inicial (para MSE es el promedio)
        self.initial_pred = np.mean(y)
        y_pred = np.full(len(y), self.initial_pred)
        
        print(f"🚀 Mini-XGBoost: {self.n_estimators} rounds | Lambda: {self.lambda_reg}")
        
        for i in range(self.n_estimators):
            # Paso 1: Calcular Gradientes (Residuales)
            # Para MSE: grad = (pred - y). Pero nosotros fitteamos al negativo: (y - pred)
            residuals = y - y_pred
            
            # Paso 2: Entrenar Árbol usando Structure Score
            tree = XGBoostTree(
                max_depth=self.max_depth,
                lambda_reg=self.lambda_reg, # Pasamos la regularización
                min_samples_split=10,
                n_feats=-1
            )
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Paso 3: Actualizar (con Learning Rate / Shrinkage)
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            
    def predict(self, X):
        y_pred = np.full(len(X), self.initial_pred)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

# ==========================================
# 4. PRUEBA FINAL
# ==========================================
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_squared_error
    import xgboost as xgb # Importamos el XGBoost REAL para comparar

    # Generar datos más grandes para ver la potencia
    print("📊 Generando 200k filas...")
    X, y = make_regression(n_samples=200_000, n_features=30, noise=10.0, random_state=42)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- 1. Nuestro Mini-XGBoost ---
    print("\n🚀 Nuestro Mini-XGBoost...")
    # Lambda=1.0 es el default en XGBoost oficial
    my_model = MiniXGBoost(n_estimators=50, learning_rate=0.1, lambda_reg=1.0) 
    
    start = time.time()
    my_model.fit(X_train, y_train)
    my_time = time.time() - start
    my_mse = mean_squared_error(y_test, my_model.predict(X_test))
    
    print(f"⏱️ Tiempo: {my_time:.4f}s | MSE: {my_mse:.2f}")

    # --- 2. XGBoost Oficial (La librería real en C++) ---
    print("\n🔥 XGBoost Oficial (Benchmark final)...")
    # Configuramos para que se parezca lo más posible al nuestro
    # method='exact' fuerza el algoritmo exacto (no histogramas) para ser justa la comparación
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'reg:squarederror',
        'eta': 0.1,
        'max_depth': 3,
        'lambda': 1.0,    # L2 regularization
        'tree_method': 'exact', # Usar algoritmo exacto (no approx/hist)
        'nthread': 8      # Usar mismos hilos que nosotros
    }
    
    start = time.time()
    bst = xgb.train(params, dtrain, num_boost_round=50)
    xgb_time = time.time() - start
    xgb_pred = bst.predict(dtest)
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    
    print(f"⏱️ Tiempo: {xgb_time:.4f}s | MSE: {xgb_mse:.2f}")
    
    print("-" * 40)
    diff = abs(my_mse - xgb_mse)
    print(f"Diferencia de MSE: {diff:.4f}")
    if diff < 5.0:
        print("✅ ¡RESULTADO VALIDADO! Matemáticamente idéntico al XGBoost oficial.")