import numpy as np
import time
from numba import njit, prange

# --- CONFIGURACIÓN DE ENTORNO (Arch Linux / MKL / OpenBLAS) ---
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Numba controlará el paralelismo
os.environ["NUMBA_NUM_THREADS"] = "8" # Ajusta a tus núcleos físicos

# ==========================================
# 1. MOTOR MATEMÁTICO (NUMBA KERNELS)
# ==========================================

@njit(fastmath=True, cache=True)
def get_best_split_regression(X_col, y):
    """
    Escaneo lineal optimizado para Regresión (MSE).
    Maximiza el criterio: (Sum_L^2 / n_L) + (Sum_R^2 / n_R)
    """
    n_samples = len(y)
    if n_samples <= 1:
        return -np.inf, 0.0

    # 1. Ordenar (Costoso pero necesario)
    idxs = np.argsort(X_col)
    X_sorted = X_col[idxs]
    y_sorted = y[idxs]

    # 2. Pre-calcular sumas totales (Estado inicial: Todo a la derecha)
    sum_y_total = 0.0
    for val in y:
        sum_y_total += val
    
    # Inicializar contadores
    sum_y_l = 0.0
    sum_y_r = sum_y_total
    n_l = 0.0
    n_r = float(n_samples)

    best_score = -np.inf
    best_thr = np.nan

    # 3. Barrido Lineal
    for i in range(n_samples - 1):
        y_curr = y_sorted[i]
        
        # Mover de Derecha a Izquierda
        sum_y_l += y_curr
        sum_y_r -= y_curr
        n_l += 1.0
        n_r -= 1.0

        # Solo evaluamos si hay cambio en X
        if X_sorted[i] != X_sorted[i + 1]:
            # Calcular Score (Proxy de reducción de varianza)
            # Evitamos divisiones por cero con lógica simple (n_l y n_r > 0 por el loop)
            current_score = (sum_y_l * sum_y_l) / n_l + (sum_y_r * sum_y_r) / n_r

            if current_score > best_score:
                best_score = current_score
                best_thr = (X_sorted[i] + X_sorted[i + 1]) / 2.0

    return best_score, best_thr

@njit(parallel=True, fastmath=True, cache=True)
def find_best_split_parallel_reg(X, y, feature_indices):
    """
    Distribuye las columnas (features) entre los núcleos.
    """
    n_selected = len(feature_indices)
    # Matriz de resultados: [Score, Threshold]
    results = np.zeros((n_selected, 2), dtype=np.float64)

    for i in prange(n_selected):
        feat_idx = feature_indices[i]
        # Extraer columna
        X_col = X[:, feat_idx]
        
        score, thr = get_best_split_regression(X_col, y)
        
        results[i, 0] = score
        results[i, 1] = thr

    # Buscar el mejor split entre todas las columnas probadas
    best_idx = np.argmax(results[:, 0])

    # Si el score es -inf (no se encontró split válido), retornar error
    if results[best_idx, 0] == -np.inf:
        return -1, np.nan

    return feature_indices[best_idx], results[best_idx, 1]

# ==========================================
# 2. CLASES (Regression Tree)
# ==========================================

class RegressionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Float (Promedio)

class RegressionTree:
    def __init__(self, min_samples_split=2, max_depth=10, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        # Asegurar tipos contiguos y FLOAT para y
        X = np.asfortranarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)

        n_features = X.shape[1]
        if self.n_feats is None: 
            self.n_feats = n_features # En regresión se suelen usar todas, o sqrt/3
        elif self.n_feats == -1: 
            self.n_feats = n_features
        else: 
            self.n_feats = min(self.n_feats, n_features)

        self.root = self._grow(X, y, 0)

    def _grow(self, X, y, depth):
        n_samples = X.shape[0]
        
        # Valor de la hoja = Promedio
        mean_val = np.mean(y) if n_samples > 0 else 0.0
        
        # Criterio de parada: Pura varianza 0 ya es difícil en float, usamos profundidad y n_samples
        if (depth >= self.max_depth or n_samples < self.min_samples_split):
            return RegressionNode(value=mean_val)

        # Si la varianza es muy baja, también paramos (opcional, evita splits inútiles)
        if np.var(y) < 1e-7:
             return RegressionNode(value=mean_val)

        # Selección de features
        feat_idxs = np.random.choice(X.shape[1], self.n_feats, replace=False)

        # --- MOTOR NUMBA ---
        best_feat, best_thr = find_best_split_parallel_reg(X, y, feat_idxs)

        if best_feat == -1:
            return RegressionNode(value=mean_val)

        # Split
        left_mask = X[:, best_feat] <= best_thr
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            return RegressionNode(value=mean_val)

        return RegressionNode(
            feature=best_feat, 
            threshold=best_thr, 
            left=self._grow(X_left, y_left, depth + 1), 
            right=self._grow(X_right, y_right, depth + 1)
        )

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

# ==========================================
# 3. BENCHMARK
# ==========================================
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_squared_error

    # 1. Generar Datos de Regresión
    print("📊 Generando datos de regresión (1M filas, 20 feats)...")
    X, y = make_regression(
        n_samples=1_000_000, 
        n_features=20, 
        n_informative=10, 
        noise=10.0, # Un poco de ruido para hacerlo difícil
        random_state=42
    )
    
    # Split manual
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 2. Sklearn (Referencia)
    print("\n🤖 Sklearn DecisionTreeRegressor...")
    sk_model = DecisionTreeRegressor(max_depth=10, min_samples_split=20)
    
    t0 = time.time()
    sk_model.fit(X_train, y_train)
    sk_time = time.time() - t0
    
    sk_pred = sk_model.predict(X_test)
    sk_mse = mean_squared_error(y_test, sk_pred)
    print(f"   ⏱️ Tiempo: {sk_time:.4f}s | MSE: {sk_mse:.2f}")

    # 3. Nuestro Modelo (Numba)
    print("\n🚀 Numba RegressionTree...")
    # Warmup
    dummy_model = RegressionTree(max_depth=2)
    dummy_model.fit(X_train[:100], y_train[:100])
    
    my_model = RegressionTree(max_depth=10, min_samples_split=20, n_feats=-1)
    
    t0 = time.time()
    my_model.fit(X_train, y_train)
    my_time = time.time() - t0
    
    my_pred = my_model.predict(X_test)
    my_mse = mean_squared_error(y_test, my_pred)
    
    print(f"   ⏱️ Tiempo: {my_time:.4f}s | MSE: {my_mse:.2f}")
    
    print("-" * 40)
    print(f"Velocidad Relativa: {sk_time / my_time:.2f}x (Sklearn vs Nuestro)")

    print(my_mse - sk_mse)
