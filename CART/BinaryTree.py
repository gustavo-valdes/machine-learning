import time
import numpy as np
import graphviz
from numba import njit, prange

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# 1. EL MOTOR (NUMBA KERNELS)
# ==========================================

@njit(fastmath=True, cache=True)
def get_best_split_linear_scan(X_col, y):
    """
    Algoritmo Sort & Scan: O(N log N).
    Encuentra el mejor corte para una sola columna.
    """
    n_samples = len(y)
    if n_samples <= 1:
        return -1.0, 0.0

    # Ordenar
    idxs = np.argsort(X_col)
    X_sorted = X_col[idxs]
    y_sorted = y[idxs]

    # Inicializar contadores globales
    total_ones = np.sum(y_sorted)
    total_zeros = n_samples - total_ones
    
    # Contadores dinámicos (Lado Izquierdo)
    left_ones = 0.0
    left_zeros = 0.0
    
    best_gain = -1.0
    best_thr = np.nan
    
    # Gini del nodo padre
    p0 = total_zeros / n_samples
    p1 = total_ones / n_samples
    parent_gini = 1.0 - (p0*p0 + p1*p1)

    # Paso C: Barrido Lineal (O(N))
    for i in range(n_samples - 1):
        # Mover un dato de Derecha a Izquierda
        if y_sorted[i] == 1.0:
            left_ones += 1
        else:
            left_zeros += 1
            
        # Solo evaluamos si el valor de X cambia
        if X_sorted[i] != X_sorted[i+1]:
            # Deducir el lado derecho
            right_ones = total_ones - left_ones
            right_zeros = total_zeros - left_zeros
            
            n_l = left_ones + left_zeros
            n_r = right_ones + right_zeros
            
            # Gini Izquierdo
            gini_l = 1.0 - ((left_zeros/n_l)**2 + (left_ones/n_l)**2)
            # Gini Derecho
            gini_r = 1.0 - ((right_zeros/n_r)**2 + (right_ones/n_r)**2)
            
            # Gain
            child_gini = (n_l/n_samples)*gini_l + (n_r/n_samples)*gini_r
            gain = parent_gini - child_gini
            
            if gain > best_gain:
                best_gain = gain
                best_thr = (X_sorted[i] + X_sorted[i+1]) / 2.0
                
    return best_gain, best_thr

@njit(parallel=True, fastmath=True, cache=True)
def find_best_split_numba_optimized(X, y, feature_indices):
    """
    Orquestador Paralelo: Reparte las columnas entre los hilos del CPU.
    """
    n_selected = len(feature_indices)
    results = np.zeros((n_selected, 2), dtype=np.float64) # [Gain, Threshold]
    
    for i in prange(n_selected):
        feat_idx = feature_indices[i]
        X_col = X[:, feat_idx]
        gain, thr = get_best_split_linear_scan(X_col, y)
        results[i, 0] = gain
        results[i, 1] = thr

    # Reducción (Argmax)
    best_local_idx = np.argmax(results[:, 0])
    best_gain = results[best_local_idx, 0]
    
    if best_gain <= 0:
        return -1, np.nan 
        
    return feature_indices[best_local_idx], results[best_local_idx, 1]

# ==========================================
# 2. LA ESTRUCTURA (PYTHON POO)
# ==========================================

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class FastDecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        # Numba necesita arrays contiguos y float64
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_labels = np.unique(y)

        # Criterios de Parada
        if (depth >= self.max_depth or len(unique_labels) == 1 or n_samples < self.min_samples_split):
            return Node(value=self._most_common(y))

        # Selección de Features (Bagging)
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        
        # --- EL CEREBRO DE NUMBA ---
        best_feat, best_thr = find_best_split_numba_optimized(X, y, feat_idxs)
        
        if best_feat == -1: # No se encontró corte válido
            return Node(value=self._most_common(y))

        # Dividir datos (Python side)
        left_mask = X[:, best_feat] <= best_thr
        right_mask = ~left_mask
        
        # Recursión (Crear hijos)
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        # Guardar la decisión en el Nodo actual
        return Node(feature=best_feat, threshold=best_thr, left=left, right=right)

    def _most_common(self, y):
        return 1.0 if np.sum(y) >= len(y)/2 else 0.0

    # --- PREDICCIÓN ---
    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value # Es una Hoja
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


def print_tree_structure(node, spacing="", side="Raíz"):
    """
    Imprime el árbol en la consola de forma recursiva.
    """
    # Si es una HOJA
    if node.value is not None:
        print(f"{spacing}└── {side}: PREDICT {node.value}")
        return

    # Si es un NODO
    print(f"{spacing}├── {side}: [X_{node.feature} <= {node.threshold:.3f}]")
    
    # Recursividad (aumentamos el espaciado)
    print_tree_structure(node.left, spacing + "│   ", "Izquierda (True)")
    print_tree_structure(node.right, spacing + "│   ", "Derecha (False)")



def export_graphviz(node, feature_names=None):
    """
    Genera el código DOT para visualizar el árbol.
    Copia el output en http://www.webgraphviz.com/
    """
    lines = []
    lines.append("digraph DecisionTree {")
    lines.append('    node [shape=box, style="filled, rounded", color="black", fontname="helvetica"];')
    lines.append('    edge [fontname="helvetica"];')

    def recurse(n, my_id):
        # Generar etiqueta del nodo
        if n.value is not None:
            # Es Hoja
            label = f"Class: {int(n.value)}"
            color = "#e5813999" if n.value == 0 else "#399de599" # Naranja vs Azul
            lines.append(f'    {my_id} [label="{label}", fillcolor="{color}", shape=ellipse];')
        else:
            # Es Nodo
            fname = f"X_{n.feature}" if feature_names is None else feature_names[n.feature]
            label = f"{fname} <= {n.threshold:.3f}"
            lines.append(f'    {my_id} [label="{label}", fillcolor="#e5e5e5"];')
            
            # Hijos
            left_id = f"{my_id}L"
            right_id = f"{my_id}R"
            
            recurse(n.left, left_id)
            recurse(n.right, right_id)
            
            # Aristas
            lines.append(f'    {my_id} -> {left_id} [labeldistance=2.5, labelangle=45, headlabel="True"];')
            lines.append(f'    {my_id} -> {right_id} [labeldistance=2.5, labelangle=-45, headlabel="False"];')

    # Iniciar recursión desde la raíz
    recurse(node, "root")
    lines.append("}")
    
    return "\n".join(lines)



# ==========================================
# 3. EL TEST (BENCHMARK)
# ==========================================

if __name__ == "__main__":
    # A. Generación de Datos Sintéticos (100k filas)
    N_ROWS = 10_000_000
    N_COLS = 20
    print(f"📊 Generando dataset {N_ROWS}x{N_COLS}...")
    
    np.random.seed(42)
    X = np.random.randn(N_ROWS, N_COLS)
    
    # REGLA SECRETA: Si (Col0 > 0.5) Y (Col5 < -0.5) -> Clase 1. Si no, Clase 0.
    # El árbol debería descubrir esto automáticamente.
    y = ((X[:, 0] > 0.5) & (X[:, 5] < -0.5)).astype(np.float64)
    
    # Dividir Train/Test (80/20) manualmente
    split = int(N_ROWS * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

    # B. Inicializar Modelo
    model = FastDecisionTree(max_depth=10, min_samples_split=10)

    # C. Warmup (Compilación JIT)
    print("🔥 Calentando motores (Compilando Numba)...")
    model.fit(X_train[:200], y_train[:200]) # Pequeño batch
    
    # D. Entrenamiento Real
    print("🚀 Entrenando Modelo (Full Speed)...")
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_time = end - start
    print(f"   ⏱️ Tiempo de Entrenamiento: {train_time:.4f} segundos")

    # E. Predicción e Inferencia
    print("🔮 Prediciendo en Test Set...")
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()
    infer_time = end - start
    
    # F. Métricas
    accuracy = np.mean(y_pred == y_test)
    print(f"   🎯 Accuracy: {accuracy * 100:.2f}%")
    print(f"   ⚡ Tiempo Inferencia: {infer_time:.4f}s")

    # G. Verificación de cordura (¿Aprendió la regla?)
    print("\n--- Anatomía de la Raíz ---")
    root = model.root
    print(f"Nodo Raíz corta en Feature: {root.feature} (Esperado: 0.5 o -0.5)")
    print(f"Umbral de corte: {root.threshold:.4f}")

    # H. Imprimir estructura del árbol
    print("\n--- Estructura del Árbol ---")
    print_tree_structure(model.root)

    dot_data = export_graphviz(model.root)

    with open("arbol.dot", "w") as f:
        f.write(dot_data)

    # Abrir con graphviz interactive (extensión vscode)

    print("🤖 Entrenando SKLEARN DecisionTreeClassifier...")
    clf_sk = DecisionTreeClassifier(max_depth=10, criterion='gini', min_samples_split=2)

    start_sk = time.time()
    clf_sk.fit(X_train, y_train)
    end_sk = time.time()
    time_sk = end_sk - start_sk
    acc_sk = accuracy_score(y_test, clf_sk.predict(X_test))
    print(f"   ⏱️ Tiempo Sklearn: {time_sk:.4f} s")
    print(f"   🎯 Accuracy Sklearn: {acc_sk:.4f}")

    speedup = time_sk / train_time
    print(f"🏆 VEREDICTO:")

    if train_time < time_sk:
        print(f"   ¡NUESTRO MODELO ES {speedup:.2f}x MÁS RÁPIDO!")
    else:
        print(f"   Sklearn ganó por {train_time/time_sk:.2f}x (Goliat sigue siendo fuerte).")
        
    print(f"   Diferencia de Accuracy: {abs(acc_sk - accuracy):.4f}")