import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from numba import njit, prange

matplotlib.use('TkAgg')

# ==========================================
# 1. KERNEL FUNCTIONS
# ==========================================


@njit(fastmath=True, cache=True)
def rbf_kernel(x1, x2, gamma):
    """K(x, y) = exp(-gamma * ||x - y||^2)"""
    # Distancia Euclidiana al cuadrado
    diff = x1 - x2
    dist_sq = np.dot(diff, diff)
    return np.exp(-gamma * dist_sq)


@njit(fastmath=True, cache=True)
def linear_kernel(x1, x2, gamma=1.0):
    return np.dot(x1, x2)


@njit(fastmath=True, cache=True)
def polynomial_kernel(x1, x2, gamma=1.0, degree=3.0, coef0=1.0):
    """(gamma * <x,y> + coef0)^degree"""
    dot_prod = np.dot(x1, x2)
    base = gamma * dot_prod + coef0
    return base**degree


@njit(fastmath=True, cache=True)
def sigmoid_kernel(x1, x2, gamma=1.0, coef0=0.0):
    """tanh(gamma * <x,y> + coef0)"""
    dot_prod = np.dot(x1, x2)
    return np.tanh(gamma * dot_prod + coef0)


@njit(fastmath=True, cache=True)
def laplacian_kernel(x1, x2, gamma=1.0):
    """exp(-gamma * |x-y|_1)"""
    # Distancia Manhattan (L1)
    diff = np.abs(x1 - x2)
    dist_l1 = np.sum(diff)
    return np.exp(-gamma * dist_l1)


# ==========================================
# 2. TRAINING (KERNEL PEGASOS)
# ==========================================


@njit(fastmath=True, cache=True)
def train_kernel_pegasos(X, y, kernelfunc, alphas, gamma, lambda_param, n_iters):
    """
    Entrena los coeficientes Alpha directamente.
    w no existe explícitamente. w = sum(alpha_i * y_i * phi(x_i))
    """
    n_samples = X.shape[0]

    # PEGASOS usa un learning rate decreciente: 1 / (lambda * t)
    # Para estabilidad numérica, usamos un contador t global
    t = 1.0

    for _ in range(n_iters):
        # Elegimos un índice aleatorio (Stochastic Gradient Descent)
        i = np.random.randint(0, n_samples)

        # 1. CALCULAR PREDICCIÓN ACTUAL f(x_i)
        # f(x) = sum(alpha_j * y_j * K(x_j, x_i)) * (1 / (lambda * t))
        # Nota: En PEGASOS Kernelizado, a menudo se mantiene un escalar de escala
        # para evitar multiplicar todo el vector alpha en cada paso.
        # Aquí haremos la versión explícita para claridad (más lenta pero legible).

        prediction = 0.0
        for j in range(n_samples):
            if alphas[j] > 0:  # Solo iteramos sobre vectores de soporte activos
                prediction += alphas[j] * y[j] * kernelfunc(X[j], X[i], gamma)

        # Escalar por el factor de regularización acumulado (simplificado para SGD)
        prediction = prediction * (1.0 / (lambda_param * t))

        # 2. VERIFICAR MARGEN (Hinge Loss)
        # Si y * pred < 1, hay error o estamos dentro del margen
        if y[i] * prediction < 1.0:
            # Actualizamos el alpha correspondiente
            # En el primal esto era: w = (1 - 1/t)w + (1/lambda*t)yx
            # En el dual (alphas) equivale a incrementar el peso de este ejemplo
            alphas[i] += 1.0

        t += 1.0

    return alphas


@njit(parallel=True, fastmath=True)
def predict_kernel_svm(
    X_test, X_train, y_train, kernelfunc, alphas, gamma, lambda_param, t_final
):
    """
    Predicción masiva usando los alphas aprendidos.
    """
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]
    preds = np.zeros(n_test)

    # Factor de escala final del entrenamiento
    scale = 1.0 / (lambda_param * t_final)

    for i in prange(n_test):
        score = 0.0
        for j in range(n_train):
            if alphas[j] > 0:
                score += (
                    alphas[j] * y_train[j] * kernelfunc(X_train[j], X_test[i], gamma)
                )

        preds[i] = scale * score

    return np.sign(preds)


def predict_score_svm(
    X_test, X_train, y_train, alphas, gamma, lambda_param, t_final, kernelfunc
):
    """Devuelve el puntaje crudo (distancia) en lugar del signo."""
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]
    scores = np.zeros(n_test)

    scale = 1.0 / (lambda_param * t_final)

    for i in prange(n_test):
        score = 0.0
        for j in range(n_train):
            if alphas[j] > 0:
                score += (
                    alphas[j] * y_train[j] * kernelfunc(X_train[j], X_test[i], gamma)
                )
        scores[i] = score * scale

    return scores


# ==========================================
# 3. CLASE ORQUESTADORA
# ==========================================


class KernelSVM:
    def __init__(self, gamma=0.5, lambda_param=0.01, n_iters=2000):
        self.gamma = gamma
        self.lambda_param = lambda_param  # Equivalente a 1 / (n * C)
        self.n_iters = n_iters
        self.alphas = None
        self.X_train = None  # Necesitamos guardar X para predecir (Kernel)
        self.y_train = None
        self.t_final = 0.0
        self.kernelfunc = None

    def fit(self, X, y, kernelfunc):
        self.kernelfunc = kernelfunc

        # Asegurar tipos
        X = np.ascontiguousarray(X, dtype=np.float64)
        y_ = np.where(y <= 0, -1, 1).astype(np.float64)

        n_samples = X.shape[0]
        self.alphas = np.zeros(n_samples, dtype=np.float64)
        self.X_train = X  # OJO: Esto consume memoria, necesario para Kernel
        self.y_train = y_

        # Entrenar
        total_steps = self.n_iters

        self.alphas = train_kernel_pegasos(
            X,
            y_,
            self.kernelfunc,
            self.alphas,
            self.gamma,
            self.lambda_param,
            total_steps,
        )
        self.t_final = float(total_steps)

    def predict(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        return predict_kernel_svm(
            X,
            self.X_train,
            self.y_train,
            self.kernelfunc,
            self.alphas,
            self.gamma,
            self.lambda_param,
            self.t_final,
        )

    def decision_function(self, X):
        """Retorna la distancia al hiperplano (sin signo)"""
        X = np.ascontiguousarray(X, dtype=np.float64)
        return predict_score_svm(
            X,
            self.X_train,
            self.y_train,
            self.alphas,
            self.gamma,
            self.lambda_param,
            self.t_final,
            self.kernelfunc,
        )


def visualize_svm_street(model, X, y, resolution=0.02):
    print("Dibujando la calle y los vectores de soporte...")

    # 1. Configurar Malla
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )

    # 2. Obtener puntajes crudos (Distancias)
    grid_points = np.array([xx1.ravel(), xx2.ravel()]).T
    Z = model.decision_function(grid_points)  # Usamos la nueva función
    Z = Z.reshape(xx1.shape)

    plt.figure(figsize=(10, 8))

    # 3. Dibujar Zonas de Color (Fondo suave)
    # Usamos niveles grandes para colorear todo el fondo según el signo
    plt.contourf(
        xx1, xx2, Z, levels=[-100, 0, 100], colors=["#FF9999", "#9999FF"], alpha=0.3
    )

    # 4. DIBUJAR LA CALLE (Líneas de Contorno)
    # Level 0 = Frontera de Decisión (Línea Sólida)
    # Levels -1, 1 = Márgenes (Líneas Discontinuas)
    contours = plt.contour(
        xx1,
        xx2,
        Z,
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],  # Dashed, Solid, Dashed
        linewidths=[1, 2, 1],
    )

    # Etiquetar las líneas (-1, 0, 1)
    plt.clabel(contours, inline=True, fontsize=10, fmt="%1.1f")

    # 5. Dibujar Puntos
    plt.scatter(
        X[y == -1, 0],
        X[y == -1, 1],
        c="red",
        marker="s",
        edgecolors="k",
        label="Clase -1",
        s=40,
    )
    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        c="blue",
        marker="o",
        edgecolors="k",
        label="Clase 1",
        s=40,
    )

    # 6. RESALTAR VECTORES DE SOPORTE
    # En PEGASOS, son los que tienen alphas > 0
    sv_indices = np.where(model.alphas > 0)[0]
    plt.scatter(
        model.X_train[sv_indices, 0],
        model.X_train[sv_indices, 1],
        s=150,
        linewidth=1.5,
        facecolors="none",
        edgecolors="k",
        label="Vectores de Soporte",
    )

    plt.title(
        f"SVM Street & Support Vectors\n(Gamma={model.gamma}, Lambda={model.lambda_param})"
    )
    plt.legend(loc="upper right")
    plt.show()


# ==========================================
# 4. PRUEBA: DATOS NO LINEALES (LUNAS)
# ==========================================
if __name__ == "__main__":
    from sklearn.datasets import make_moons
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC

    # 1. Generar Datos NO LINEALES (Dos lunas entrelazadas)
    print("Generando Moons Dataset (No lineal)...")
    X, y = make_moons(n_samples=1_000, noise=0.1, random_state=42)

    # Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 2. Nuestro Kernel SVM (RBF)
    print("\nNumba Kernel SVM (RBF)...")
    # Lambda bajo = C alto (Harder margin)
    # Gamma alto = Radio de influencia pequeño (modelo más complejo)
    my_svm = KernelSVM(gamma=2.0, lambda_param=0.001, n_iters=10_000)

    t0 = time.time()
    my_svm.fit(X_train, y_train, kernelfunc=rbf_kernel)
    my_time = time.time() - t0

    my_pred = my_svm.predict(X_test)
    # Mapeo inverso de -1/1 a 0/1
    my_pred_mapped = np.where(my_pred == -1, 0, 1)
    my_acc = accuracy_score(y_test, my_pred_mapped)

    print(f"Tiempo: {my_time:.4f}s | Acc: {my_acc:.4f}")

    # 3. Sklearn SVC (Referencia)
    print("\nSklearn SVC (RBF)...")
    # Convertimos nuestros hiperparámetros a los de Sklearn
    # C = 1 / (n_samples * lambda) aprox
    C_equivalent = 1.0 / (len(X_train) * 0.001)

    sk_svm = SVC(kernel="rbf", gamma=2.0, C=C_equivalent)

    t0 = time.time()
    sk_svm.fit(X_train, y_train)
    sk_time = time.time() - t0

    sk_pred = sk_svm.predict(X_test)
    sk_acc = accuracy_score(y_test, sk_pred)

    print(f"Tiempo: {sk_time:.4f}s | Acc: {sk_acc:.4f}")

    print("-" * 40)

    # Visualización de los datos y los resultados
    y_train_svm = np.where(y_train <= 0, -1, 1)
    visualize_svm_street(my_svm, X_train, y_train_svm)
