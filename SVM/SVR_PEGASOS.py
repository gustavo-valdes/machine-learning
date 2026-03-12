import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange


# ==========================================
# 1. KERNELS (El motor geométrico)
# ==========================================
@njit(fastmath=True, cache=True)
def rbf_kernel(x1, x2, gamma):
    """Kernel Gaussiano: Similitud basada en distancia"""
    diff = x1 - x2
    dist_sq = np.dot(diff, diff)
    return np.exp(-gamma * dist_sq)


@njit(fastmath=True, cache=True)
def linear_kernel(x1, x2, gamma=0.0):
    """Kernel Lineal: Producto punto simple"""
    return np.dot(x1, x2)


@njit(fastmath=True, cache=True)
def poly_kernel(x1, x2, gamma=0.0, degree=3.0, c=1.0):
    """Kernel Polinomial: (x.y + c)^d"""
    return (np.dot(x1, x2) + c) ** degree


# ==========================================
# 2. ENTRENAMIENTO (PEGASOS Puro para w)
# ==========================================
@njit(fastmath=True)
def train_svr_weights(X, y, kernelfunc, betas, gamma, epsilon, lambda_param, n_iters):
    """
    Entrena solo los coeficientes 'betas' (la forma de la curva).
    El bias 'b' se calcula analíticamente después.
    """
    n_samples = X.shape[0]
    t = 1.0

    # Bucle de SGD
    for _ in range(n_iters):
        i = np.random.randint(0, n_samples)

        # 1. Predicción parcial (solo parte de w, sin b)
        # f_partial(x) = (1/lambda*t) * sum(beta * K(x, xi))
        scale = 1.0 / (lambda_param * t)
        prediction_w = 0.0

        for j in range(n_samples):
            if betas[j] != 0:
                prediction_w += betas[j] * kernelfunc(X[j], X[i], gamma)

        prediction_w *= scale

        # 2. Calcular Gradiente sobre la Hinge Loss Epsilon-Insensible
        # Error = Real - Predicho
        error = y[i] - prediction_w

        # Actualizar betas si violamos el tubo epsilon
        if error > epsilon:
            betas[i] += 1.0  # El punto "jala" hacia arriba
        elif error < -epsilon:
            betas[i] -= 1.0  # El punto "jala" hacia abajo

        t += 1.0

    return betas, t


# ==========================================
# 3. CÁLCULO DEL BIAS (Recuperación Analítica)
# ==========================================
@njit(fastmath=True)
def compute_bias(X, y, kernelfunc, betas, gamma, epsilon, lambda_param, t_final):
    """
    Calcula 'b' usando los Vectores de Soporte Libres.
    Teoría: En el margen, la predicción debe diferir del real exactamente en epsilon.
    b = y - (w*x) - epsilon * sign(beta)
    """
    scale = 1.0 / (lambda_param * t_final)
    n_samples = X.shape[0]

    sum_b = 0.0
    count = 0

    for i in range(n_samples):
        # Usamos puntos que son soporte (tienen peso)
        if np.abs(betas[i]) > 0:
            # Calcular parte de w (w*x)
            wx_i = 0.0
            for j in range(n_samples):
                if np.abs(betas[j]) > 0:
                    wx_i += betas[j] * kernelfunc(X[j], X[i], gamma)
            wx_i *= scale

            # Recuperar b candidato
            sign_beta = 1.0 if betas[i] > 0 else -1.0
            b_candidate = y[i] - wx_i - (epsilon * sign_beta)

            sum_b += b_candidate
            count += 1

    if count == 0:
        return 0.0

    # Promediamos todos los candidatos para estabilidad numérica
    return sum_b / count


# ==========================================
# 4. FUNCIONES AUXILIARES (Predicción y Costo)
# ==========================================
@njit(parallel=True, fastmath=True)
def predict_full(
    kernelfunc, X_test, X_train, betas, bias, gamma, lambda_param, t_final
):
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]
    preds = np.zeros(n_test)
    scale = 1.0 / (lambda_param * t_final)

    for i in prange(n_test):
        w_x = 0.0
        for j in range(n_train):
            if betas[j] != 0:
                w_x += betas[j] * kernelfunc(X_train[j], X_test[i], gamma)

        # Y = (w * x) + b
        preds[i] = (w_x * scale) + bias

    return preds


@njit(fastmath=True)
def calc_regularization_energy(X, kernelfunc, betas, gamma, lambda_param, t_final):
    """
    Calcula ||w||^2 = beta.T * K * beta
    Esta es la 'Energía' que el modelo intenta minimizar para ser suave.
    """
    n_samples = X.shape[0]
    scale = 1.0 / (lambda_param * t_final)
    norm_sq = 0.0

    # Doble sumatoria optimizada (solo betas != 0)
    for i in range(n_samples):
        if betas[i] != 0:
            for j in range(n_samples):
                if betas[j] != 0:
                    # K(xi, xj)
                    k_val = kernelfunc(X[i], X[j], gamma)
                    norm_sq += betas[i] * betas[j] * k_val

    # Ajustar por la escala al cuadrado
    return 0.5 * (scale**2) * norm_sq


# ==========================================
# 5. CLASE PRINCIPAL (Interfaz tipo Scikit-Learn)
# ==========================================
class SVR:
    def __init__(self, gamma=1.0, epsilon=0.1, C=1.0, n_iters=10000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.C = C
        # Relación teórica: Lambda ~ 1 / (C * N)
        # Usamos un factor de escala empírico para convergencia rápida en SGD
        self.lambda_param = 1.0 / (C * 100.0)
        self.n_iters = n_iters
        self.bias = 0.0
        self.betas = None
        self.X_train = None
        self.y_train = None
        self.t_final = 0.0
        self.kernelfunc = None

    def fit(self, X, y, kernelfunc):
        self.kernelfunc = kernelfunc
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        self.X_train = X
        self.y_train = y
        self.betas = np.zeros(len(X), dtype=np.float64)

        total_steps = self.n_iters * len(X)

        # 1. Optimizar w
        self.betas, self.t_final = train_svr_weights(
            X,
            y,
            self.kernelfunc,
            self.betas,
            self.gamma,
            self.epsilon,
            self.lambda_param,
            int(total_steps),
        )

        # 2. Calcular b
        self.bias = compute_bias(
            X,
            y,
            self.kernelfunc,
            self.betas,
            self.gamma,
            self.epsilon,
            self.lambda_param,
            self.t_final,
        )

    def predict(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        return predict_full(
            self.kernelfunc,
            X,
            self.X_train,
            self.betas,
            self.bias,
            self.gamma,
            self.lambda_param,
            self.t_final,
        )

    def get_model_stats(self):
        """Devuelve los componentes de la función objetivo."""
        # A. Término de Regularización (1/2 ||w||^2)
        reg_term = calc_regularization_energy(
            self.X_train,
            self.kernelfunc,
            self.betas,
            self.gamma,
            self.lambda_param,
            self.t_final,
        )

        # B. Término de Error (Loss)
        preds = self.predict(self.X_train)
        errors = np.abs(self.y_train - preds)
        # Loss insensible: max(0, |e| - epsilon)
        loss_contrib = np.maximum(0, errors - self.epsilon)
        total_loss = self.C * np.sum(loss_contrib)

        return {
            "Regularization (Smoothness)": reg_term,
            "Error Term (Data Fit)": total_loss,
            "Total Objective": reg_term + total_loss,
            "Support Vectors": np.sum(self.betas != 0),
            "Bias (Intercept)": self.bias,
        }


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        # Evitar división por cero si la desviación es 0
        self.scale[self.scale == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.scale

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        # Útil para recuperar la predicción real de 'y'
        return (X_scaled * self.scale) + self.mean


# ==========================================
# 6. DEMOSTRACIÓN VISUAL
# ==========================================
if __name__ == "__main__":
    # 1. Generar Datos con un poco de ruido para ver el efecto del tubo
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(60, 1), axis=0)
    y = np.sin(X).ravel() + 10.0
    # Añadimos ruido artificial a algunos puntos
    y[::5] += 1.5 * (np.random.rand(12) - 0.5)

    # 2. ESCALAR
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    print("🚀 Entrenando PEGASOS SVR con Numba...")

    # Parámetros del modelo
    EPSILON_SCALED = 0.3  # Epsilon en el espacio escalado
    svr = SVR(gamma=0.5, epsilon=EPSILON_SCALED, C=10.0, n_iters=10_000)
    svr.fit(X_scaled, y_scaled, rbf_kernel)

    # 3. Predicción para graficar (Malla fina)
    X_plot = np.linspace(0, 5, 200)[:, None]
    X_plot_scaled = scaler_x.transform(X_plot)

    # Predicción central (en escala normalizada)
    y_pred_scaled = svr.predict(X_plot_scaled)

    # 4. CALCULAR LA BANDA (En escala normalizada primero)
    # El tubo es y_pred ± epsilon
    y_upper_scaled = y_pred_scaled + EPSILON_SCALED
    y_lower_scaled = y_pred_scaled - EPSILON_SCALED

    # 5. DES-ESCALAR TODO (Volver al mundo real)
    # Recuperamos la predicción central
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Recuperamos los bordes de la banda
    y_upper_real = scaler_y.inverse_transform(y_upper_scaled.reshape(-1, 1)).ravel()
    y_lower_real = scaler_y.inverse_transform(y_lower_scaled.reshape(-1, 1)).ravel()

    ### Identificar Vectores de Soporte
    # Predecir sobre los datos de entrenamiento (escalados)
    y_train_pred_scaled = svr.predict(X_scaled)
    
    # Calcular el error absoluto de cada punto
    errores_absolutos = np.abs(y_scaled - y_train_pred_scaled)
    
    # Filtrar aquellos cuyo error sea mayor o igual a epsilon
    # Usamos una tolerancia de 1e-4 para evitar problemas de precisión de punto flotante
    tolerancia = 1e-4
    sv_indices = np.where(errores_absolutos >= (EPSILON_SCALED - tolerancia))[0]
    print(f"Número de Vectores de Soporte: {len(sv_indices)} de {len(X)}")
    
    # 6. GRAFICAR
    plt.figure(figsize=(10, 6))

    # a) Datos Reales
    plt.scatter(X, y, color="darkorange", label="Datos Entrenamiento", zorder=2)

    # b) Vectores de Soporte (resaltados)
    plt.scatter(
        X[sv_indices],
        y[sv_indices],
        s=100,
        facecolors="none",
        edgecolors="k",
        linewidth=1.5,
        label="Vectores de Soporte",
        zorder=3,
    )

    # c) Línea de Predicción
    plt.plot(
        X_plot, y_pred_real, color="navy", linewidth=2, label="Modelo SVR (PEGASOS)"
    )

    # d) BANDA EPSILON-INSENSIBLE
    # Rellenamos entre el límite superior e inferior real
    plt.fill_between(
        X_plot.ravel(),
        y_lower_real,
        y_upper_real,
        color="navy",
        alpha=0.2,
        label=rf"Banda Insensible ($\epsilon$={EPSILON_SCALED})",
    )

    plt.title(r"SVR con PEGASOS con Visualización del Tubo $\epsilon$-Insensible")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
