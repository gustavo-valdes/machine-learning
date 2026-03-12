import numpy as np
import time
from RegressionTree import RegressionTree 

class MiniGradientBoosting:
    def __init__(self, n_estimators=20, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = 0.0

    def fit(self, X, y):
        # 1. Predicción Inicial (Base Score): El promedio
        # Esto es F_0(x)
        self.initial_pred = np.mean(y)
        
        # Vector de predicciones actuales
        y_pred = np.full(len(y), self.initial_pred)
        
        print(f"🚀 Iniciando Gradient Boosting ({self.n_estimators} rondas)...")
        
        for i in range(self.n_estimators):
            # 2. Calcular el Gradiente Negativo (Residuales)
            # r_i = y - F_{m-1}(x)
            residuals = y - y_pred
            
            # 3. Entrenar un 'Weak Learner' para predecir los residuales
            # Usamos tu RegressionTree optimizado con poca profundidad
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=10,
                n_feats=-1 # En boosting solemos mirar todas las features
            )
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # 4. Actualizar predicciones
            # F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            
            # Monitoreo (Opcional)
            train_mse = np.mean((y - y_pred)**2)
            if (i+1) % 5 == 0:
                print(f"   ✓ Ronda {i+1}: Train MSE bajó a {train_mse:.2f}")

    def predict(self, X):
        # Empezamos con el valor base
        y_pred = np.full(len(X), self.initial_pred)
        
        # Sumamos la contribución ponderada de cada árbol
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
            
        return y_pred

# ==========================================
# BENCHMARK DE BOOSTING
# ==========================================
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import GradientBoostingRegressor

    # 1. Datos (100k para prueba rápida de concepto de boosting)
    # Boosting es secuencial, así que tarda más que un solo árbol
    print("📊 Generando datos (100k filas, 20 feats)...")
    X, y = make_regression(n_samples=100_000, n_features=20, noise=5.0, random_state=42)
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 2. Nuestro Mini-GBoost
    print("\n🚀 Entrenando Mini-GBoost...")
    # Usamos 50 árboles de profundidad 3 (configuración clásica)
    my_gbm = MiniGradientBoosting(n_estimators=50, learning_rate=0.1, max_depth=3)
    
    start = time.time()
    my_gbm.fit(X_train, y_train)
    my_time = time.time() - start
    
    my_mse = mean_squared_error(y_test, my_gbm.predict(X_test))
    print(f"⏱️ Tiempo Propio: {my_time:.4f}s | MSE: {my_mse:.2f}")

    # 3. Comparación con Sklearn GBM
    print("\n🤖 Entrenando Sklearn GradientBoosting...")
    sk_gbm = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)
    
    start = time.time()
    sk_gbm.fit(X_train, y_train)
    sk_time = time.time() - start
    
    sk_mse = mean_squared_error(y_test, sk_gbm.predict(X_test))
    print(f"⏱️ Tiempo Sklearn: {sk_time:.4f}s | MSE: {sk_mse:.2f}")
    
    print("-" * 40)
    print(f"🏆 Velocidad relativa: {sk_time / my_time:.2f}x")