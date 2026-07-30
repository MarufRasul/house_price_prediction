import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

class ModelTrainer:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self, X, y):
        """Разделяет данные и масштабирует признаки."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.y_train, self.y_test = y_train, y_test
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        print(f"✅ Данные подготовлены: {len(X_train)} обучающих, {len(X_test)} тестовых объектов.")
        return self.X_train_scaled, self.X_test_scaled, y_train, y_test

    def train_linear_regression(self):
        """Обучает обычную линейную регрессию."""
        model = LinearRegression()
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        r2 = r2_score(self.y_test, y_pred)
        self.models['linear'] = model
        self.results['linear'] = {'r2': r2, 'coef': model.coef_, 'intercept': model.intercept_}
        return model

    def train_ridge(self, alpha=1.0):
        """Обучает гребневую регрессию с заданным alpha."""
        model = Ridge(alpha=alpha)
        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)
        r2 = r2_score(self.y_test, y_pred)
        self.models['ridge'] = model
        self.results['ridge'] = {'r2': r2, 'coef': model.coef_, 'intercept': model.intercept_}
        return model

    def compare_models(self):
        """Выводит сравнение моделей."""
        print("\n🔍 Сравнение моделей:")
        for name, res in self.results.items():
            print(f"  {name}: R² = {res['r2']:.4f}")
        best = max(self.results, key=lambda k: self.results[k]['r2'])
        print(f"\n🏆 Лучшая модель: {best} с R² = {self.results[best]['r2']:.4f}")
        return best