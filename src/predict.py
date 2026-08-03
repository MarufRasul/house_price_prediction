import numpy as np
from sklearn.preprocessing import StandardScaler

def predict_price(model, scaler, size, bedrooms, age):
    """
    Принимает обученную модель, scaler и параметры дома,
    возвращает предсказанную цену.
    """
    X_new = np.array([[size, bedrooms, age]])
    X_new_scaled = scaler.transform(X_new)
    return model.predict(X_new_scaled)[0]

def get_user_input():
    """Интерактивный ввод параметров дома."""
    try:
        size = float(input("Введите площадь (кв.фт): "))
        bedrooms = int(input("Введите количество спален (1-5): "))
        age = float(input("Введите возраст (лет): "))
        return size, bedrooms, age
    except ValueError:
        print(" Ошибка ввода. Используем стандартные значения: 1500, 3, 10")
        return 1500, 3, 10

def true_price_formula(size, bedrooms, age):
    """Истинная формула для проверки (без шума)."""
    return 50 * size + 10000 * bedrooms - 1000 * age + 50000