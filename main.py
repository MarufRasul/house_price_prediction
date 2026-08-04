import sys
import os
# Добавляем папку src в путь
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_generation import generate_house_data
from model_training import ModelTrainer
from predict import predict_price, get_user_input, true_price_formula
from visualization import plot_feature_importance, show_predictions

def main():
    print("=" * 60)
    print(" ПРОЕКТ: ПРЕДСКАЗАНИЕ ЦЕНЫ ДОМА")
    print("=" * 60)

    # 1. Генерация данных
    df = generate_house_data(n_samples=100, seed=42)
    X = df[['size', 'bedrooms', 'age']]
    y = df['price']

    # 2. Подготовка и обучение
    trainer = ModelTrainer()
    trainer.prepare_data(X, y)
    lr_model = trainer.train_linear_regression()
    ridge_model = trainer.train_ridge(alpha=1.0)

    # 3. Сравнение
    best_model_name = trainer.compare_models()

    # 4. Интерактивное предсказание
    size, bedrooms, age = get_user_input()
    true_price = true_price_formula(size, bedrooms, age)

    # Предсказания обеих моделей
    predictions = {
        'Linear Regression': predict_price(lr_model, trainer.scaler, size, bedrooms, age),
        'Ridge Regression': predict_price(ridge_model, trainer.scaler, size, bedrooms, age)
    }
    show_predictions(predictions, true_price)

    # 5. Визуализация важности признаков
    feature_names = ['size', 'bedrooms', 'age']
    models = {'Linear': lr_model, 'Ridge': ridge_model}
    plot_feature_importance(models, feature_names)

    print("\n Проект завершён!")

if __name__ == "__main__":
    main()