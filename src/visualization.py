import matplotlib.pyplot as plt

def plot_feature_importance(models, feature_names, model_names=None):
    """
    models: dict, где ключ — имя модели, значение — объект модели.
    feature_names: список названий признаков.
    model_names: словарь с отображаемыми именами (опционально).
    """
    if model_names is None:
        model_names = {name: name for name in models.keys()}

    plt.figure(figsize=(8, 4))
    for i, (name, model) in enumerate(models.items()):
        coef = model.coef_
        color = plt.cm.tab10(i)
        label = model_names.get(name, name)
        plt.bar([f"{f} ({label})" for f in feature_names], coef,
                alpha=0.7, color=color, label=label)

    plt.title('Коэффициенты моделей (важность признаков)')
    plt.ylabel('Вес')
    plt.legend()
    plt.grid(axis='y')
    plt.show()

def show_predictions(predictions, true_price):
    """Выводит предсказания в красивом формате."""
    print("\n Предсказания для вашего дома:")
    for name, price in predictions.items():
        print(f"  {name}: ${price:,.2f}")
    print(f" Теоретическая цена (без шума): ${true_price:,.2f}")