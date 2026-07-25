import numpy as np
import pandas as pd


def generate_house_data():
    """
    Генерирует датасет о домах и возвращает pandas DataFrame.
    Если save_csv=True, сохраняет в файл 'data/houses.csv'.
    """
    n_samples=100, 
    seed=42, 
    save_csv=False
    np.random.seed(seed)
    n = n_samples

    size = np.random.uniform(500, 3000, n)
    bedrooms = np.random.randint(1, 6, n)
    age = np.random.uniform(0, 50, n)

    # Истинная формула + шум
    price = 50 * size + 10000 * bedrooms - 1000 * age + 50000 + np.random.normal(0, 20000, n)

    df = pd.DataFrame({
        'size': size,
        'bedrooms': bedrooms,
        'age': age,
        'price': price
    })

    if save_csv:
        import os
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/houses.csv', index=False)
        print(f" Данные сохранены в 'data/houses.csv'")

    return df

if __name__ == "__main__":
    # Если файл запущен напрямую, генерируем данные и показываем первые строки
    df = generate_house_data()
    print(" Первые 5 строк данных:")
    print(df.head())