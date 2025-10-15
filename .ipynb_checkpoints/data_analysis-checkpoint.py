import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Путь к файлу
csv_path = os.path.join("data", "Car_Price_Prediction.csv")

# Загрузка датасета
df = pd.read_csv(csv_path)

# 1. Размер и структура
print("Размер датасета:", df.shape)
print("\nТипы данных:\n", df.dtypes)

# 2. Первые и последние строки
print("\nПервые 5 строк:\n", df.head())
print("\nПоследние 5 строк:\n", df.tail())

# 3. Статистика по числовым признакам
print("\nСтатистика по числовым признакам:\n", df.describe())

# 4. Пропущенные значения
print("\nПропущенные значения:\n", df.isnull().sum())

# Настройка отображения графиков
sns.set(style="whitegrid")

# Числовые признаки
numeric_features = ['Year', 'Engine Size', 'Mileage', 'Price']

# Гистограммы для числовых признаков
for col in numeric_features:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Распределение признака {col}")
    plt.xlabel(col)
    plt.ylabel("Количество")
    plt.show()

# Категориальные признаки
categorical_features = ['Make', 'Model', 'Fuel Type', 'Transmission']

# Столбчатые диаграммы для категориальных признаков
for col in categorical_features:
    plt.figure(figsize=(8,4))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index)
    plt.title(f"Распределение признака {col}")
    plt.xlabel(col)
    plt.ylabel("Количество")
    plt.xticks()
    plt.show()

