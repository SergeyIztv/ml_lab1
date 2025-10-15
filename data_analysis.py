import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = os.path.join("data", "Car_Price_Prediction.csv")

df = pd.read_csv(csv_path)

print("Размер датасета:", df.shape)
print("\nТипы данных:\n", df.dtypes)

print("\nПервые 5 строк:\n", df.head())
print("\nПоследние 5 строк:\n", df.tail())

print("\nСтатистика по числовым признакам:\n", df.describe())

print("\nПропущенные значения:\n", df.isnull().sum())

sns.set(style="whitegrid")

numeric_features = ['Year', 'Engine Size', 'Mileage', 'Price']

for col in numeric_features:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Распределение признака {col}")
    plt.xlabel(col)
    plt.ylabel("Количество")
    plt.show()

categorical_features = ['Make', 'Model', 'Fuel Type', 'Transmission']

for col in categorical_features:
    plt.figure(figsize=(8,4))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index)
    plt.title(f"Распределение признака {col}")
    plt.xlabel(col)
    plt.ylabel("Количество")
    plt.xticks()
    plt.show()

