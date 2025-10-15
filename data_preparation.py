import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from data_analysis import df

df = df.dropna()

numeric_features = ['Year', 'Engine Size', 'Mileage']
categorical_features = ['Make', 'Model', 'Fuel Type', 'Transmission']

X = df.drop("Price", axis=1)
y = df["Price"]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)

print(f"Исходные признаки: {X.shape}")
print(f"Предобработанные признаки: {X_processed.shape}")
