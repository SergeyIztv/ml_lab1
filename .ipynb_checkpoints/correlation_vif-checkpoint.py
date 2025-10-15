import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

from data_preparation import X_processed, preprocessor, y

# Создаём DataFrame из предобработанных признаков
X_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())

# Добавим целевую переменную
df_corr = X_df.copy()
df_corr["Price"] = y

# Матрица корреляций
corr_matrix = df_corr.corr()

# Визуализация
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Матрица корреляций")
plt.show()

# Рассчитываем VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]

print(f"VIF:\n{vif_data}")