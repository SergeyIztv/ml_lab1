from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import numpy as np
from data_preparation import X_processed, y

# 1. Разделяем данные на train/test
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 2. Применяем PCA — оставим столько компонент, чтобы сохранялось 95% дисперсии
pca = PCA(n_components=0.99)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"Исходное количество признаков: {X_train.shape[1]}")
print(f"После PCA: {X_train_pca.shape[1]} компонентов")

# 3. Линейная регрессия после PCA
lin_reg_pca = LinearRegression()
lin_reg_pca.fit(X_train_pca, y_train)
y_pred_lin_pca = lin_reg_pca.predict(X_test_pca)

# 4. Гребневая регрессия после PCA
ridge_reg_pca = Ridge(alpha=1.0)
ridge_reg_pca.fit(X_train_pca, y_train)
y_pred_ridge_pca = ridge_reg_pca.predict(X_test_pca)

# 5. Оценка качества
def evaluate_model(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"\n{name}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    print(f"MAPE: {mape:.2f}")

evaluate_model(y_test, y_pred_lin_pca, "Линейная регрессия (PCA)")
evaluate_model(y_test, y_pred_ridge_pca, "Гребневая регрессия (PCA)")