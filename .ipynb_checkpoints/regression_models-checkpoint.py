import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np

from correlation_vif import X_df
from data_preparation import y

# 1. Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# 2. Линейная регрессия
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Предсказания
y_pred_lin = lin_reg.predict(X_test)

# 3. Оценка качества линейной регрессии
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
r2_lin = r2_score(y_test, y_pred_lin)
mape_lin = mean_absolute_percentage_error(y_test, y_pred_lin)

print("Линейная регрессия:")
print(f"RMSE: {rmse_lin:.2f}")
print(f"R²: {r2_lin:.2f}")
print(f"MAPE: {mape_lin:.2f}")

# 4. Гребневая регрессия
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)

y_pred_ridge = ridge_reg.predict(X_test)

rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)
mape_ridge = mean_absolute_percentage_error(y_test, y_pred_ridge)

print("\nГребневая регрессия:")
print(f"RMSE: {rmse_ridge:.2f}")
print(f"R²: {r2_ridge:.2f}")
print(f"MAPE: {mape_ridge:.2f}")

# 5. Кросс-валидация (по 5 фолдам)
cv_scores_lin = cross_val_score(lin_reg, X_df, y, cv=5, scoring='r2')
cv_scores_ridge = cross_val_score(ridge_reg, X_df, y, cv=5, scoring='r2')

print("\nКросс-валидация (R²):")
print(f"Линейная регрессия: {cv_scores_lin.mean():.2f}")
print(f"Гребневая регрессия: {cv_scores_ridge.mean():.2f}")
