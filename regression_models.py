from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
from correlation_vif import X_df
from data_preparation import y

def evaluate_model(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"\n{name}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    print(f"MAPE: {mape:.2f}")

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_lin = lin_reg.predict(X_test)

ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)

y_pred_ridge = ridge_reg.predict(X_test)

evaluate_model(y_test, y_pred_lin, "Линейная регрессия")
evaluate_model(y_test, y_pred_ridge, "Гребневая регрессия")

cv_scores_lin = cross_val_score(lin_reg, X_df, y, cv=5, scoring='r2')
cv_scores_ridge = cross_val_score(ridge_reg, X_df, y, cv=5, scoring='r2')

print("\nКросс-валидация (R²):")
print(f"Линейная регрессия: {cv_scores_lin.mean():.2f}")
print(f"Гребневая регрессия: {cv_scores_ridge.mean():.2f}")
