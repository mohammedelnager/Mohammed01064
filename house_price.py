import numpy as np 
import matplotlib.pyplot as plt 
 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, r2_score 
 
 
# ========================= 
# 1. Data 
# ========================= 
 
X = np.array([ 
    [80,  2, 20], 
    [100, 2, 15], 
    [120, 3, 10], 
    [150, 3, 8], 
    [180, 4, 5], 
    [200, 4, 3], 
    [220, 5, 2], 
    [250, 5, 1], 
    [300, 6, 1], 
    [350, 6, 0] 
]) 
 
y = np.array([ 
    250, 
    300, 
    350, 
    420, 
    500, 
    570, 
    630, 
    700, 
    850, 
    1000 
]) 
 
 
# ========================= 
# 2. Split 
# ========================= 
 
X_train, X_test, y_train, y_test = train_test_split( 
    X, 
    y, 
    test_size=0.2, 
    random_state=42 
) 
 
 
# ========================= 
# 3. Model 
# ========================= 
 
model = RandomForestRegressor( 
    n_estimators=100, 
    random_state=42 
) 
 
 
# ========================= 
# 4. Training 
# ========================= 
 
model.fit(X_train, y_train) 
 
 
# ========================= 
# 5. Prediction 
# ========================= 
 
y_pred = model.predict(X_test) 
 
 
# ========================= 
# 6. Evaluation 
# ========================= 
 
mse = mean_squared_error(y_test, y_pred) 
 
rmse = np.sqrt(mse) 
 
R2 = r2_score(y_test, y_pred) 
 
 
print("Actual =", y_test) 
print("Prediction =", y_pred) 
 
print("MSE =", mse) 
print("RMSE =", rmse) 
print("R2 =", R2) 
 
 
# ========================= 
# 7. Feature Importance 
# ========================= 
 
print("\nFeature Importance =") 
 
print(model.feature_importances_) 
 
 
# ========================= 
# 8. Actual vs Predicted 
# ========================= 
 
plt.figure() 
 
plt.scatter( 
    y_test, 
    y_pred 
) 
 
plt.xlabel("Actual Price") 
plt.ylabel("Predicted Price") 
 
plt.title("Actual vs Predicted House Prices") 
 
plt.grid(True) 
 
plt.show() 
 
 
# ========================= 
# 9. 3D Visualization 
# ========================= 
 
fig = plt.figure(figsize=(9, 7)) 
 
ax = fig.add_subplot(111, projection="3d") 
 
 
# Area → X axis 
# Rooms → Y axis 
# Price → Z axis 
 
ax.scatter( 
    X[:, 0], 
    X[:, 1], 
    y, 
    s=80 
) 
 
 
ax.set_xlabel("Area") 
ax.set_ylabel("Rooms") 
ax.set_zlabel("Price") 
 
ax.set_title("House Prices - 3D Visualization") 
 
plt.show() 