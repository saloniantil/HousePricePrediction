# house_price_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# STEP 1: Create dummy dataset
data = {
    'Area': [1000, 1500, 1200, 1800, 2000, 2200, 1400, 1600],
    'Bedrooms': [2, 3, 2, 4, 3, 4, 3, 3],
    'Location': ['Suburb', 'City', 'Suburb', 'City', 'City', 'Suburb', 'City', 'Suburb'],
    'Price': [300000, 450000, 350000, 500000, 600000, 550000, 400000, 420000]
}
df = pd.DataFrame(data)

# STEP 2: Encode Categorical Data
df = pd.get_dummies(df, columns=['Location'], drop_first=True)

# Features and Target
X = df.drop('Price', axis=1)
y = df['Price']

# STEP 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# STEP 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 5: Predict
y_pred = model.predict(X_test)

# STEP 6: Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R² Score: {r2:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# STEP 7: Save Results to CSV
results = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': y_pred
})
results.to_csv('predicted_prices.csv', index=False)
print("\nResults saved to 'predicted_prices.csv'")

# STEP 8: Visualization
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, color='green', s=100)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.tight_layout()
plt.show()
