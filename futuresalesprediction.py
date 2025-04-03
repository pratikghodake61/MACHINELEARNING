import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('advertising_data.csv')

# Display the first few rows of the DataFrame
print("DataFrame head:")
print(data.head())

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Option 1: Drop rows with missing values
data = data.dropna()

# Option 2: Fill missing values (uncomment the line below to use this option)
# data.fillna(data.mean(), inplace=True)

# Prepare the features and target variable
X = data[['TV', 'Radio', 'Newspaper']]  # Features (independent variables)
y = data['Sales']                        # Target variable (dependent variable)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()

# Predict future sales (example)
future_data = pd.DataFrame({
    'TV': [100, 200, 300],
    'Radio': [20, 30, 40],
    'Newspaper': [10, 20, 30]
})

future_predictions = model.predict(future_data)
print("Future Sales Predictions:")
for i, sales in enumerate(future_predictions, start=1):
    print(f'Predicted sales for future data point {i}: {sales:.2f}')