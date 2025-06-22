import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# Set the correct path to your CSV file
file_path = "C:/Users/mothi/Downloads/house-prices-advanced-regression-techniques/train.csv"
df = pd.read_csv(file_path)

# Preview the first 5 rows
df.head(10)


# Use GrLivArea (sqft), BedroomAbvGr (bedrooms), FullBath (bathrooms)
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")


# Predict price for: 2000 sqft, 3 bedrooms, 2 full baths
sample = [[2000, 3, 2]]
sample = pd.DataFrame([[2000, 3, 2]], columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])
predicted_price = model.predict(sample)
print("The Predicted price of the House is $",int(predicted_price[0]),".")


plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # 45-degree line
plt.show()


