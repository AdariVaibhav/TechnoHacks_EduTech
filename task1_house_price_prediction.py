import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

url = "/content/kc_house_data.csv"
df = pd.read_csv(url)

plt.scatter(df['sqft_living'], df['price'])
plt.title('Square Footage vs House Price')
plt.xlabel('Square Footage of Living Area')
plt.ylabel('House Price')
plt.show()

corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
            'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
            'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

new_house_features = [3, 2.5, 2000, 5000, 2, 0, 1, 3, 7, 1800, 200, 1990, 0, 98001, 47.5, -122.2, 2200, 5500]
scaled_new_house_features = scaler.transform([new_house_features])
predicted_price = model.predict(scaled_new_house_features)

print(f'Predicted Price for the New House: ${predicted_price[0]:,.2f}')
