import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('salary.csv')

# Input and output
X = data[['YearsExperience']]
y = data['Salary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test set
y_pred = model.predict(X_test)

# Plot training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, model.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Plot test set
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, model.predict(X_train), color='blue')  # Same regression line
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


