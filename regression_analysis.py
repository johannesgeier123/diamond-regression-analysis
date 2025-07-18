# Import necessary libraries for data manipulation, visualization, and modeling
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the diamond dataset into a DataFrame
dataFrame = pd.read_csv("diamond_dataset.csv")

# Compute the covariance and correlation matrices to understand relationships between variables
covarianceMatrix = dataFrame.cov()
correlationMatrix = dataFrame.corr()

# Display the covariance and correlation matrices
print("This is the Covariance Matrix")
print(covarianceMatrix)

print("----------------------------------------------------------------------")

print("This is the Correlation Matrix")
print(correlationMatrix)

# Create a new feature: square root of the 'carat' variable
carat_root = np.sqrt(dataFrame["carat"])
dataFrame["carat_root"] = carat_root

# Create another new feature: logarithm of the 'price' variable
log_price = np.log(dataFrame["price"])
dataFrame["log_price"] = log_price

# Convert categorical 'cut' variable to numeric values for modeling
cut_num = dataFrame["cut"].map({"Ideal": 1, "Premium": 2, "Very Good": 3, "Good": 4, "Fair": 5})
dataFrame["cut_num"] = cut_num

print("----------------------------------------------------------------------")

# Create a scatter plot of carat_root vs. log_price, colored by cut_num
plt.scatter(dataFrame["carat_root"], dataFrame["log_price"], c=dataFrame["cut_num"], cmap="rainbow")
plt.xlabel("Carat Root")
plt.ylabel("Log Price")
plt.legend(dataFrame["cut"], loc='lower right')  # May throw warning if too many labels
plt.colorbar()  # Adds a color legend based on 'cut_num'
plt.show()

# Prepare data for regression model: single feature (carat_root)
X_column = dataFrame["carat_root"]
X = X_column.values.reshape(-1,1)
Y = dataFrame["log_price"]

# Split the data into training and testing sets (60% train, 40% test)
X_trainData, X_testData, Y_trainData, Y_testData = train_test_split(X, Y, test_size=0.4, random_state=0)

# Initialize linear regression model
linear_Model = LinearRegression()

# Fit the model to training data
linear_Model.fit(X_trainData, Y_trainData)

# Predict target values using the test data
predicted_value_of_Y = linear_Model.predict(X_testData)

# Plot histograms to compare actual vs. predicted values
plt.hist(Y_testData, bins=20, alpha=0.5, label="Actual values of the Log Price", color="blue")
plt.hist(predicted_value_of_Y, bins=20, alpha=0.5, label="Predicted values of the Log Price", color="green")
plt.xlabel("Log Price")
plt.ylabel("Frequency") 
plt.title("Predicted values vs. Actual values")
plt.legend()
plt.show()

# Calculate and display the R² value (model performance metric)
r_Square = linear_Model.score(X_testData, Y_testData)
print(" ")
print(" ")
print("This is the value for R-Square: ", r_Square)

# Add a second independent variable (cut_num) to the model
X2 = dataFrame["cut_num"]
X_twoIndepenteVariables = np.column_stack((X_column, X2))  # Combine carat_root and cut_num into a 2D array

# Split the new dataset into training and testing sets
X_twoIndepenteVariables_trainData, X_twoIndepenteVariables_testData, Y_trainData, Y_testData = train_test_split(X_twoIndepenteVariables, Y, test_size=0.4, random_state=0)

# Fit the model again using both independent variables
linear_Model.fit(X_twoIndepenteVariables_trainData, Y_trainData)
prediction_of_Y = linear_Model.predict(X_twoIndepenteVariables_testData)

# Calculate new R² value for the multivariate model
r_Square_No2 = linear_Model.score(X_twoIndepenteVariables_testData, Y_testData)

print("----------------------------------------------------------")
print("R-Square Number 2: ", r_Square_No2)

# Compare performance of single-variable vs. multi-variable models
if r_Square > r_Square_No2:
    print("Adding a second independent variable doesn't lead to a better fit to the model")
elif r_Square == r_Square_No2:
    print("Adding an extra independent variable doesn’t make any difference")
else:
    print("Adding an extra independent variable does improve the model")
