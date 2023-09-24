import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


file_path = 'data_dm3.csv'
data = pd.read_csv(file_path, sep=',', header=None)

# last column is the output
y = data.iloc[:, -1]

# all columns without the last one are the features
x = data.iloc[:, :-1]

# Check if the explanatory variables are centered

means= x.mean()

# check if means is close to 0


centered_variables = means.abs() < 1e-10

# Check if the explanatory variables are standardized

standard_deviations = x.std()

# check if standard deviations is close to 1

standardized_variables = abs(standard_deviations-1) < 1e-10

# standardize the variables

X = (x - means)/standard_deviations

# check if the output y is centered
means_y = y.mean()
centered_y = abs(means_y) < 1e-10
Y = (y - means_y)/y.std()

# sample four explanatory variables

X_sample = X.iloc[:, 0:4]

random_index = np.random.choice(X.columns, 4)
random_vars = X[random_index]

# Compute the scatter plot of random_vars against Y
for column in random_vars.columns:
    plt.scatter(random_vars[column], Y, label=column)
    
# Add labels and a legend
plt.xlabel("Explanatory Variables")
plt.ylabel("Output (Y)")
plt.legend()
plt.title("Scatter Plot of Explanatory Variables vs. Output")

# Save the plot to a file
plt.savefig("output_plot.png")
# Provide the scatter plot

data_scatter = pd.concat([random_vars, Y], axis=1)

# Close the plot (optional)
plt.close()
# Create a training and a test set
# Assuming X is your explanatory variables and Y is your output variable
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Check the sizes of the two samples
print("Size of X_train:", X_train.shape)
print("Size of X_test:", X_test.shape)

# Covariance matrix of X_train

covariance_matrix = X_train.cov()

# calculate the eigenvalues of the covariance matrix

eigenvalues = np.linalg.eigvals(covariance_matrix)

# sorted the eigenvalues in descending order

sorted_eigenvalues = np.sort(eigenvalues)[::-1]

# create the plot of the eigenvalues
# Create a plot of the sorted eigenvalues
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(sorted_eigenvalues) + 1), sorted_eigenvalues, marker='o', linestyle='-')
plt.title('Eigenvalues of Covariance Matrix (Descending Order)')
plt.xlabel('Eigenvalue Index')
plt.ylabel('Eigenvalue Magnitude')
plt.grid(True)
plt.savefig('eigenvalues.png')
plt.close()


# Exercice 2 - Ridge Regression Compute the PCA of the training set before the OLS regression.

# standardize the training set

X_train = (X_train - X_train.mean())/X_train.std()
X_test = (X_test - X_test.mean())/X_test.std()

# Compute the PCA of the training set and choose the 52 components.

pca = PCA(n_components=52)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Compute the OLS regression of Y_train on X_train_pca

ols_model = LinearRegression()

ols_model.fit(X_train_pca, Y_train)

pca_ols_predictions = ols_model.predict(X_test_pca)

# plot the coefficients of the OLS regression

plt.figure(figsize=(10, 6))
plt.bar(range(len(ols_model.coef_)), ols_model.coef_)
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.title('Coefficients of Linear Regression (excluding intercept)')
plt.grid(True)
plt.savefig('coefPCARegNoInter.png')
plt.close()

# Classical OLS regression without PCA

ols_classical = LinearRegression()

ols_classical.fit(X_train, Y_train)
plt.figure(figsize=(10, 6))
plt.bar(range(len(ols_classical.coef_)), ols_classical.coef_)
plt.xlabel('coefficient Index')
plt.ylabel('coefficient value')
plt.title('coefficients of classical Regression (excluding intercept)')
plt.grid(True)
plt.savefig('coefClassicalOls.png')
plt.close()

# Calculate the residuals of text sample before the OLS regression

residuals_pca = Y_test - pca_ols_predictions

# Create a histogram of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals_pca, bins=20, density=True, alpha=0.6, color='b')

# Add labels and title

plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Density Histogram of Residuals')
plt.grid(True)
plt.savefig('residualsPca.png')
plt.close()

# Calculate the residuals of the test sample in the classical OLS regression

ols_predictions = ols_classical.predict(X_test)
residuals_classical = Y_test - ols_predictions

# Create a histogram of the residual

plt.figure(figsize=(10, 6))
plt.hist(residuals_classical, bins=20, density=True, alpha=0.6, color='b')

# Add labels and title

plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Density Histogram of Residuals')
plt.grid(True)
plt.savefig('residualsClassical.png')
plt.close()

# Calcule the determination coefficient for the test sample.

r_squaredPca = r2_score(Y_test,pca_ols_predictions)

# Calcule the determination coefficient foer the test sample classical ols

r_square = r2_score(Y_test, ols_predictions)










if __name__ == '__main__':
   # print(data.head())
   # print(data.shape[0])
   # print(data.shape[1])
   # print(y)
   # print(x)
   # print(means)
   # print(centered_variables)
   # print(standardized_variables)
   # print(X)
   # print(random_vars)
   # print(means_y)
   # print(covariance_matrix)
   # print(sorted_eigenvalues)
   print("R-squared (Coefficient of Determination):", r_squaredPca)
   print('Done!')
   print(data_scatter.head())
   sns.pairplot(data_scatter)
   plt.legend()
   plt.title("Scatter Plot of Variables")
   plt.savefig("scatter_plot.png") 
