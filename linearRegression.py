import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model
import statsmodels.api as sm

# Creation of function of min_sq

def min_sq(x,y):
    x_bar, y_bar = np.mean(x), np.mean(y)
    beta_1 = np.sum((x-x_bar)*(y-y_bar))/np.sum((x-x_bar)**2)
    beta_0 = y_bar - beta_1*x_bar
    
    return [beta_0, beta_1]

# Generation of data

N = 100
a = np.random.normal(loc=1, scale=1, size=N)

b=  np.random.randn(1)

x = np.random.randn(N)

y = a + b*x + np.random.randn(N)

a1, b1 = min_sq(x,y)    

xx = x - np.mean(x)
yy = y - np.mean(y)
a2, b2 = min_sq(xx,yy)

# prediction

x_seq = np.arange(-3,3,0.1)

y_pred = a1 + b1*x_seq
y_pred2 = a2 + b2*x_seq

# Plot 

plt.scatter(x,y, c='black') # plot of the points
plt.axhline(y=0, c='black') # plot of the line
plt.axvline(x=0, c='black') # plot of the line
plt.plot(x_seq, y_pred, c='red') # plot of the line
plt.plot(x_seq, y_pred2, c='blue') # plot of the line
plt.close()



## This program estimates the intercept and the slope via the least squares method for a linear regression model

n = 100
beta = np.array([1,2,3])
x = np.random.randn(n,2)
y = beta[0] + beta[1]*x[:,0] + beta[2]*x[:,1] + np.random.randn(n)
X = np.insert(x,0,1,axis=1)

# Resolving the linear system

beta_hat = np.linalg.solve(np.dot(X.T,X), np.dot(X.T,y))

# For each degree of freedom up to m for chi-squared distribution, we depict the probability density function 
plt.figure(figsize=(6, 4))
x = np.arange(0,20,0.1) # sequence of values for 0 up to 20 with step 0.1

for i in range(0,10):
    plt.plot(x, stats.chi2.pdf(x,i), label='df='+str(i))
plt.legend(loc='upper right')
plt.close()

"""
 We will plot the distribution of student t for different degrees of freedom
We will show that when the degree of freedom increases, the distribution of student t converges to the normal distribution
"""

x = np.arange(-10,10,0.1) # sequence of values for -10 up to 10 with step 0.1

plt.figure(figsize=(6, 4))

plt.plot(x, stats.norm.pdf(x,0,1), label = 'normal',c='black',linewidth=2)

for i in range(1,11):
    plt.plot(x, stats.t.pdf(x,i), label='df='+str(i), linewidth=0.8)
    
plt.legend(loc='upper right')
plt.title(" changes of the distribution of student t for different degrees of freedom")
plt.close()
"""
 We wish to perform  a hypothesis test for a null hypothesis H0: beta_j = 0 j=0,1 
 We will use the t-statistic of student t distribution
 
"""
N = 100
x = np.random.randn(N)
y = np.random.randn(N)

beta_1, beta_0 = min_sq(x,y)

RSS = np.sum((y-beta_0-beta_1*x)**2)
RSE = np.sqrt(RSS/(N-1-1))

B_0 = (x.T.dot(x)/N)/(np.sum((x-np.mean(x))**2))
B_1 = 1/(np.sum((x-np.mean(x))**2))

se_0 = RSE*np.sqrt(B_0)
se_1 = RSE*np.sqrt(B_1)

t_0 = beta_0/se_0
t_1 = beta_1/se_1

p_0 = 2*(1-stats.t.cdf(np.abs(t_0),N-1))
p_1 = 2*(1-stats.t.cdf(np.abs(t_1),N-1))


"""
 We do same thing in python
"""

reg = linear_model.LinearRegression()

x = x.reshape(-1,1) 
y = y.reshape(-1,1) # we reshape the data to have a matrix of dimension (N,1)

reg.fit(x,y) 

X = np.insert(x,0,1,axis=1)

model = sm.OLS(y,X)

results = model.fit()

# We repeat the estimation of \beta_1 one thousand times (r = 1000) to construct the histogram of the distribution

N = 100

r = 1000

T = []

for i in range(r):
    x = np.random.randn(N)
    y = np.random.randn(N)
    beta_1, beta_0 = min_sq(x,y)
    pre_y = beta_0 + beta_1*x
    RSS = np.sum((y-pre_y)**2)
    RSE = np.sqrt(RSS/(N-1-1))
    B_0 = (x.T.dot(x)/N)/(np.sum((x-np.mean(x))**2))
    B_1 = 1/(np.sum((x-np.mean(x))**2))
    se_1 = RSE*np.sqrt(B_1)
    T.append(beta_1/se_1)

# plot the histogram of the distribution

plt.hist(T, bins=20, density=True, alpha=0.6, range=[-3,3],color='g')
x = np.linspace(-4,4,100) # sequence of values for -4 up to 4 with step 0.1
plt.plot(x, stats.t.pdf(x,98))
plt.close()

"""
The function that calculate the determination coefficient
"""

def R_2(x,y):
    n = x.shape[0] # number of observations
    xx =np.insert(x,0,1,axis=1)
    beta_hat = np.linalg.solve(np.dot(xx.T,xx), np.dot(xx.T,y))
    y_hat = np.dot(xx,beta_hat)
    y_bar = np.mean(y)
    RSS = np.sum((y-y_hat)**2)
    TSS = np.sum((y-y_bar)**2)
    R_2 = 1 - RSS/TSS
    return R_2

"""
 This function calculates the VIF (Variance Inflation Factor) for each variable 
"""

def vif(x):
    values = []
    for i in range(x.shape[1]) :
        S = delete(x,i,1)
        values.append(1/(1-R_2(S,x[:,i])))
    return values

"""
     Less draw the interval of confidence that surrounds the line and prediction interval
"""

N = 100
x = np.random.randn(N,1) # we reshape the data to have a matrix of dimension (N,1)
print(x)
X = np.insert(x,0,1,axis=1)

beta = np.array([1,2]) # beta_0 = 1 and beta_1 = 2
epsilon = np.random.randn(N)
y = np.dot(X,beta) + epsilon

U = np.linalg.inv(np.dot(X.T,X))

beta_hat = np.dot(U,np.dot(X.T,y))

# for the interval of confidence, we need to compute the RSS and the RSE

RSS= np.sum((y-np.dot(X,beta_hat))**2)

RSE = np.sqrt(RSS/(N-2))
alpha = 0.05

# The function that draw the interval of confidence 

def f(x, a) : # a = 0 means confidence interval and a = 1 means prediction interval
    x = np.array([1,x])
    range = stats.t.ppf(1-alpha/2,N-2)*RSE*np.sqrt(a + np.dot(x,np.dot(U,x.T)))
    lower = np.dot(x,beta_hat) - range
    upper = np.dot(x,beta_hat) + range
    return ([lower, upper])

## Draw an example of interval of confidence

x_seq = np.arange(-3,3,0.1)

lower_x = []
upper_x = []  

lower_xp = []
upper_xp = []  

for i in x_seq:
    # Confidence interval
    lower_x.append(f(i,0)[0])
    upper_x.append(f(i,0)[1])
    # Prediction interval
    lower_xp.append(f(i,1)[0])
    upper_xp.append(f(i,1)[1])
    
# prediction value by regression

yy = beta_hat[0] + beta_hat[1]*x_seq

# Show plot

plt.scatter(x,y, c='black') # plot of the points
plt.axhline(y=0, c='black') # plot of the line
plt.axvline(x=0, c='black') # plot of the line
plt.xlim(np.min(x_seq),np.max(x_seq))
plt.ylim(np.min(lower_xp),np.max(upper_xp))
plt.plot(x_seq, yy, c='red') # plot of the line
plt.plot(x_seq, lower_x, c='blue', linestyle = "dashed") # plot of the line
plt.plot(x_seq, upper_x, c='blue', linestyle ="dashed") # plot of the line
plt.xlabel('x')
plt.ylabel('y')

plt.close()


    
    


