'''
#Gradient descent with Step as activation function and SME for loss:

Gradient descent is an optimization algorithm used to minimize some 
function by iteratively moving in the direction of steepest descent as 
defined by the negative of the gradient. In machine learning, we use 
gradient descent to update the parameters of our model. Parameters refer 
to coefficients in Linear Regression and weights in neural networks.

We will use the STEP FUNCTION here
#--------------------------------------------#

#1 - Get the current Hypotesis for the data (the linear function)
#2 - Calc the current loss for b and w1 (Where we'll use MSE function for that)
#3 - Use loss to calcule the derivative of b and w1
#4 - Update the b and w1 with the new data got from derivative
#5 - Repeate de process until minimize the error
#NOTE: This algorithm (formula) works with 2 features only 
#NOTE: MSE is used for continuous values, not for classificatoion!!!


#--------------------------------------------#

### Equations ###
#Calc the Y_hat (Hypotesis): 
# Y_hat = b + w1 * x
# OR W1* x + b

##Function MSE Cost/loss = 
# Sum of: ((y_hat - y) ** 2) / N 

##Error function:
#Eb = y_hat - y
#Ew1 = (y_hat - y) * x

##Derivatives function: 
#Derivative of w1: Dw1 = w1 - learning_rate * (1/N) * Ew1
#Derivative of b: Db = b - learning_rate * (1/N) * Eb 
'''

import  matplotlib.pyplot as plt 
import numpy as np

#Initializing data
learning_rate = 0.01 
epochs = 1500
b = 5
w1 = 1
costs = np.zeros(epochs)
        

#Hypotesis function
def y_hat(x, w1, b):
    #Y_hat = b + w1 * x 
    # OR W1* x + b
    return b + w1 * x    

#Cost/loss function
#MSE Cost/loss = Sum of: ((y_hat - y) ** 2) / N
# N = Number of elemets
def MSE(X, y, w1, b):         
    cost = 0.0
    m = float(len(X))
    for i, _y in enumerate(X):         
        cost += (y_hat(x[i], w1, b) - y[i]) **2  

    return cost/m 

#gradient descent step
def gradientDescent_stepFunction(X, y, w1, b):
    m = float(len(X))

    error_b = 0
    error_w1 = 0

    for i in range(0, len(X)):
        #Error function: 
        #Eb = y_hat - y
        #Ew1 = (y_hat - y) * x
        error_b += y_hat(X[i], w1, b) - y[i]
        error_w1 += (y_hat(X[i], w1, b) - y[i]) * X[i]

    #Derivatives function: 
    #Derivative of b: Db = b - learning_rate * (1/N) * Eb 
    #Derivative of w1: Dw1 = w1 - learning_rate * (1/N) * Ew1
    new_b = b - learning_rate * ((1/m) * error_b)
    new_w1 = w1 - learning_rate * (1/m) * error_w1

    return new_b,  new_w1
    

#gradient descent function
def gradient_descent(X, y, w1, b):
  
    #Call the gradient descent step for x epochs
    for i in range(epochs):
        #execute gradient descent step
         b, w1 = gradientDescent_stepFunction(X, y, w1, b)

         #calc loss/cost
         costs[i] = MSE(X, y, w1, b)

    return w1, b, costs

#Plot cost/loss
def plot_costs():
    fig, ax = plt.subplots()
    ax.plot(np.arange(epochs), costs, 'r')
    ax.set_xlabel('epochs')
    ax.set_ylabel('Cost')
    ax.set_title('MSE vs Epochs')
    plt.show()  

#Plot data
def plot_data(X, Y, b, w1, title):
    x_values = [i for i in range (int(min(X)) -1, int(max(X)) +2 )]  
    y_values = [y_hat(x, b, w1) for  x in x_values]      
    plt.plot(x_values, y_values, 'r')
    plt.title(title)
    plt.plot(X, Y, 'bo')
    plt.show() 


#dataset
x = [1, 3, 5, 10, 14, 16]
y = [1.0, 3.5, 5.0, 5.5, 7.0, 7.5]

x_test = [4, 7]
y_test = [4.0, 5.1]
##Call the functions

#gradient descent function
_w1, _b, costs = gradient_descent(x, y, w1, b)

##plot functions
#plot costs
plot_costs()

#plot data
plot_data(x, y, _b, _w1, 'train data')

#plot test data
plot_data(x_test, y_test, _b, _w1, 'test data')







            

