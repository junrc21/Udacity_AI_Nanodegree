'''
#Gradient descent with Sigmoid as activation function and cross-entropy for loss:

Gradient descent is an optimization algorithm used to minimize some 
function by iteratively moving in the direction of steepest descent as 
defined by the negative of the gradient. In machine learning, we use 
gradient descent to update the parameters of our model. Parameters refer 
to coefficients in Linear Regression and weights in neural networks.

We will use the SIGMOID FUNCTION here
#--------------------------------------------#

#1 - Get the current Hypotesis for the data classifing between 0 and 1(probabilities)
#2 - Calc the current loss for b and w1 (Where we'll use MSE function for that)
#3 - Use loss to calcule the derivative of b and w1
#4 - Update the b and w1 with the new data got from derivative
#5 - Repeate de process until minimize the error
#NOTE: This algorithm (formula) works with 2 features only 

#----------------------------------------------#

### Equations ###
#Calc the Y_hat (Hypotesis): 
# Y_hat = b + w1 * x

# Sigmoid equation:
# Sig = 1 / (e^-y_hat)

##Function Cross-entropy Cost/loss = 
# -Sum of: (y * log2(y_hat) + (1 - y) * log2(1 - y_hat)) / N
# at the end, multiply the result for -1 to avoid negative numbers

##Error function:
#Eb = y_hat - y
#Ew1 = (y_hat - y) * x

##Derivatives function: 
#Derivative of w1: Dw1 = w1 - learning_rate * Ew1
#Derivative of b: Db = b - learning_rate  Eb 
'''


import  matplotlib.pyplot as plt 
import numpy as np

#Initializing data
learning_rate = 0.01 
epochs = 20
b = 5
w1 = 1
costs = []
        

def Sigmoid_function(w1, x, b):
    return 1 / (1 + np.exp(-y_hat(x, b, w1)))

#Hypotesis function
def y_hat(w1, x, b):
    #Y_hat = b + w1 * x 
    # OR W1* x + b
    return w1 * x + b   

##Function Cross-entropy Cost/loss = 
# -Sum of: (y * log2(y_hat) + (1 - y) * log2(1 - y_hat)) / N
# at the end, multiply the result for -1 to avoid negative numbers
def cross_entropy(X, Y, b, w1):
    CE = 0.0
    m = float(len(X))
    for i, x in enumerate(X):      
            CE += ( (y[i] * np.log2(Sigmoid_function(x, b, w1)))  + ( 1 - y[i]) * np.log2(1 - Sigmoid_function(x, b, w1))) 

    return  (CE/m) * -1 


#gradient descent step
def gradient_descent_step(X, y, w1, b):
    
    error_b = 0
    error_w1 = 0

    for i in range(0, len(X)):
        #Error function: 
        #Eb = y_hat - y
        #Ew1 = (y_hat - y) * x
        error_b += Sigmoid_function(X[i], b, w1) - y[i]
        error_w1 += (Sigmoid_function(X[i], b, w1) - y[i]) * X[i]

    #Derivatives function: 
    #Derivative of b: Db = b - learning_rate * (1/N) * Eb 
    #Derivative of w1: Dw1 = w1 - learning_rate * (1/N) * Ew1
    new_b = b - learning_rate * error_b
    new_w1 = w1 - learning_rate * error_w1

    return new_w1, new_b

#Logistic function
def logistic_function(X, Y, w1, b):
  
    #Call the gradient descent step for x epochs
    for i in range(epochs):
        
        #execute gradient descent step
        w1, b = gradient_descent_step(X, y, w1, b) 
            
        #calc loss/cost
        loss = cross_entropy(X, Y, b, w1)
        costs.append(loss)

    return w1, b, costs            
             

#Plot cost/loss
def plot_costs():
    fig, ax = plt.subplots()
    ax.plot(np.arange(epochs), costs, 'r')
    ax.set_xlabel('epochs')
    ax.set_ylabel('Cost')
    ax.set_title('Cross-Entropy vs Epochs')
    plt.show()  

#Plot data
def plot_data(X, Y, w1, b, title):
    x_values = [i for i in range (int(min(X)) -1, int(max(X)) +2 )]  
    y_values = [Sigmoid_function(x, b, w1) for  x in x_values]          
    plt.plot(x_values, y_values, 'r')
    plt.title(title)
    plt.plot(X, Y, 'bo')
    plt.show() 


#dataset
x = [5.0, 7.0, 1.5, 4.5, 9.0, 3.0]
y = [0, 1, 1, 0, 1, 0]

x_test = [1.0, 7.0, 5.5, 8.5]
y_test = [0, 1, 0, 1]


#gradient descent function
_w1, _b, costs = logistic_function(x, y, w1, b)

#plot costs
plot_costs()

#plot data
plot_data(x, y, _w1, _b, 'train dataset')

#plot test data
plot_data(x_test, y_test, _w1, _b, 'test dataset')









            

