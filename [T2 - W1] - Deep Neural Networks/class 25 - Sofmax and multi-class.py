import  matplotlib.pyplot as plt 
import numpy as np

#Initializing data
learning_rate = 0.01 
epochs = 100
b = 5
w1 = 1
        

# Softmax equation:
# SM = e^x1 / (e^x1 + e^x2 + e^x3 + e^Xn)

def softmax(Z):
    expZ = np.exp(Z)
    sumExpZ = sum(expZ)
    result = []
    for z in expZ:
        result.append(z/sumExpZ)
    return result

def Sigmoid_function(w1, x, b):
    return 1 / (1 + np.exp(-y_hat(w1, x, b)))

#Hypotesis function
def y_hat(w1, x, b):
    #Y_hat = b + w1 * x 
    # OR W1* x + b
    return w1 * x + b   

##Function Cross-entropy Cost/loss = 
# -Sum of: (y * log2(y_hat) + (1 - y) * log2(1 - y_hat)) / N
# at the end, multiply the result for -1 to avoid negative numbers
def cross_entropy(Prob, Y, b, w1):
    CE = 0.0
    m = float(len(Y))  
    for i, p in enumerate(Prob):   
        x =  ( (Y[i] * np.log2(p)))
        CE += x 

    return -CE  


#gradient descent step
def gradient_descent_step(X, y, w1, b):
    m = float(len(X))

    error_b = 0
    error_w1 = 0

    for i in range(0, len(X)):
        #Error function: 
        #Eb = y_hat - y
        #Ew1 = (y_hat - y) * x
        error_b += Sigmoid_function(w1, x[i], b) - y[i]
        error_w1 += (Sigmoid_function(w1, x[i], b) - y[i]) * X[i]

    #Derivatives function: 
    #Derivative of b: Db = b - learning_rate * (1/N) * Eb 
    #Derivative of w1: Dw1 = w1 - learning_rate * (1/N) * Ew1
    new_b = b - learning_rate * ((1/m) * error_b)
    new_w1 = w1 - learning_rate * (1/m) * error_w1

    return new_w1, new_b

#Logistic function
def logistic_function(X, Y, w1, b):
    costs = []
    #Call the gradient descent step for x epochs
    for i in range(epochs):
        
        #execute gradient descent step
        w1, b = gradient_descent_step(X, Y, w1, b) 

        logits = [Sigmoid_function(w1, x, b) for x in X] 
        prob = softmax(logits)

        #calc loss/cost
        loss = cross_entropy(prob, Y, b, w1)
        costs.append(loss)

    return w1, b, costs            
             

#Plot cost/loss
def plot_costs(loss):
    fig, ax = plt.subplots()
    ax.plot(np.arange(epochs), loss, 'r')
    ax.set_xlabel('epochs')
    ax.set_ylabel('Cost')
    ax.set_title('Cross-Entropy vs Epochs')
    plt.show()  



#Plot data
def plot_data(X, Y, w1, b, title):
    x_values = [i for i in range (int(min(X)) -1, int(max(X)) +2 )]  
    y_values = [Sigmoid_function(w1, x, b) for  x in x_values]          
    plt.plot(x_values, y_values, 'r')
    plt.title(title)
    plt.plot(X, Y, 'bo')
    plt.show() 


#dataset 
#Students score = x
#Result = y

x = [5.0, 7.0, 1.5, 4.5, 9.0, 3.0, 5.5, 9.5, 2.5, 3.5, 7.2, 4.5, 6.3] #score
#y = [1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 2, 1, 2] #0 - Not Approved, 1 - Recuperation, 2 - Approved 

y_approved =     [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
y_Notapproved =  [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0]
y_Recuperation = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]

y_test = [7.0, 1.0, 5.0]
x_test = [2, 0, 1]

#gradient descent function
w1, b, costs1 = logistic_function(x, y_approved, w1, b)
w1, b, costs2 = logistic_function(x, y_Notapproved, w1, b)
w1, b, costs3 = logistic_function(x, y_Recuperation, w1, b)

#plot costs
plot_costs(costs1)
plot_costs(costs2)
plot_costs(costs3)

#plot data
#plot_data(x, y, _w1, _b, 'train dataset')

#plot test data
plot_data(x_test, y_test, w1, b, 'test dataset')
#predict(x_test, y_test, _w1, _b)