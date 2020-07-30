import numpy as np
import matplotlib.pyplot as plt 

class Perceptron:

    def __init__(self, learning_rate = 0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.activation_func = self._unit_step_function
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        #init weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        #This is to normalize the data, so we have to have only 0 and 1 values
        #So, if the i is > than 0, it could be 0.25 for example, it will be changed for 1
        #if the i is < than 0, it could be -0.1 for example, it will be changed for 0
        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                # Linear = (x1 * w1 + x2 * w2... Xn * Wn) + bias
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                #The error function is to correct the bias
                error_func = self.lr * (y_[idx] - y_predicted) 
                self.weights += error_func * x_i
                self.bias += error_func 

    def predict(self, x):        
        #This is to normalize the data, so we have to have only 0 and 1 values
        #So, if the i is > than 0, it could be 0.25 for example, it will be changed for 1
        #if the i is < than 0, it could be -0.1 for example, it will be changed for 0
        x_ = np.array([1 if i > 0 else 0 for i in x])

        #Calc linear function
        # Linear = (x1 * w1 + x2 * w2... Xn * Wn) + bias
        linear_output = np.dot(x_, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted  


    def _unit_step_function(self, x):
        return 1 if x >= 0 else 0    

 #create arrays with values      
_inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
_outputs = [1, 0, 0, 1]

test = np.array([2.5, 2])

p = Perceptron(learning_rate=0.01, epochs=100)
p.fit(_inputs, _outputs)

pred = p.predict(test)

print(pred)


            

