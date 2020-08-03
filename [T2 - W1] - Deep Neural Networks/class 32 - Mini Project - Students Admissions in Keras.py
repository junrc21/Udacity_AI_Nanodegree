import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

'''
Activation functions:
- Sigmoid: Results are between 0 and 1 
- Tanh: Results are between -1 and 1
- Relu: If result is > 0 = X, else the result will be 0 (Result >= 0 -> X else 0)

Optimizers:
- SGD: 
    This is Stochastic Gradient Descent. It uses the following parameters:
    - Learning rate.
    - Momentum (This takes the weighted average of the previous steps, in order to get a bit of momentum and go over bumps, as a way to not get stuck in local minima).
    - Nesterov Momentum (This slows down the gradient when it's close to the solution).

-Adam:
    Adam (Adaptive Moment Estimation) uses a more complicated exponential decay 
        that consists of not just considering the average (first moment), 
        but also the variance (second moment) of the previous steps.

- RMSProp:
    RMSProp (RMS stands for Root Mean Squared Error) decreases the learning rate 
    by dividing it by an exponentially decaying average of squared gradients.

'''

class StudentsEvaluation:

    def __init__(self, epochs=1000):
            self.epochs = epochs  

    #Load our data from file
    def loadData(self):
        return pd.read_csv('https://stats.idre.ucla.edu/stat/data/binary.csv')

    #Method to plot our data - optional
    def plot_points(self, data):
        X = np.array(data[["gre","gpa"]])
        y = np.array(data["admit"])
        admitted = X[np.argwhere(y==1)]
        rejected = X[np.argwhere(y==0)]
        plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'red', edgecolor = 'k')
        plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'cyan', edgecolor = 'k')
        plt.xlabel('Test (GRE)')
        plt.ylabel('Grades (GPA)')        
        plt.show()

        #Plot our data divided by rank - optional
    def plot_by_rank(self, data):
        data_rank1 = data[data["rank"]==1]
        data_rank2 = data[data["rank"]==2]
        data_rank3 = data[data["rank"]==3]
        data_rank4 = data[data["rank"]==4]

        self.plot_points(data_rank1)        
        self.plot_points(data_rank2)       
        self.plot_points(data_rank3)        
        self.plot_points(data_rank4)
        

    #Normalize our data
    def Normalize_data(self, data): 
        # remove NaNs
        data = data.fillna(0)

        # One-hot encoding the rank (because now it's between 1 to 4)
        #So, we will transform these 4 values in 4 inputs        
        processed_data = pd.get_dummies(data, columns=['rank'])

        #GRE has score between 0 and 800, it's huge. We need normalize the data.
        #So we will get the value and divide each value to 800
        processed_data["gre"] = processed_data["gre"]/800

        #GPA is the same way, but the value is 0 to 4
        #Let's normalize our data
        processed_data["gpa"] = processed_data["gpa"]/4

        #After all normalize, we will have something like that:        
        #admit    gre     gpa    rank_1  rank_2  rank_3  rank_4
        # 1     0.825    0.9175    0       0       1       0
        # 1     1.000    1.0000    1       0       0       0
        # 0     0.650    0.7325    0       0       0       1

        return processed_data
       

    #Prepare our data dividing them in two arrays
    def prepareData(self, data):
        #Now, we split our data input into X, and the labels y
        #and one-hot encode the output, so it appears as two classes (accepted and not accepted).
        X = np.array(data)[:,1:]
        
        #This can be used for categorical data, like [0, 1]
        #y = to_categorical(np.array(data["admit"], 1))

        #As we want just one output, I don't want categorize our data.
        y = np.array(data["admit"])
        
        #Create dataset for train and test
        X_train, y_train = X[:300],  y[:300]
        X_test, y_test = X[300:],  y[300:]
        
        return X_train, y_train, X_test, y_test

    #Build the model
    def createModel(self, X_train, y_train):  
        #Create the Sequential model
        model = Sequential()

        #1st Layer:
        #Specify the number of nods (units), activation function and imputs dimension
        model.add(Dense(units=128, dropout=0.0, activation='relu', input_dim=6))

        #2nd Layer - Add a fully connected output layer with the activation function for the layer
        model.add(Dense(units=64, dropout=0.0, activation='relu'))

        #3rd Layer - Add a fully connected output layer with the activation function for the layer
        #I set unit = 1 because I waant 1 output1
        #And I using sigmoid because it's better for binary classification
        model.add(Dense(units=1, dropout=0.0, activation='sigmoid'))

        #specify the loss function, optimizer and metrics we want to evaluate the model with
        #I choose the binary_crossentropy as loss function, because we have a binary output
        model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        #Show the resulting model architecture -optional
        #model.summary()

        #Fitting the model
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=10, verbose=0)

        #Scoring the model
        score = model.evaluate(X_train, y_train)
        print("\nAccuracy: ", score[-1])

        return model

    #Checking the predictions
    def predict(self, X_test):
        #Checking the predictions
        prediction = model.predict(X_test)

        print("\nPredictions:")        
        #print(model.predict(X_test))

        return prediction

    def showResult(self, prediction, y_test):

        for i,p in enumerate(prediction):
            print('Real: {} - Predicted: {}'.format((y_test[i] > 0.5),(p > 0.5)))
        


#Create an instance
se = StudentsEvaluation(epochs=3000)

#Load our data from file
data = se.loadData()

#plot our data
#se.plot_points(data)

#plot our data by rank
#se.plot_by_rank(data)

#Normalize our data
processed_data = se.Normalize_data(data)

#Prepare our data dividing them in two arrays
X_train, y_train, X_test, y_test = se.prepareData(processed_data)

#Build the model
model = se.createModel(X_train, y_train)

#Checking the predictions
prediction = se.predict(X_test)

#Show the result
se.showResult(prediction, y_test)


  



