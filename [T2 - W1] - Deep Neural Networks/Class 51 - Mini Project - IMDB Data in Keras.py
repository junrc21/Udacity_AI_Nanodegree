import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

from keras.datasets import imdb

class IMDB_Reviews:

    def __init__(self, epochs=1000):
            self.epochs = epochs 

    def loadDataset(self):

        (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                            num_words=1000,
                                                            skip_top=0,
                                                            maxlen=None,
                                                            seed=113,
                                                            start_char=1,
                                                            oov_char=2,
                                                            index_from=3)
                                                        
        return x_train, y_train, x_test, y_test

    
    def normalizeData(self, x_train, y_train, x_test, y_test):
        # One-hot encoding the output into vector mode, each of length 1000    
        tokenizer = Tokenizer(num_words=1000)
        x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
        x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
        

        # One-hot encoding the output
        num_classes = 2
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        return x_train, y_train, x_test, y_test

    def buildModel(self, x_train, y_train):  
        #Create the Sequential model
        model = Sequential()

        #1st Layer:
        #Specify the number of nods (units), activation function and imputs dimension
        model.add(Dense(units=60, activation='relu', input_dim=1000))
        model.add(Dropout(0.0))

        #2nd Layer - Add a fully connected output layer with the activation function for the layer
        model.add(Dense(units=60, activation='relu'))
        model.add(Dropout(0.0))

        #3rd Layer - Add a fully connected output layer with the activation function for the layer
        #I set unit = 1 because I waant 1 output1
        #And I using sigmoid because it's better for binary classification
        model.add(Dense(units=2, activation='sigmoid'))
        model.add(Dropout(0.0))

        #specify the loss function, optimizer and metrics we want to evaluate the model with
        #I choose the binary_crossentropy as loss function, because we have a binary output
        model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        #Show the resulting model architecture -optional
        #model.summary()

        #Fitting the model
        model.fit(x_train, y_train, epochs=self.epochs, batch_size=500, verbose=0)

        #Scoring the model
        score = model.evaluate(x_train, y_train)
        print("\nAccuracy: ", score[-1])

        return model

    def predict(self, x_test):
        #Checking the predictions
        prediction = model.predict(x_test)

        print("\nPredictions:")        
        #print(prediction)

        return prediction

    def showResult(self, prediction, y_test):
        error = 0
        correct = 0
        for i,p in enumerate(prediction):
            #print('Real: {} - Predicted: {}'.format((y_test[i] > 0.5),(p > 0.5)))
            y = (y_test[i] > 0.5)
            y_pred = (p > 0.5)
            if y[0] != y_pred[0]:
                error += 1
            else:
                correct += 1
            
        print('Neural Network final data:')
        print('Incorrect: %d' % error)
        print('Correct: %d' % correct)
           

#Create an instance
rv = IMDB_Reviews(22)

#Load our dataset from keras
x_train, y_train, x_test, y_test = rv.loadDataset()

#Normalize our data
x_train, y_train, x_test, y_test = rv.normalizeData(x_train, y_train, x_test, y_test)

#Build the model
model = rv.buildModel(x_train, y_train)

#Checking the predictions
prediction = rv.predict(x_test)

#Visualizing the prediction
rv.showResult(prediction, y_test)





                                                        

