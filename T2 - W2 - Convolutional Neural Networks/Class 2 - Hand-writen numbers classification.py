import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.callbacks import ModelCheckpoint

from keras.datasets import mnist

class numbersClassification:

    def __init__(self, epochs=1000):
            self.epochs = epochs 

    def loadDataset(self):

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
                                                        
        return x_train, y_train, x_test, y_test

    
    #Method to plot our data - optional
    def plot_image(self, x_train):
        fig = plt.figure(figsize=(20, 20))
        for i in range(6):
            ax = fig.add_subplot(1,6, i+1, xticks=[], yticks=[])
            ax.imshow(x_train[i], cmap='gray')
            ax.set_title(str(y_train[i]))     
        plt.show()

    
    def visualize_input(self, img):
        fig = plt.figure(figsize = (12,12)) 
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap='gray')
        width, height = img.shape
        thresh = img.max()/2.5
        for x in range(width):
            for y in range(height):
                ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='white' if img[x][y]<thresh else 'black')
        plt.show()

    
    def normalizeData(self, x_train, y_train, x_test, y_test):
        # Hot-coding
        # Rescale image from [0,255] pixels to 0 and 1    
        x_train = x_train.astype('float32')/255
        x_test = x_test.astype('float32')/255

        # one-hot encode the labels
        y_train = np_utils.to_categorical(y_train, 10)
        y_test = np_utils.to_categorical(y_test, 10)
          
        return x_train, y_train, x_test, y_test

    def buildModel(self, x_train, y_train):  
        #Create the Sequential model
        model = Sequential()

        #1st Layer:
        #Specify the number of nods (units), activation function and imputs dimension
        model.add(Flatten(input_shape=x_train.shape[1:]))        
        #model.add(Dropout(0.0))

        #2nd Layer - Add a fully connected output layer with the activation function for the layer
        model.add(Dense(units=512, activation='relu'))        
        model.add(Dropout(0.2)) #Probablity of a note be removed in the training

        #3rd Layer - Add a fully connected output layer with the activation function for the layer
        model.add(Dense(units=512, activation='relu'))
        model.add(Dropout(0.2)) #Probablity of a note be removed in the training

        #4th Layer - Add a fully connected output layer with the activation function for the layer
        #I set unit = 10 because we have 10 digits as possible output
        #And I using softmax because it's better for probability classification
        model.add(Dense(units=10, activation='softmax'))
        #model.add(Dropout(0.0))

        #specify the loss function, optimizer and metrics we want to evaluate the model with
        #I choose the binary_crossentropy as loss function, because we have a binary output
        model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        #Show the resulting model architecture -optional
        #model.summary()

        #train de model
        # In this architecture we have 3 datasets: Train, validation and test
        # Train: Is used for fit the model weights
        # Validation: Check how model is doing
        # Test: Check accuracy of the trained model 
        # In our case only the best weight will be saved
        checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=0,save_best_only=1)
        
        #Fitting the model with train validation         
        # In this case 20% of the test dataset will be use for train our model as said before 
        hist = model.fit(x_train, y_train, epochs=self.epochs, batch_size=128, 
                            validation_split=0.2,callbacks=[checkpointer],
                            shuffle=True,verbose=0)        

        # load the weights that yielded the best validation accuracy
        model.load_weights('mnist.model.best.hdf5')

        #Scoring the model
        score = model.evaluate(x_train, y_train)
        print("\nAccuracy: ", score[-1])

        return model

    def predict(self, x_test):
        #Checking the predictions
        prediction = model.predict(x_test)

        print("\nPredictions:")        
        print(prediction)

        return prediction

    def showResult(self, prediction, y_test):
        error = 0
        correct = 0
        for i,p in enumerate(prediction):            
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
mnistDT = numbersClassification(5)

#Load our dataset from keras
x_train, y_train, x_test, y_test = mnistDT.loadDataset()

#Show the images
#mnistDT.plot_image(x_train)

#Normalize our data
x_train, y_train, x_test, y_test = mnistDT.normalizeData(x_train, y_train, x_test, y_test)

#Build the model
model = mnistDT.buildModel(x_train, y_train)

#Visualize the image number
mnistDT.visualize_input(x_test[10])

#Checking the predictions
prediction = mnistDT.predict(x_test)

#Visualizing the prediction
mnistDT.showResult(prediction, y_test)

#print the predicted number
print(np.argmax(prediction[10]))





                                                        

