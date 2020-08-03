import numpy as np
from keras.utils import np_utils
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

tf.python.control_flow_ops = tf

# Set random seed
np.random.seed(42)

# Our data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# One-hot encoding the output
#y = np_utils.to_categorical(y)

### Building the model
# Create the Sequential model
xor = Sequential()

#1st Layer:
#Specify the number of nods (units), activation function and imputs dimension
xor.add(Dense(units=4, activation="tanh", input_dim=2))

#2nd Layer - Add a fully connected output layer with the activation function for the layer
#I set unit = 1 because this is the output layer, and we want just one ouput for the XOR problem
#1 for True and 0 for False
xor.add(Dense(units=1, activation="sigmoid"))

#specify the loss function, optimizer and metrics we want to evaluate the model with
xor.compile(loss="binary_crossentropy", optimizer="adam", metrics = ['accuracy'])

#Show the resulting model architecture -optional
#xor.summary()

#Fitting the model
history = xor.fit(X, y, epochs=4000, batch_size=1, verbose=0)

#Scoring the model
score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])

#Checking the predictions
print("\nPredictions:")
print(xor.predict(X))

#To print true or false for XOR problem
print(xor.predict(X) > 0.5)