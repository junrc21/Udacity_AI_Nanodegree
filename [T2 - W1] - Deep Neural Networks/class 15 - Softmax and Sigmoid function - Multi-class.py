import numpy as np
import  matplotlib.pyplot as plt 

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.

#Euler number
#euler = 2.7182818284590452353602874713527

# Softmax equation:
# SM = e^x1 / (e^x1 + e^x2 + e^x3 + e^Xn)

# Sigmoid equation:
# S = 1 / (e^-x)
#Where x = sum of all features

def softmax(L):

    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result



L = [-2,-4, 0, 2, 4, 6, 8]

softmax_result = softmax(L)

plt.plot(softmax_result, 'r', color='green')
plt.show()

