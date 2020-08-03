import numpy as np
import  matplotlib.pyplot as plt 

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.

#Euler number
#euler = 2.7182818284590452353602874713527

# Softmax equation:
# SM = e^x1 / (e^x1 + e^x2 + e^x3 + e^Xn)

def softmax(Z):

    expZ = np.exp(Z)
    sumExpZ = sum(expZ)
    result = []
    for z in expZ:
        result.append(z/sumExpZ)
    return result

def softmax2(X):
    z1 = 0
    _z = 0
    result = []
    for x in X:
        z1 = np.exp(x)
        _z = 0 
        for _x in X:            
            _z += np.exp(_x)
    
        result.append(z1 / _z)
    
    return result

def softmax3(Z):
    expZ = np.exp(Z)
    sumExpZ = sum(expZ)
    result = []
    for z in expZ:
        result.append(z/sumExpZ)
    return result


L = [1.5, 2.5, 0.8]

softmax_result = softmax(L)
softmax_result2 = softmax2(L)
softmax_result3 = softmax3(L)

print(softmax_result)
print(softmax_result2)
print(softmax_result3)

plt.plot(softmax_result, 'r', color='green')
plt.show()

