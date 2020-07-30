import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
#Equation: 
# If Y == 1: Sum = Yx * log(p) * -1
# IF Y == 0: Sum = (1 - Yx) * log(1 - p) * -1
# NOTE: The "* -1" at the end is to change the result to negative to positive.

def cross_entropy(Y, P):
    CE = 0
    for i, _y in enumerate(Y): 
        if _y == 1:
            CE += (_y * np.log2(P[i])) * -1
    return  CE 


Y = [0, 1, 0]
P = [0.15, 0.60, 0.25]

ce = cross_entropy(Y, P)
print('%.2f' % ce)