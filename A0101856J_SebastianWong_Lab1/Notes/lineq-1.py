import numpy as np
import numpy.linalg as la

A = np.matrix([[1,1,1],[2,0,1],[1,2,1]])
print "A = \n", A
b = np.matrix([[2],[1],[3]])
print "b = \n", b
x = la.inv(A) * b
print "x = \n", x	
print "A * x = \n ", A * x

# using built in solve function
x = la.solve(A,b)
print "Using built in solve function \n A * x = \n", A * x