import numpy as np
import numpy.linalg as la

file = open("data.txt")
data = np.genfromtxt(file,delimiter=",")
file.close()
print "data read is \n", data
rows = len(data[:,0])
# Creating b
b = np.matrix(np.reshape(data[:,0:2],(2*rows,1)))
print "b = \n",b

# Constructing first 3 columns of M
# Creating an array of zeros
firstHalf = np.zeros([rows*2,3])
# Copying values from last two columns of data
m1 = np.array(data[:,2:4])
# Inserting a row of 1s vertically
m2 = np.insert(m1,2,1,axis = 1)
# Assigning m2 to odd rows of zeros
firstHalf[::2] = m2
# Constructing next 3 columns of M
secondHalf = np.zeros([rows*2,3])
# Assigning m2 to even rows of zeros
secondHalf[1::2] = m2
M = np.append(firstHalf,secondHalf,axis = 1)
M = np.matrix(M)
print "M is \n", M
# Solving for least square solution
a, e, r, s = la.lstsq(M,b)
print "a is \n", a
ma = M*a
print "M*a is \n", ma
# Computing sum-squared error between M*a and b
norm = la.norm(M*a - b)
sumsqerror = norm*norm
print "sum-squared error between M*a and b is \n", sumsqerror
print "residue computed is \n", e

