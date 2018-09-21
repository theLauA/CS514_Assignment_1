import numpy as np
#from sklearn.preprocessing import normalize
from scipy.linalg import orth

# Both dif and maxloop are stopping criteria
def powerIterate(A, x, dif, maxloop):
	curr_dif = 1.0
	count = 0
	x_i = x
	
	while curr_dif >  dif or count < maxloop:
		x_tmp = np.dot(A, x_i)
		x_tmp = x_tmp / np.linalg.norm(x_tmp)
		curr_dif = np.linalg.norm(x_i - x_tmp, 2)
		x_i = x_tmp
		count+=1
	
	return x_i

def powerIterate_v2(A, x, dif, maxloop):
	curr_dif = 1.00
	count = 0
	x_i = orth(x)
	while dif > dif or count < maxloop:
		x_tmp = np.dot(A,x_i)
		x_tmp = orth(x_tmp)
		curr_dif = np.linalg.norm(x_i - x_tmp, 2)
		x_i = x_tmp
		count+=1
	
	return x_i

#Main
A = np.array([[1,2,3,4,5,6,7,8,9,10],[2,3,4,5,6,7,8,9,10,0],[3,4,5,6,7,8,9,10,0,0],[4,5,6,7,8,9,10,0,0,0],[5,6,7,8,9,10,0,0,0,0],[6,7,8,9,10,0,0,0,0,0],[7,8,9,10,0,0,0,0,0,0],[8,9,10,0,0,0,0,0,0,0],[9,10,0,0,0,0,0,0,0,0],[10,0,0,0,0,0,0,0,0,0]])

x = np.transpose(np.array([1,1,1,1,1,1,1,1,1,1]))
x_4 = np.random.rand(10,4)

difference = 1

v1 = powerIterate(A, x, 0, 100)
v4 = powerIterate_v2(A,x_4,0,100)
print(v1)
print(v4)
	
	
U,D, Vh = np.linalg.svd(A)
print(Vh.T)

