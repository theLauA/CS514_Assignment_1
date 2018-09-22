from svd_power import *

#Main
A = np.array([[1,2,3,4,5,6,7,8,9,10],[2,3,4,5,6,7,8,9,10,0],[3,4,5,6,7,8,9,10,0,0],[4,5,6,7,8,9,10,0,0,0],[5,6,7,8,9,10,0,0,0,0],[6,7,8,9,10,0,0,0,0,0],[7,8,9,10,0,0,0,0,0,0],[8,9,10,0,0,0,0,0,0,0],[9,10,0,0,0,0,0,0,0,0],[10,0,0,0,0,0,0,0,0,0]])

x = np.random.rand(10,1)
x_4 = np.random.rand(10,4)

difference = 1

v1 = powerIterate(A, x, 0, 100)
v4 = powerIterate_v2(A,x_4,0,100)

print("The First Eigenvector from Power Method")
print(v1)
print("The First Four Eigenvector from orth-basis Power Method")
print(v4)
	

#Debug with Numpy's SVD
#U,D, Vh = np.linalg.svd(A)
#print(Vh.T)

