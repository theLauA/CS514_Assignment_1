Run With Python 2.7 

For CS514 Homework Assignment 1

Method `powerIterate_v3(A, maxloop)` in svd_power.py is an implementation of Power Method from slides/textbook.
It takes two parameters, A is a N, M matrix, maxloop is the maximum iteration for Power Method; returns the first singular vector of A.
Method perform Power Method on B = A.T.dot(A), so that (B^k).dot(x) -> v1, which is the first singular vector of A

Method `SVD_POWER(A, maxloop)` in svd_power.py is an implementation singular value decomposition with Power Method.
It takes two paramters, A is a N by M matrix, maxloop is the maximum iteration for Power Method; returns U,D,Vh where A = U.dot(np.diag(D)).dot(Vh).
The implementation runs powerIterate_v3 on A to find v1, then runs powerIterate_v3 on A' = A - d11 * u1.dot(v1), where d11 is the first singular value, u1 is the first left singular vector, and v1 is the first right singular vector.
Repeats until min(N, M), or the resulting singular value is larger than the last singular value.