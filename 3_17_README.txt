Run 3_17.py with Python 2

Part 1
Method powerIterate_v3(A, maxloop) in svd_power.py is an implementation of Power Method from slides/textbook.
It takes two parameters, A is a N, M matrix, maxloop is the maximum iteration for Power Method; returns the first singular vector of A.
Method perform Power Method on B = A.T.dot(A), so that (B^k).dot(x) -> v1, which is the first singular vector of A.

Part 2
Method powerIterate_v2(A, x, dif, maxloop) in svd_power.py is an alternative implementation of Power Method from question.
It takes four parameters, A is a N by N matrix, x is a N by K matrix with random vectors, maxloop is the maximum iteration for Power Method; returns the first k singular vectors of A.
Method perform the alternative Power Method on A, where in each iteration, it finds othornormal basis of A.dot(x). The vectors of the orthornomal basis will approch first k singular vectors of A.
 