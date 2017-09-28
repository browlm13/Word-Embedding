library(clusterGeneration)
library(Metrics)

#works for symetric positive definite matrices (eigen values must be positive)
symmetric_decomposition <- function(A){

	v <- svd(A)$v
	d <- svd(A)$d
	W = v %*% sqrt(diag(d))
	return(W)
}

#can we add elements to diagonal of symetric matrix to ensure it is positive definite?
#i think the diagnol entries should be greater than each non diagnol entry but im not sure how to prove this
if (FALSE){
dimesions = 15
A = genPositiveDefMat("c-vine",dim= dimesions)$Sigma
A = A + 5
v <- svd(A)$v
d <- svd(A)$d
W = v %*% sqrt(diag(d))
A2 = W %*% t(W)

print(A)
print(A2)
print(mse(A,A2))

A = genPositiveDefMat("eigen",dim= dimesions)$Sigma
A = A + 5
v <- svd(A)$v
d <- svd(A)$d
W = v %*% sqrt(diag(d))
A2 = W %*% t(W)

print(A)
print(A2)
print(mse(A,A2))

A = genPositiveDefMat("onion",dim= dimesions)$Sigma
A = A + 5
v <- svd(A)$v
d <- svd(A)$d
W = v %*% sqrt(diag(d))
A2 = W %*% t(W)

print(A)
print(A2)
print(mse(A,A2))

A = genPositiveDefMat("unifcorrmat",dim= dimesions)$Sigma
A = A + 5
v <- svd(A)$v
d <- svd(A)$d
W = v %*% sqrt(diag(d))
A2 = W %*% t(W)

print(A)
print(A2)
print(mse(A,A2))
print(mae(A,A2))
}
