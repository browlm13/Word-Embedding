library(clusterGeneration)
library(Metrics)

#works for symetric positive definite matrices (eigen values must be positive)
symmetric_decomposition <- function(A){

	v <- svd(A)$v
	d <- svd(A)$d
	W = v %*% sqrt(diag(d))
	return(W)
}

#Q: Can we add elements to diagonal of a symetric matrix to ensure it is positive definite?

#answer: A diagonally dominant(by rows) symetric matrix with diagonal elements all greater than zero is positive definite.

symmetric_force_positive_definite <- function(A){
	#force diagonal row dominance on symetric matrix A
	new_diagonals <- rowSums(abs(A)) + 1
	diag(A) <- new_diagonals
	return(A)
}

#if S is positive definite multiply sqrt(S) by any matrix Q that has orthonormal columns (so that QtQ = I). then Q*sqrt(S) is a choice for A (AtA = S)
#AtA = (Qsqrt(S))t(Qsqrt(S))= sqrt(S)Qt Qsqrt(S) = S

#positive definite conditions
#1. All eigenvalues of S must be greater than 0 
#2. The energy is non negitiv for every: x: xtSx >= 0
#3.S has the form AtA

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
print(mae(A,A2))
}
