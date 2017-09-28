library(clusterGeneration)
library(Metrics)

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