#python3

#internal
import logging
import random

#external
import numpy as np

"""
Matrix Decomposition


	Method 1) Singular Value Decomposition of Forced Square Symmetric Positive Definite Matrix
	Method 2) Eigen Decomposition of Forced Square Symmetric Positive Matrix
	Method 3) Stocastic Gradient Decent Decomposition

"""

#set up logger
#filename= __name__ + ".log"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# check symmetry of matrix up to some tolerance
def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

#force square symetric matrix to symetric positive definite matrix
def to_positive_definite(S):
	""" Take absolute value of S and update diagnol entries to make a diagonally dominant matrix with diagonal entries greater than 0. """
	# take absolute value of S
	S = np.absolute(S)

	# Sum rows in S
	new_diagonals = S.sum(axis=1)

	#replace diagnols in S
	np.fill_diagonal(S, new_diagonals)

	return S

"""
####################################################################################
# Singular Value Decomposition of Forced Square Symmetric Positive Definite Matrix #
####################################################################################

1.) Force Cooccurence Matrix A to Positive Definite Matrix
	" A diagonally dominant(by rows) symetric matrix with diagonal elements all greater than zero is positive definite."
	Take symmetric matrix and make diagonally dominant with diagnonal entries greater than 0

2.) Square Symmetric Positive Definite Matrix Decomposition
	" If A is positive definite, then A = QLQt = UDV (where U=V=Q and L=D) can be written as A = WWt where W = Qsqrt(L) "
	-SVD: A=UDV, W = Vsqrt(diagnol(D))
	Find V and D from singular value decomposition of A
	return W = Vsqrt(D)

overview of code:

	#
	# force symmetric matrix to positive definite matrix
	#

	# take absolute value of A
	A = np.absolute(A)

	# Sum rows in a
	new_diagonals = A.sum(axis=1)

	#replace diagnols in A
	np.fill_diagonal(A, new_diagonals)

	#
	# decompose positive definite matrix
	#

	# singular value decomposition
	U, D, V = np.linalg.svd(A, full_matrices=False)

	#
	# compute W from V and D of singular value decomposition
	#

	# Create matrix W = Vtsqrt(diagnol(D)) #why Vt?
	W = np.dot(np.transpose(V), np.sqrt(np.diag(D)))

	#A = WWt
"""

def svd_spd_decomposition(P):
	""" return M such that P = MMt, where matrix parameter P is SPD """
	# Assert Matrix P is symetric
	assert check_symmetric(P)

	# singular value decomposition
	U, D, V = np.linalg.svd(P, full_matrices=False)

	# Create matrix W = Vtsqrt(diagnol(D)) #why Vt?
	M = np.dot(np.transpose(V), np.sqrt(np.diag(D)))

	#print(np.transpose(V))

	return M

#
# Perform Symmetric Positive Definite Decomposition
#

#P = to_positive_definite(A)
#W = svd_spd_decomposition(P)

"""
##################################################################
# Eigen Decomposition of Forced Square Symmetric Positive Matrix #
##################################################################

1.) Force Cooccurence Matrix A to Positive Definite Matrix
	" A diagonally dominant(by rows) symetric matrix with diagonal elements all greater than zero is positive definite."
	Take symmetric matrix and make diagonally dominant with diagnonal entries greater than 0

2.) Square Symmetric Positive Definite Matrix Decomposition
	" If A is positive definite, then A = QLQt = UDV (where U=V=Q and L=D) can be written as A = WWt where W = Qsqrt(L) "
	-EVD: A=QLQt, W = Qsqrt(diagnol(L))
	Find Q and L from eigen decomposition of A
	return W = Qsqrt(L)

overview of code:

	#
	# force symmetric matrix to positive definite matrix
	#

	# take absolute value of A
	A = np.absolute(A)

	# Sum rows in a
	new_diagonals = A.sum(axis=1)

	#replace diagnols in A
	np.fill_diagonal(A, new_diagonals)

	#
	# decompose positive definite matrix
	#

	# singular value decomposition
	L, Q = np.linalg.eig(A)

	#
	# compute W from Q and L of singular value decomposition
	#

	# Create matrix W = Vtsqrt(diagnol(D)) #why Vt?
	W = np.dot(np.transpose(Q), np.sqrt(np.diag(L)))

	#A = WWt
"""

#
# Perform Eigen Decomposition of Forced Square Symmetric Positive Matrix
#

def evd_spd_decomposition(P):
	""" return M such that P = MMt, where matrix parameter P is SPD """
	
	# Assert Matrix P is symetric
	assert check_symmetric(P)	

	# singular value decomposition
	L, Q = np.linalg.eig(P)

	#if L and Q returned in incorrect order
	#L = np.sort(L)
	#Q = Q[:, L.argsort()]

	# Create matrix W = Vtsqrt(diagnol(D))
	M = np.dot(Q, np.sqrt(np.diag(L)))

	return M

#P = to_positive_definite(A)
#W = evd_spd_decomposition(P)

