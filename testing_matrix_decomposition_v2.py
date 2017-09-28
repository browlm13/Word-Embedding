#python 3

#internal
import logging
from collections import Counter
import random
import itertools

#external
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk import FreqDist
from nltk.util import ngrams

#dir
from matrix_decomposition import *


"""
	Corpus Manipulation Methods
"""

#set up logger
#filename= __name__ + ".log"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def word_frequency_dict(corpus):
	""" returns a dictionary of word and their assosiated frequencies """

	logger.info("Creating \'word frequency\'' dictionary from corpus")

	# remove punctuation, convert to lower case, tokenize, cout frequencies
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(corpus.lower())		# tokens = nltk.word_tokenize(corpus.lower()) # without removing punctiation
	fdist = FreqDist(tokens) 						# fdist.keys() fdist.values()
	return dict(fdist)

def word_ids(corpus):
	""" returns a dictionary of key: id, value: word for every unique word in corpus """

	logger.info("Creating \'word id\'' dictionary from corpus")

	# remove punctuation, convert to lower case, tokenize, keep unqiue
	tokenizer = RegexpTokenizer(r'\w+')
	unique_tokens = set(tokenizer.tokenize(corpus.lower()))
	word_ids = {i:word for i,word in enumerate(unique_tokens)}
	return word_ids

def process_text(text):
	""" return list of lowercase alphabetic words from text """
	tokenizer = RegexpTokenizer(r'\w+')
	return tokenizer.tokenize(text.lower())

def ngram_tupples(corpus, n):
	""" Create ngram tupples by sentence. Where n is the distance between words in a sentence. """
	sentences = sent_tokenize(corpus)

	pairs = []
	for s in sentences:
		unique_tokens = process_text(s)
		pairs.extend(ngrams(unique_tokens,n))

	return pairs


def get_unique_words(corpus):
	return list(set(process_text(corpus)))

def w2id_id2w_maps(unique_words):
	""" return both dictonaries for mapping between words and ids """
	id2w = {i:w for i,w in enumerate(unique_words)}
	w2id = {w:i for i,w in id2w.items()}
	return w2id, id2w

def ngram_inc_amt(n):
	""" return float for increment weight of pair occurence n distance appart. \nWeight increment ~ 1/n """
	return 1/float(n**2)

def words2ids(words, w2id):
	""" return list of ids inplace of list of words using w2id dictionary """
	return [w2id[w] for w in words]

def cooccurence_pair_of_distance(sentence_list, d):
	""" return list of unique coocurence pairs of distace d """

	all_ngrams = ngrams(sentence_list,d)

	all_pairs = []
	for t in all_ngrams:
		if len(t) > 1:
			all_pairs.extend(list(itertools.combinations(t, 2)))

	return list(set(all_pairs))

def break_corpus(corpus):
	""" Build Cooccurence Matrix. Return A, n, w2id, id2w """

	unique_words = get_unique_words(corpus)
	n = len(unique_words)
	w2id, id2w = w2id_id2w_maps(unique_words)

	#create empty cooccurence matrix
	#A = np.zeros([n,n],np.float32)
	A = np.ones([n,n],np.float32)

	#compute cooccurence matrix
	sentences = sent_tokenize(corpus)
	for s in sentences:
		s = process_text(s)
		max_distance = len(s) + 1
		s = [w2id[w] for w in s]	#convert words to ids

		for d in range(2,max_distance):
			pairs = cooccurence_pair_of_distance(s, d)

			#update cooccurence matrix for each pair
			for p in pairs:
				A[p[0],p[1]] += ngram_inc_amt(d)
				A[p[1],p[0]] += ngram_inc_amt(d)

	return A, n, w2id, id2w

def total_squared_error():
	total_error = 0
	for i in range(0,n):
		wi = np.transpose(W[:,i])
		for j in range(0,n):
			if i != j:
				wj = W[:,j]
				total_error += (np.dot(wi, wj) - A[i,j])**2
	return total_error

def mean_squared_error(A, B):
	return ((A - B) ** 2).mean(axis=None)

def djdx(x):
	wi = np.transpose(W[:,x])
	total_error = 0
	for i in range(0,n):
		if i != x:
			wj = W[:,i]
			total_error += np.dot(wi, wj) - A[x,i]
	return np.multiply(W[:,x],total_error)


#
#
#	Testing
#
#

corpus = """Oh, a sleeping drunkard
	Up in Central Park,
	And a lion-hunter
	In the jungle dark,
	And a Chinese dentist, 
	And a British queen--
	All fit together 
	In the same machine. 
	Nice, nice, very nice;
	Nice, nice, very nice;
	Nice, nice, very nice--
	So many different people
	In the same device."""

A, n, w2id, id2w = break_corpus(corpus)


# intilize random word vector (range (-0.5, 0.5])
vector_size = n #not sure how to choose thise
#W = ((np.random.rand(vector_size, n) - 0.5) / float(vector_size + 1))
W = ((np.random.rand(vector_size, n) - 2) / float(vector_size + 1))

#
#	Cooccurence Matrix
#

print ("\nWord Map:\n")
print (w2id)
#print("\nCooccurence Matrix A:\n")
#print (A)

#
# Stocastic Gradient Decent Decomposition
#

iterations = 10000
learning_rate = 0.005

print("\n\n\nStocastic Gradient Decent Decomposition:\n\tNumber of Iterations: %d, \tLearning Rate: %f\n" % (iterations, learning_rate))

# Random W - WWt computation for comparison
WWt = np.dot(W, np.transpose(W))
print ("Start Mean Squared Error (A,WWt): %f" % mean_squared_error(A, WWt))

for i in range(iterations):
	x = random.randrange(0, n)
	W[:,x] = W[:,x] - learning_rate * djdx(x)

# Computed WWt
WWt = np.dot(W, np.transpose(W))

print ("End Mean Squared Error (A,WWt) %f" % mean_squared_error(A, WWt))

#print("\n\nA:")
#print (A)
#print("\nWtW:\n")
#print(WWt)

#
# Perform SVD Symmetric Positive Definite Decomposition
#

P = to_positive_definite(A)
W = svd_spd_decomposition(P)
WWt = np.dot(W, np.transpose(W))

print("\n\n\nSVD Symmetric Positive Definite Decomposition:\n")
#print("\n\nA:")
#print (A)
#print("\n\nP (A modified into SPD):\n")
#print(P)
#print("\nWWt:\n")
#print(WWt)
print("\nMean Squared Error (P,WWt): %f\n" % mean_squared_error(WWt,P))

#
# Perform EVD Symmetric Positive Definite Decomposition
#

P = to_positive_definite(A)
W = evd_spd_decomposition(P)
WWt = np.dot(W, np.transpose(W))

print("\n\n\nEVD Symmetric Positive Definite Decomposition:\n")
#print("\n\nA:")
#print (A)
#print("\n\nP (A modified into SPD):\n")
#print(P)
#print("\nWWt:\n")
#print(WWt)
print("\nMean Squared Error (P,WWt): %f\n" % mean_squared_error(WWt,P))
