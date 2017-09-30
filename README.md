# Word-Embedding
LJ Brown
## Experiments With Word Embedding 

In this repository we test diffrent techniques to map all unique words found in a corpus to vectors in a vector space. The idea, and hope, is that some relationships between words found in the corpus will be preserved through this mapping and will manifest as characteristics of the word vectors. [More Information On Vector Representations Of Words](https://www.tensorflow.org/tutorials/word2vec)

1. Build a co-occurrence matrix from a corpus which represents how frequently word pairs occur together.

1. Search for word vectors with the soft constraint that given a word vector pair, their inner product will yield a value close to the two values in the co-occurrence matrix associated with those two words.

Methods implimented in this repository for decomposing the cooccurence matrix into word vectors:

* Stochastic Gradient Descent, which draws heavily on the implementations by "Word2vec" and "GloVe"

* Methods usgin Eigen Decomposition and Singular Value Decomposition.
