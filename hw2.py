import os
import numpy as np
import nltk
import gensim
from gensim.models import Word2Vec
import pandas as pd
import time
path = "alt.atheism/"

def get_similar_most_words(model, target_words, num_of_words=5):
    res = {}
    for i in target_words:
        res[i] = model.wv.most_similar(i)[:num_of_words]
    return res


if __name__ == '__main__':
	# reading in the folder of corpus
	text = []
	for file in os.listdir(path):
	    f = open(path + file, 'r', encoding = 'utf8', errors='ignore')
	    text.append(f.read())
	    f.close()

	# word-level tokenization
	tokenized = []
	for t in text:
	    tokenized.append(nltk.word_tokenize(t))

	# normalization, removing non-alphanumeric characters, turning everthing into lower case
	normalized = []
	for tokenized_text in tokenized:
	    alphanum = [i.lower() for i in tokenized_text if i.isalpha()]
	    normalized.append(alphanum)

	# writing to output
	with open('output.txt', 'w') as f:
	    for doc in normalized:
	        for token in doc:
	            f.write(token+' ')
	        f.write('\n')

	# building a word2vec model based on the corpus
	cbow_win5_emb100 = Word2Vec(normalized, size=100, window=5, min_count=3, sg=0, workers=4)

	# show the embedded vector for the word 'computer'
	vector = cbow_win5_emb100.wv['computer']
	vector

	# model spec 1
	start = time.time()
	cbow_win5_emb100 = Word2Vec(normalized, size=100, window=5, min_count=3, sg=0, workers=4)
	print("word2vec using CBOW, window size 5, 100 embedding size takes %s"%(time.time()-start))

	# model spec 2
	start = time.time()
	cbow_win10_emb150 = Word2Vec(normalized, size=150, window=10, min_count=3, sg=0, workers=4)
	print("word2vec using CBOW, window size 10, 150 embedding size takes %s"%(time.time()-start))

	# model spec 3
	start = time.time()
	skip_win5_emb100 = Word2Vec(normalized, size=100, window=5, min_count=3, sg=1, workers=4)
	print("word2vec using skip-gram, window size 5, 100 embedding size takes %s"%(time.time()-start))


	# list of target word to test for similar words extraction
	target_words = ['god', 'human', 'science', 'data', 'earth', 'play', 'physics', 'computer', 'study', 'myth']

	# print dataframes of similar words to the above list for the three embeddings
	pd.DataFrame(get_similar_most_words(cbow_win5_emb100, target_words))
	pd.DataFrame(get_similar_most_words(cbow_win10_emb150, target_words))
	pd.DataFrame(get_similar_most_words(skip_win5_emb100, target_words))