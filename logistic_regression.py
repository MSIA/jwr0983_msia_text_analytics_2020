import numpy as np
import pandas as pd
import json
import random

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import multiprocessing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import confusion_matrix

def generate_data():
	review_df = pd.read_csv('cleaned.csv')

	train, test = train_test_split(review_df[['cleaned_review', 'label']].dropna(), random_state = 123)

	train_x = train['cleaned_review'].values
	train_y = train['label'].values
	test_x = test['cleaned_review'].values
	test_y = test['label'].values

	return train_x, train_y, test_x, test_y

def tf_idf(train, test, max_df=0.95, ngram=(1,1)):

    tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_df=max_df, ngram_range=ngram)
    tfidf_vectorizer.fit_transform(train)
    train_feature = tfidf_vectorizer.transform(train)
    test_feature = tfidf_vectorizer.transform(test)
    return train_feature, test_feature

def performance_metric(model, test, pred):

	cm = confusion_matrix(test, pred)
	recall = np.diag(cm) / np.sum(cm, axis = 1)
	precision = np.diag(cm) / np.sum(cm, axis = 0)

	f1 = 2*recall*precision/(recall+precision)

	(unique, counts) = np.unique(test, return_counts=True)


	with open('performance_metrics/{}.txt'.format(model), 'w') as file:
		file.write("Micro-recall is: %s"%(sum(counts*recall)/sum(counts)))
		file.write('\n')
		file.write("Micro-precision is %s"%(sum(counts*precision)/sum(counts)))
		file.write('\n')
		file.write("Micro-F1 score is %s"%(sum(counts*f1)/sum(counts)))
		file.write('\n')
		file.write("Overall Accuracy is %s"%(sum(pred==test)/len(pred)))

if __name__ == '__main__':

	train_x, train_y, test_x, test_y = generate_data()

	print('Data successfully read in. ')

	train_x_tfidf, test_x_tfidf = tf_idf(train_x, test_x, max_df=0.95, ngram=(1,1))
	train_x_tfidf_12, test_x_tfidf_12 = tf_idf(train_x, test_x, max_df=0.95, ngram=(1,2))
	train_x_tfidf_075, test_x_tfidf_075 = tf_idf(train_x, test_x, max_df=0.75, ngram=(1,1))
	train_x_tfidf_075_12, test_x_tfidf_075_12 = tf_idf(train_x, test_x, max_df=0.75, ngram=(1,2))

	lr_1gram_maxdf095_c5 = LogisticRegression(solver='liblinear', random_state=123, C=5, penalty='l1', max_iter=100)
	model = lr_1gram_maxdf095_c5.fit(train_x_tfidf,train_y)
	pred = model.predict(test_x_tfidf)
	performance_metric('lr_1gram_maxdf095_c5', test_y, pred)

	print('First model built.')

	lr_12gram_maxdf095_c5 = LogisticRegression(solver='liblinear', random_state=123, C=5, penalty='l1', max_iter=100)
	model = lr_12gram_maxdf095_c5.fit(train_x_tfidf_12,train_y)
	pred = model.predict(test_x_tfidf_12)
	performance_metric('lr_12gram_maxdf095_c5', test_y, pred)

	print('Second model built.')

	lr_1gram_maxdf075_c5 = LogisticRegression(solver='liblinear', random_state=123, C=5, penalty='l1', max_iter=100)
	model = lr_1gram_maxdf075_c5.fit(train_x_tfidf_075,train_y)
	pred = model.predict(test_x_tfidf_075)
	performance_metric('lr_1gram_maxdf075_c5', test_y, pred)

	print('Third model built.')

	lr_12gram_maxdf095_c5 = LogisticRegression(solver='liblinear', random_state=123, C=5, penalty='l1', max_iter=100)
	model = lr_12gram_maxdf095_c5.fit(train_x_tfidf_075_12,train_y)
	pred = model.predict(test_x_tfidf_075_12)
	performance_metric('lr_12gram_maxdf075_c5', test_y, pred)

	print('Fourth model built.')

	train_x_tfidf, test_x_tfidf = tf_idf(train_x, test_x, max_df=0.95, ngram=(1,2))
	lr_1gram_maxdf095_c5 = LogisticRegression(solver='liblinear', random_state=123, C=100, penalty='l1', max_iter=100)
	model = lr_1gram_maxdf095_c5.fit(train_x_tfidf_12,train_y)
	pred = model.predict(test_x_tfidf_12)
	performance_metric('lr_12gram_maxdf095_c100', test_y, pred)

	print('Fifth model built. ')






