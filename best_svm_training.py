import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import svm

import pickle as pkl

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
    return tfidf_vectorizer, train_feature, test_feature



if __name__ == '__main__':

	train_x, train_y, test_x, test_y = generate_data()

	print('Data successfully read in.')

	vectorizer, train_x_tfidf, test_x_tfidf = tf_idf(train_x, test_x, max_df=0.95, ngram=(1,1))
	
	print('Datasets prepared!')

	with open('vectorizer.pkl', 'wb') as f:
	    pkl.dump(vectorizer, f)


	clf = svm.LinearSVC(random_state=123, penalty='l2', C=1.0)
	clf.fit(train_x_tfidf, train_y)

	print('model built')

	pkl.dump(clf, open('best_svm.sav', 'wb'))

	print('model saved')
