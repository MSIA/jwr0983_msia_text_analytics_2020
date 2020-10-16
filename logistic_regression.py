import numpy
import pandas as pd
import json
import random
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import multiprocessing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import confusion_matrix

def generate_data():

	reviews = []
	with open('reviews_Movies_and_TV_5.json') as json_file: 
	    for rec in json_file:
	        dic = json.loads(rec)
	        reviews.append(dic)

	random.seed(123)
	random.shuffle(reviews)
	reviews = reviews[:500000]

	review = []
	label = []

	for rev in reviews:
	    review.append(rev['reviewText'])
	    label.append(rev['overall'])

	review_df = pd.DataFrame({"review":review, "label":label})

	stop_words = set(stopwords.words('english')) 
	def process_review(text):
	    clean_rev = [w.lower() for w in word_tokenize(text) if w not in stop_words and w.isalpha()]
	    return ' '.join(clean_rev)

	clean_rev = [process_review(i) for i in review]
	review_df['cleaned_review'] = clean_rev

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

if __name__ == '__main__':

	train_x, train_y, test_x, test_y = generate_data()

	train_x_tfidf, test_x_tfidf = tf_idf(train_x, test_x)

	lr_1gram = LogisticRegression(solver='liblinear', random_state=123, C=5, penalty='l1', max_iter=100)
	
	model = lr_1gram.fit(train_x_tfidf,train_y)

	pred = model.predict(test_x_tfidf)

	cm = confusion_matrix(test_y, pred)
	recall = np.diag(cm) / np.sum(cm, axis = 1)
	precision = np.diag(cm) / np.sum(cm, axis = 0)

	f1 = 2*recall*precision/(recall+precision)

	(unique, counts) = np.unique(test_y, return_counts=True)

	print("Micro-recall is: %s"%(sum(counts*recall)/sum(counts)))
	print("Micro-precision is %s"%(sum(counts*precision)/sum(counts)))
	print("Micro-F1 score is %s"%(sum(counts*f1)/sum(counts)))
	print("Overall Accuracy is %s"%(sum(pred==test_y)/len(pred)))
