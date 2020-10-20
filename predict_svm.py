import numpy as np
import pandas as pd
import pickle as pkl
import json

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

import argparse

def process_review(text):
	stop_words = set(stopwords.words('english')) 
	clean_rev = [w.lower() for w in word_tokenize(text) if w not in stop_words and w.isalpha()]
	return ' '.join(clean_rev)

def predict(text, svm):
	pred = svm.predict(text)
	confidence = svm.decision_function(text)
	result = {'label':pred.tolist(), 'confidence':confidence.tolist()}
	with open('best_svm_predcition.json', 'w') as f:
		json.dump(result, f)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Predict label of review.")
	parser.add_argument('text', help='review text')
	parser.add_argument('--model', '-m', default='best_svm.sav', help='svm model')
	parser.add_argument('--vectorizer', '-v', default='vectorizer.pkl')
	args = parser.parse_args()

	svm = pkl.load(open(args.model, 'rb'))
	clean_rev = process_review(args.text)
	with open(args.vectorizer, 'rb') as f:
		transformer = pkl.load(f)

	review = transformer.transform([clean_rev])

	predict(review, svm)


