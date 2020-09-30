import nltk
import os
import glob
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.stem import PorterStemmer
import time
import spacy
from joblib import Parallel, delayed
import re

def nltk_tokenize(contents, level = 'word'):
    if level == 'word':
        return word_tokenize(contents)
    else:
        return sent_tokenize(contents)

def nltk_stemming(contents):
    ps = PorterStemmer() 
    words = word_tokenize(contents)
    stem = {}
    for word in words:
        stem[word] = ps.stem(word)
    return stem

def nltk_pos_tagging(contents):
    words =  word_tokenize(contents)
    nltk_pos = nltk.pos_tag(words)
    return nltk_pos

def spacy_tokenization(contents):
    nlp = spacy.load('en_core_web_sm') 
    nlp.max_length = 5000000
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    doc = nlp(contents)
    sents = [sent.string.strip() for sent in doc.sents]
    words = [token.text for token in doc]
    return sents, words

def spacy_pos_tagging(contents):
    nlp = spacy.load('en_core_web_sm') 
    nlp.max_length = 5000000
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    doc = nlp(contents)
    words = [token.text for token in doc]
    pos = {word:word.pos_ for word in doc}
    return pos

def email_matching(text):
	email = re.findall(r'[A-Za-z0-9_\-\.]+\@[A-Za-z0-9_\-\.]+\.[A-Za-z0-9_\-\.]+', text)
	return email

def date_matching(contents):
	dates_1 = re.findall(r'[0-9]+/[0-9]+/[0-9]+', contents)
	dates_2 = re.findall(r'[0-9]{4}\-[0-9]{2}\-[0-9]{2}', contents)
	dates_3 = re.findall(r'\d+ [JFAMSOND]\w+ \d+', contents)
	dates_4 = re.findall(r'[JFAMSOND]\w+ \d+[th|st]+ \d+', contents)
	all_dates = dates_1 + dates_2 + dates_3 + dates_4
	return all_dates

if __name__ == '__main__':
	path = '/Users/wangjue/Documents/NWU/Courses/fall_2020/490_nlp/20_newsgroups/alt.atheism'
	dirs = os.listdir(path)
	os.chdir(path)

	d = []
	for file in glob.glob("*"):
	    with open(file, 'r', encoding="utf8", errors="ignore") as f:
	        data = f.read()
	    d.append(data)

	contents = ''.join(d)

	# NLTK Tokenization
	start_time = time.time() 
	words = nltk_tokenize(contents, level = 'word')
	sents = nltk_tokenize(contents, level = 'sent')
	elapsed_time = time.time() - start_time 
	print('Tokenization (both word and sentence level) in nltk for this corpus is :', elapsed_time)

	# NLTK Stemming
	start_time = time.time()
	stems = nltk_stemming(contents)
	elapsed_time = time.time() - start_time 
	print('Stemming in nltk took:', elapsed_time)

	# NLTK POS Tagging
	start_time = time.time()
	nltk_pos = pos_tagging(contents)
	print ("POS tagging takes %s"%(time.time()-start_time))

	# NLTK Tokenization Parallelized
	start_time = time.time()
	words = Parallel(n_jobs=3)(delayed(nltk_tokenize)(i) for i in d)
	sents = Parallel(n_jobs=3)(delayed(nltk_tokenize)(i) for i in d)
	print('Tokenization (both word and sentence level) in nltk with tokenization for this corpus is :', elapsed_time)


	# Spacy Tokenization
	start_time = time.time()
	sents, words = spacy_tokenization(contents)
	print ("Tokneization takes %s"%(time.time()-start_time))

	# Spacy Stemming

	# Spacy POS Tagging
	start_time = time.time()
	pos = pos_tagging(contents)
	print ("POS tagging takes %s"%(time.time()-start_time))


	# REGEX Macthing
	email = email_matching(contents)
	dates = date_matching(contents)





