import json 
import pandas as pd
import numpy as np
import nltk
import string
import sys
from multiprocessing import Pool
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.svm import SVC

# nltk.download('punkt') ###Enable when you submit this assignment
# nltk.download('stopwords') ###Enable when you submit this assignment

def clean_string(x):
	x = x[2:-2]
	x = x.replace("', '", ' ')
	x = x.replace("', \"", ' ')
	x = x.replace("\", \"", ' ')
	x = x.replace("\", '", ' ')
	return x

df_train = pd.read_csv('data/train-pp.csv')
del df_train['Unnamed: 0']
df_train['review_pp'] = pd.Series(list(map(lambda x : clean_string(x), df_train['review_pp'])))

df_dev = pd.read_csv('data/dev-pp.csv')
del df_dev['Unnamed: 0']
df_dev['review_pp'] = pd.Series(list(map(lambda x : clean_string(x), df_dev['review_pp'])))

corpus_train = np.asarray(df_train["review_pp"])
y_train = np.asarray(list(map(int, df_train["ratings"])))

if sys.argv[1] == '1' or sys.argv[1] == '3':
	vectorizer = CountVectorizer()
if sys.argv[1] == '2' or sys.argv[1] == '4':
	vectorizer = TfidfVectorizer()

vectorizer.fit(corpus_train)
X_train = vectorizer.transform(corpus_train)

corpus_dev = np.asarray(df_dev["review_pp"])
X_dev = vectorizer.transform(corpus_dev)
y_dev = np.asarray(list(map(int, df_dev["ratings"])))

if sys.argv[1] == '1' or sys.argv[1] == '2':
	clf = MultinomialNB()
	clf.fit(X_train, y_train)
	print(clf.score(X_dev, y_dev))
	y_predicted = clf.predict(X_dev)
	print("Confusion Matrix for Classifier:")
	print(confusion_matrix(y_dev,y_predicted))
	print("Score: ",round(accuracy_score(y_dev,y_predicted))*100,2)
	print("Classification Report:")
	print(classification_report(y_dev,y_predicted))

	if sys.argv[1] == '1':
		print('NB - Count Vectors')
	if sys.argv[1] == '2':
		print('NB - TF-IDF Vectors')

if sys.argv[1] == '3' or sys.argv[1] == '4':
	clf = SVC(kernel='linear', gamma='auto')
	clf.fit(X_train, y_train)
	print(clf.score(X_dev, y_dev))
	if sys.argv[1] == '3':
		print('SVM - Count Vectors')
	if sys.argv[1] == '4':
		print('SVM - TF-IDF Vectors')