import json 
import pandas as pd
import numpy as np
import nltk
import string
from multiprocessing import Pool
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
# nltk.download('punkt') ###Enable when you submit this assignment
# nltk.download('stopwords') ###Enable when you submit this assignment

#-----READ DATA------

data = []
for line in open('data/train.json', 'r'):
	data.append(json.loads(line))

df_train = pd.DataFrame.from_dict(data, orient='columns')
print("read training data")

data = []
for line in open('data/dev.json', 'r'):
	data.append(json.loads(line))

df_dev = pd.DataFrame.from_dict(data, orient='columns')
print("read dev data")

# For parallelisation
num_partitions = 100 #number of partitions to split dataframe
num_cores = 25 #number of cores on your machine
porter_stemmer = PorterStemmer()

def parallelize_dataframe(df, func):
	df_split = np.array_split(df, num_partitions)
	print('split the dataset')
	pool = Pool(num_cores)
	df = pd.concat(pool.map(func, df_split))
	pool.close()
	pool.join()
	return df

def stemmer(sen_tokenized):
	sen_post_stem = []
	for word in sen_tokenized:
		if word in string.punctuation:
			continue
		if word in stopwords.words('english'):
			continue
		sen_post_stem.append(porter_stemmer.stem(word))
	return sen_post_stem


def token_and_stem(data):
	data['review_pp'] = data['review'].apply(lambda x: stemmer(nltk.word_tokenize(x)))
	# print(str(data.name) + " done")
	return data

df_train1 = parallelize_dataframe(df_train, token_and_stem)
df_train1.to_json('data/train-pp.json')
del df_train1['review']
df_train1.to_csv('data/train-pp.csv')
print('train pre-processing done')

df_dev1 = parallelize_dataframe(df_dev, token_and_stem)
df_dev1.to_json('data/dev-pp.json')
del df_dev1['review']
df_dev1.to_csv('data/dev-pp.csv')
print('dev pre-processing done')


#Tokenization + Stemming
# for i in range(len(df_train)):
# 	tokenized_sen = nltk.word_tokenize(df_train.loc[i, "review"])
# 	# df_train.loc[i, "review"] = tokenized_sen
# 	print(str(i)+" tokenization done.")
# 	sen_pre_stem = tokenized_sen
# 	sen_post_stem = []
# 	for word in sen_pre_stem:
# 		sen_post_stem.append(porter_stemmer.stem(word))
# 	df_train.loc[i, "review"] = str(sen_post_stem)
# 	print(str(i)+" stemming done.")

# 	# df_train.loc[, "review_pp"] = tokenized_sen