import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import  roc_auc_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('D:/Work in Progress/Sentiment Analysis Twitter/Offline Sentiment/trainingDateset.txt', sep='\t', names=['liked','txt'])
#print(df.head(10))
#df.dropna()
df = df.dropna(how='any',axis=0)


#TDIDF Vectorizer
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)

#in this case, our dependent variable will be liked as 0 (didn't like the movie) or 4 ( liked the movie)



y = df.liked
#y = y.as_matrix().astype(np.float)

#convert df.txt from text to features
X = vectorizer.fit_transform(df['txt'].values.astype('U'))
#X = X.as_matrix().astype(np.float)
#print(y.shape)
#print(X.shape)

#test Train split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#training using Naive_bayes classifier
clf = naive_bayes.MultinomialNB()
clf.fit(X_train, y_train)

#testing accuracy of model
print(roc_auc_score(y_test,clf.predict(X_test)) )


df2 = pd.read_csv('C:/Users/FIXO/Desktop/4.2/Projecct II/datasets/data (1).csv', sep='\t')

df = np.array(["I love kenya"])
movie_vectorizer = vectorizer.transform(df)
#print(clf.predict(movie_vectorizer))

for tweet in movie_vectorizer:
    print(clf.predict(tweet))

# picking positive tweets from tweets

