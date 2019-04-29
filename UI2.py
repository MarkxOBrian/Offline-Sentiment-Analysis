from PyQt5 import QtCore, QtGui, uic, QtWidgets
# from PyQt5. import QApplication

import sys, tweepy, csv, re
from textblob import TextBlob
import matplotlib.pyplot as plt

import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

qtCreatorFile = "C:/Users/FIXO/Desktop/4.2/Projecct II/Sentiment Analysis Project/UI/interface2.ui"  # Enter file here.

qtCreatorFile2 = "C:/Users/FIXO/Desktop/4.2/Projecct II/Sentiment Analysis Project/UI/otherWindow.ui"

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

Ui_MainWindow2, QtBaseClass2 = uic.loadUiType(qtCreatorFile2)

"""
class MyApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.getInput)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())





def getInput(self):
    searchTermInput= int(self.hashtag.toPlainText())

    total_price = price  + ((tax / 100) * price)
    total_price_string = "The total price with tax is: " + str(total_price)
    self.results_window.setText(total_price_string)

"""



class OtherWindow(QtWidgets.QMainWindow, Ui_MainWindow2):
    tweet = ""
    count = 0
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow2.__init__(self)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.callInitialFunction)







    def callInitialFunction(self):
        global tweet

        tweet = self.enterTweet.toPlainText()
        ow = OtherWindow()
        ow.prepareModel()
        global count
        if count == 1:
            self.result.setText("Negative")
        elif count == 4:
            self.result.setText("Positive")

    def prepareModel(self):
        df = pd.read_csv('C:/Users/FIXO/Desktop/4.2/Projecct II/datasets/trainingDateset.txt', sep='\t',
                         names=['liked', 'txt'])
        # print(df.head(10))
        # df.dropna()
        df = df.dropna(how='any', axis=0)

        # TDIDF Vectorizer
        stopset = set(stopwords.words('english'))
        vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)

        # in this case, our dependent variable will be liked as 0 (didn't like the movie) or 4 ( liked the movie)



        y = df.liked
        # y = y.as_matrix().astype(np.float)

        # convert df.txt from text to features
        X = vectorizer.fit_transform(df['txt'].values.astype('U'))
        # X = X.as_matrix().astype(np.float)
        # print(y.shape)
        # print(X.shape)

        # test Train split
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        # training using Naive_bayes classifier
        clf = naive_bayes.MultinomialNB()
        clf.fit(X_train, y_train)

        # testing accuracy of model
        print(roc_auc_score(y_test, clf.predict(X_test)))

        my_array = np.array([tweet])

        my_vectorizer = vectorizer.transform(my_array)

        for tweet2 in my_vectorizer:
            print(clf.predict(tweet2))
            result = str(clf.predict(tweet2))
            #final = result.tostring()
            global count
            if result == "[0.]":

                count = 1
                print("Negative")
            else:
                count = 4
                print("positive")

if __name__== "__main__":

    app = QtWidgets.QApplication(sys.argv)
    window = OtherWindow()
    window.show()
    sys.exit(app.exec_())
