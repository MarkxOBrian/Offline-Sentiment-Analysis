
from PyQt5 import QtCore, QtGui, uic, QtWidgets
#from PyQt5. import QApplication

import sys,tweepy,csv,re
from textblob import TextBlob
import matplotlib.pyplot as plt

import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import  roc_auc_score
from sklearn.linear_model import LogisticRegression

from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#import twitter_credentials
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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




class SentimentAnalysis (QtWidgets.QMainWindow, Ui_MainWindow):
    searchTermInput = ""
    noOfTweetsInput = 1
    valueOfSentiment = ""
    myValue = 0


    def __init__(self):
        self.tweets = []
        self.tweetText = []
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.callInitialFunctions)
        #self.pushButton.clicked.connect(self.callInitialFunctions)
        self.outputWindow.setText("General Report: ")

        """
        global valueOfSentiment
        if myValue == 1:
            self.outputWindow.setText(valueOfSentiment)
            """

        self.pushButton_2.clicked.connect(self.newWindow)
        self.statistics.clicked.connect(self.getUserInput)
        self.pushButton_3.clicked.connect(self.getStatistics)

    def getStatistics(self):

        import time
        from tweepy import Stream
        from tweepy import OAuthHandler
        from tweepy.streaming import StreamListener
        import json
        from textblob import TextBlob
        import matplotlib.pyplot as plt
        import re
        import test

        "# -- coding: utf-8 --"

        def calctime(a):
            return time.time() - a

        positive = 0
        negative = 0
        compound = 0

        count = 0
        initime = time.time()
        plt.ion()



        consumer_key = "rbjc5Bc9L5cO7Xtq9IN3oHekg"
        consumer_secret = "njTD2N1ANx9piP4PUjwCgRhmwihLUzTH46jUykJZKnkmiKmut9"
        access_token = "4866833991-iA61aJuWNRpxN7PkGNCOL9zp6vvQ1VqTQJbuZEf"
        access_secret = "jfiU8S5kNb7hgaecwO9ZP0bJobT6RoXyI6HbMXz3upkQ6"

        class listener(StreamListener):
            def on_data(self, data):
                global initime
                t = int(calctime(initime))
                all_data = json.loads(data)
                tweet = all_data["text"].encode("utf-8")
                # username=all_data["user"]["screen_name"]
                tweet = " ".join(re.findall(b"[a-zA-Z].decode('utf-8', 'backslashreplace')+", tweet))
                blob = TextBlob(tweet.strip())

                global positive
                global negative
                global compound
                global count

                count = count + 1
                senti = 0
                for sen in blob.sentences:
                    senti = senti + sen.sentiment.polarity
                    if sen.sentiment.polarity >= 0:
                        positive = positive + sen.sentiment.polarity
                    else:
                        negative = negative + sen.sentiment.polarity
                compound = compound + senti
                print(count)
                print(tweet.strip())
                print(senti)
                print(t)
                print(str(positive) + ' ' + str(negative) + ' ' + str(compound))

                plt.axis([0, 70, -20, 20])
                plt.xlabel('Time')
                plt.ylabel('Sentiment')
                plt.plot([t], [positive], 'go', [t], [negative], 'ro', [t], [compound], 'bo')
                plt.pause(0.0001)
                plt.show()

                if count == 200:
                    return False
                else:
                    return True

            def on_error(self, status):
                print(status)

        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)

        twitterStream = Stream(auth, listener(count))
        twitterStream.filter(track=["Donald trump"])

    def getUserInput(self):
            global searchTermInput
            global noOfTweetsInput
            searchTermInput = str(self.hashtag.toPlainText())
            noOfTweetsInput = int(self.noOfTweets.toPlainText())

            twitter_client = TwitterClient()
            tweet_analyzer = TweetAnalyzer()

            api = twitter_client.get_twitter_client_api()

            tweets = api.user_timeline(screen_name="realDonaldTrump", count=20)

            # print(dir(tweets[0]))
            # print(tweets[0].retweet_count)

            df = tweet_analyzer.tweets_to_data_frame(tweets)

            # Get average length over all tweets:
            print(np.mean(df['len']))

            # Get the number of likes for the most liked tweet:
            print(np.max(df['likes']))

            # Get the number of retweets for the most retweeted tweet:
            print(np.max(df['retweets']))

            # print(df.head(10))

            # Time Series
            # time_likes = pd.Series(data=df['len'].values, index=df['date'])
            # time_likes.plot(figsize=(16, 4), color='r')
            # plt.show()

            # time_likes = pd.Series(data=df['likes'].values, index=df['date'])
            # time_likes.plot(figsize=(16, 4), legend=True)
            # plt.show()

            # time_retweets = pd.Series(data=df['retweets'].values, index=df['date'])
            # time_retweets.plot(figsize=(16, 4), legend=True)
            # plt.show()

            # Layered Time Series:
            time_likes = pd.Series(data=df['likes'].values, index=df['date'])
            time_likes.plot(figsize=(16, 4), label="likes", legend=True)

            time_retweets = pd.Series(data=df['retweets'].values, index=df['date'])
            time_retweets.plot(figsize=(16, 4), label="retweets", legend=True)
            plt.show()
            plt.close('all')



    def callInitialFunctions(self):
        global searchTermInput
        global noOfTweetsInput
        searchTermInput = str(self.hashtag.toPlainText())
        noOfTweetsInput = int(self.noOfTweets.toPlainText())
        sa = SentimentAnalysis()
        sa.DownloadData()
       # sa.print_tweets()




    #creating another window
    def newWindow(self):
        myOtherWindow = OtherWindow()
        myOtherWindow.show()


    def DownloadData(self):
        # authenticating
        consumerKey = "rbjc5Bc9L5cO7Xtq9IN3oHekg"
        consumerSecret = "njTD2N1ANx9piP4PUjwCgRhmwihLUzTH46jUykJZKnkmiKmut9"
        accessToken = "4866833991-iA61aJuWNRpxN7PkGNCOL9zp6vvQ1VqTQJbuZEf"
        accessTokenSecret = "jfiU8S5kNb7hgaecwO9ZP0bJobT6RoXyI6HbMXz3upkQ6"
        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth)
        """
        tweets_data = api.home_timeline()
        for tweet in tweets_data:
            print(tweet.id, " : ", tweet.text)
        """


        # input for term to be searched and how many tweets to search
       # searchTermInput = input("Enter Keyword/Tag to search about: ")
       # noOfTweetsInput = int(input("Enter how many tweets to search: "))

        # searching for tweets
        self.tweets = tweepy.Cursor(api.search, q=searchTermInput, lang = "en").items(noOfTweetsInput)

        # Open/create a file to append data to
        csvFile = open('result.csv', 'a')

        # Use csv writer
        csvWriter = csv.writer(csvFile)


        # creating some variables to store info
        polarity = 0
        positive = 0
        wpositive = 0
        spositive = 0
        negative = 0
        wnegative = 0
        snegative = 0
        neutral = 0


        # iterating through tweets fetched
        for tweet in self.tweets:
            #Append to temp so that we can store in csv later. I use encode UTF-8
            self.tweetText.append(self.cleanTweet(tweet.text).encode('utf-8'))
            # print (tweet.text.translate(non_bmp_map))    #print tweet's text
            analysis = TextBlob(tweet.text)
            # print(analysis.sentiment)  # print tweet's polarity
            polarity += analysis.sentiment.polarity  # adding up polarities to find the average later

            #if (analysis.sentiment == 0):  # adding reaction of how people are reacting to find average later
               # neutral += 1


            if (analysis.sentiment.polarity > 0):
                positive += 1


            elif (analysis.sentiment.polarity < 0):
                negative += 1



        # Write to csv and close csv file
        csvWriter.writerow(self.tweetText)
        csvFile.close()

        # finding average of how people are reacting
        no = noOfTweetsInput - (positive + negative)
        neutral = self.percentage(no, noOfTweetsInput)
        positive = self.percentage(positive, noOfTweetsInput)


        negative = self.percentage(negative, noOfTweetsInput)


        #neutral = self.percentage(neutral, noOfTweetsInput)



        # finding average reaction
        polarity = polarity / noOfTweetsInput

        # printing out data
        print("How people are reacting on " + searchTermInput + " by analyzing " + str(noOfTweetsInput) + " tweets.")
        print()
        self.outputWindow.setText("General Report: ")
        print("General Report: ")

        global valueOfSentiment
        if (polarity == 0):
            print("Neutral")
            valueOfSentiment = "Neutral"

        elif (polarity > 0.3 and polarity <= 0.6):
            print("Positive")
            valueOfSentiment = "Positive"


        elif (polarity > -0.6 and polarity <= -0.3):
            print("Negative")
            valueOfSentiment = "Negative"


        print()
        print("Detailed Report: ")
        print(str(positive) + "% people thought it was positive")

        print(str(negative) + "% people thought it was negative")

        print(str(neutral) + "% people thought it was neutral")

        self.plotPieChart(positive, negative, neutral, searchTermInput, noOfTweetsInput)


    def cleanTweet(self, tweet):
        # Remove Links, Special Characters etc from tweet
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())

    # function to calculate percentage
    def percentage(self, part, whole):
        temp = 100 * float(part) / float(whole)
        return format(temp, '.2f')

    def plotPieChart(self, positive, negative, neutral, searchTermInput, noOfsearchTermInputs):
        labels = ['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]',
                  'Negative [' + str(negative) + '%]']
        sizes = [positive, neutral, negative,]
        colors = ['darkgreen', 'gold', 'red']
        patches, texts = plt.pie(sizes, colors=colors, startangle=90)
        plt.legend(patches, labels, loc="best")
        plt.title('How people are reacting on ' + searchTermInput + ' by analyzing ' + str(noOfsearchTermInputs) + ' Tweets.')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        global  myValue
        myValue = 1

    def get_tweet_sentiment(self, tweet):
        ''' 
        Utility function to classify sentiment of passed tweet 
        using textblob's sentiment method 
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.cleanTweet(tweet))
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def get_tweets(self, query, count4):
        ''' 
        Main function to fetch tweets and parse them. 
        '''
        # empty list to store parsed tweets
        tweets = []


        try:
            # call twitter api to fetch tweets
            fetched_tweets = self.api.search(q=query, count=count4)

            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}

                # saving text of tweet
                parsed_tweet['text'] = tweet.text
                # saving sentiment of tweet
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)

                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)

                    # return parsed tweets
            return tweets

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))

    def print_tweets(self):
        global searchTermInput
        global noOfTweetsInput
        api = TwitterClient()
        # calling function to get tweets

        tweets = self.api.get_tweets(query=searchTermInput, count=noOfTweetsInput)

        # picking positive tweets from tweets
        ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
        # percentage of positive tweets
        print("Positive tweets percentage: {} %".format(100 * len(ptweets) / len(tweets)))
        # picking negative tweets from tweets
        ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
        # percentage of negative tweets
        print("Negative tweets percentage: {} %".format(100 * len(ntweets) / len(tweets)))
        # percentage of neutral tweets
        print("Neutral tweets percentage: {} % \
                  ".format(100 * len(tweets - ntweets - ptweets) / len(tweets)))

        # printing first 5 positive tweets
        print("\n\nPositive tweets:")
        for tweet in ptweets[:10]:
            print(tweet['text'])

        # printing first 5 negative tweets
        print("\n\nNegative tweets:")
        for tweet in ntweets[:10]:
            print(tweet['text'])


CONSUMER_KEY = "rbjc5Bc9L5cO7Xtq9IN3oHekg"
CONSUMER_SECRET = "njTD2N1ANx9piP4PUjwCgRhmwihLUzTH46jUykJZKnkmiKmut9"
ACCESS_TOKEN = "4866833991-iA61aJuWNRpxN7PkGNCOL9zp6vvQ1VqTQJbuZEf"
ACCESS_TOKEN_SECRET = "jfiU8S5kNb7hgaecwO9ZP0bJobT6RoXyI6HbMXz3upkQ6"

# # # # TWITTER CLIENT # # # #
class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets

    def get_friend_list(self, num_friends):
        friend_list = []
        for friend in Cursor(self.twitter_client.friends, id=self.twitter_user).items(num_friends):
            friend_list.append(friend)
        return friend_list

    def get_home_timeline_tweets(self, num_tweets):
        home_timeline_tweets = []
        for tweet in Cursor(self.twitter_client.home_timeline, id=self.twitter_user).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets


# # # # TWITTER AUTHENTICATER # # # #
class TwitterAuthenticator():
    def authenticate_twitter_app(self):
        """
        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        api = tweepy.API(auth)
        """
        auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        return auth


# # # # TWITTER STREAMER # # # #
class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """

    def __init__(self):
        self.twitter_autenticator = TwitterAuthenticator()

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_autenticator.authenticate_twitter_app()
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords:
        stream.filter(track=hash_tag_list)


# # # # TWITTER STREAM LISTENER # # # #
class TwitterListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """

    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True

    def on_error(self, status):
        if status == 420:
            # Returning False on_data method in case rate limit occurs.
            return False
        print(status)


class TweetAnalyzer():
    """
    Functionality for analyzing and categorizing content from tweets.
    """

    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['tweets'])

        df['id'] = np.array([tweet.id for tweet in tweets])
        df['len'] = np.array([len(tweet.text) for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['source'] = np.array([tweet.source for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])

        return df

"""
if __name__ == '__main__':
    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()

    api = twitter_client.get_twitter_client_api()

    tweets = api.user_timeline(screen_name="realDonaldTrump", count=20)

    # print(dir(tweets[0]))
    # print(tweets[0].retweet_count)

    df = tweet_analyzer.tweets_to_data_frame(tweets)

    # Get average length over all tweets:
    print(np.mean(df['len']))

    # Get the number of likes for the most liked tweet:
    print(np.max(df['likes']))

    # Get the number of retweets for the most retweeted tweet:
    print(np.max(df['retweets']))

    # print(df.head(10))

    # Time Series
    #time_likes = pd.Series(data=df['len'].values, index=df['date'])
    #time_likes.plot(figsize=(16, 4), color='r')
    #plt.show()

    #time_likes = pd.Series(data=df['likes'].values, index=df['date'])
    #time_likes.plot(figsize=(16, 4), legend=True)
    #plt.show()

    #time_retweets = pd.Series(data=df['retweets'].values, index=df['date'])
    #time_retweets.plot(figsize=(16, 4), legend=True)
    #plt.show()

    # Layered Time Series:
    time_likes = pd.Series(data=df['likes'].values, index=df['date'])
    time_likes.plot(figsize=(16, 4), label="likes", legend=True)

    time_retweets = pd.Series(data=df['retweets'].values, index=df['date'])
    time_retweets.plot(figsize=(16, 4), label="retweets", legend=True)
    plt.show()
    plt.close('all')
"""


if __name__== "__main__":

    app = QtWidgets.QApplication(sys.argv)
    window = SentimentAnalysis()
    window.show()
    sys.exit(app.exec_())
