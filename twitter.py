# author = rhnvrm <hello@rohanverma.net>

import os
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D,Conv1D,LSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

class TwitterClient(object):
    '''
    Generic Twitter Class for the App
    '''
    def __init__(self, query, retweets_only=False, with_sentiment=False):
        # keys and tokens from the Twitter Dev Console
        consumer_key = '0hTTBhVzE8v4Z7QfAfLk5tEkJ'
        consumer_secret = '6fizqcesSlzzjiVmxin6o0WaZfgVrY9ELuJruBcHr76UOQ2u82'
        access_token ='1214144886523355136-oKpnyRfh84bHohZOz4czF0QlHh77we'
        access_token_secret ='0eol7Ehfb7LNvQJw6hvcfh91CzPLrSwkpTCuKCW20iVl8'
        
        movie_reviews = pd.read_csv("/content/drive/My Drive/IMDB Dataset.csv")

        #movie_reviews.isnull().values.any()

        # movie_reviews.shape
        # movie_reviews.head()
        # import seaborn as sns

        # sns.countplot(x='sentiment', data=movie_reviews)

        def preprocess_text(sen):
            # Removing html tags
            sentence = remove_tags(sen)

            # Remove punctuations and numbers
            sentence = re.sub('[^a-zA-Z]', ' ', sentence)

            # Single character removal
            sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

            # Removing multiple spaces
            sentence = re.sub(r'\s+', ' ', sentence)

            return sentence

        TAG_RE = re.compile(r'<[^>]+>')

        def remove_tags(text):
            return TAG_RE.sub('', text)

        X = []
        sentences = list(movie_reviews['review'])
        for sen in sentences:
            X.append(preprocess_text(sen))



        y = movie_reviews['sentiment']

        y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X_train)
        model=load_model('keras_model.h5')
            # Attempt authentication
        try:
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            self.query = query
            self.retweets_only = retweets_only
            self.with_sentiment = with_sentiment
            self.api = tweepy.API(self.auth)
            self.tweet_count_max = 100  # To prevent Rate Limiting
        except:
            print("Error: Authentication Failed")

    def set_query(self, query=''):
        self.query = query

    def set_retweet_checking(self, retweets_only='false'):
        self.retweets_only = retweets_only

    def set_with_sentiment(self, with_sentiment='false'):
        self.with_sentiment = with_sentiment

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
       analysis = TextBlob(self.clean_tweet(tweet))
        if analysis.sentiment.polarity > 0:
            r= 'positive'
        elif analysis.sentiment.polarity == 0:
            r= 'neutral'
        else:
            r= 'negative'
        instance = tokenizer.texts_to_sequences(tweet)

        flat_list = []
        for sublist in instance:
            for item in sublist:
                flat_list.append(item)

        flat_list = [flat_list]

        instance = pad_sequences(flat_list, padding='post', maxlen=100)

        value=model.predict(instance)
        if(value>0.5):
            r1= 'positive'
        elif(value<0.4):
            r1= 'negative'
        else:
            r1= 'neutral'

        if(r=='positive' & r1=='positive'):
            return 'positive'
        elif(r=='positive' & r1=='negative'):
            return 'neutral'
        else(r=='negative' & r1=='negative' | (r=='negative' & r1=='positve' )):
            return 'negative'
        else(r=='positive' & r1=='positive'):
            return 'neutral'

    def get_tweets(self):
        tweets = []

        try:
            recd_tweets = self.api.search(q=self.query,
                                          count=self.tweet_count_max)
            if not recd_tweets:
                pass
            for tweet in recd_tweets:
                parsed_tweet = {}

                parsed_tweet['text'] = tweet.text
                parsed_tweet['user'] = tweet.user.screen_name
                
                if self.with_sentiment == 1:
                    parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
                else:
                    parsed_tweet['sentiment'] = 'unavailable'

                if tweet.retweet_count > 0 and self.retweets_only == 1:
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                elif not self.retweets_only:
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)

            return tweets

        except tweepy.TweepError as e:
            print("Error : " + str(e))
