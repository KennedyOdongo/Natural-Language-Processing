#!/usr/bin/env python
# coding: utf-8

# # Sentiment and web scraping analysis with Python's NLTK

# #### Import modules:

# In[2]:


#import modules.
import Sent as S #this line imports the module named 'Sentiment which is skeleton of the NLP pipeline, that we use for analysis'
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import matplotlib.pyplot as plt
import matplotlib as style
import matplotlib.animation as animation
import time


# In[3]:


#graph style to use.
plt.style.use('ggplot')


# ### To Access twitter data or any other platform's data, we can either scrape it or we can connect to an API. Using API's is easier

# In[4]:


#get api-keys.... We use api keys to to access about 1% of twitter data, roughly 5 million tweets a day, that's a lot of data.
from API_KEYS import api_key,api_secret,atoken,asecret


# In[5]:


# these are the variables we pass into the Python class below to get data from the twitter API
authorization=OAuthHandler(api_key, api_secret)
authorization.set_access_token(atoken, asecret)


# ## An example with a common word: "Facebook"

# In[6]:


#build listen class..Listen because we are listening to tweets.., for this case we are collecting tweets about facebook
#This happens in real time. This will collect the tweets for as long as you let it.... so I set my limit to 20.
collection=[]
class Listen(StreamListener):
    def __init__(self, api=None):
        super(Listen, self).__init__()
        self.num_tweets = 0

    def on_status(self, status):
        record = {'Text': status.text, 'Created At': status.created_at}
        print (record)
        self.num_tweets += 1
        if self.num_tweets < 20:
            collection.append(record)
            return True
        else:
            return False
stream = Stream(authorization, Listen())
stream.filter(track=['Facebook'])   


# ## An example with the word:"Genome"

# In[24]:


#now let's try that with genome and  genome editing.......   
class Listen(StreamListener):
    def __init__(self, api=None):
        super(Listen, self).__init__()
        self.num_tweets = 0

    def on_status(self, status):
        record = {'Text': status.text, 'Created At': status.created_at}
        print (record)
        self.num_tweets += 1
        if self.num_tweets < 10:
            collection.append(record)
            return True
        else:
            return False
stream = Stream(authorization, Listen())
stream.filter(track=['genome'])  
#it takes a significantly longer time to collect data on the word genome as compared to the a common word like "Facebook"


# #### After we get these tweets, we can use the sentiment module for some preprocessing then we can run our analysis.

# In[ ]:




