#!/usr/bin/env python
# coding: utf-8

# # NLTK  Analysis 

# In[1]:


# import modules
import nltk
import random #used to shuffle the pre labelled corpus of words
#from nltk.corpus import movie_reviews #has 1000 positive and 1000 negative movie reviews
import pickle
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier #includes sklearn's algorithms within the nltk module itself,its
#a wrapper to include the scikit learn algorithms within the NLTK classifier itself
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC,NuSVC
from nltk.classify import ClassifierI #so we can inherit from the NLTK classifier class
from statistics import mode #this is how we build the ensemble
# from matplotlib import style
# style.use("ggplot")


# ## Ensemble:

# In[2]:


#This is how we build the ensemble
class Algo_votes(ClassifierI):
    def __init__(self, *classifiers): #we are going to pass a list of vote classifiers through our vote classifier
        self._classifiers=classifiers
    def classify(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        return mode(votes)
    def confidence(self, features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        choice_votes=votes.count(mode(votes))
        confidence_=choice_votes/len(votes) #how many chosen out of the total number of categories, this gives you a confidence
        #level
        return confidence_


# ### Pickling(saving) the Algorithms so that we do not have to retrain them each time we run the script:

# In[3]:


save_documents=open("pickled_docs.pickle","rb")
documents=pickle.load(save_documents)
save_documents.close()


# In[4]:


save_word_features=open("word_features.pickle","rb")
word_features=pickle.load(save_word_features)
save_word_features.close()


# In[5]:


def find_features(document):
    words=word_tokenize(document)
    features={}
    for w in word_features:
        features[w]=(w in words) #the key in the number of words we have chosen is going to equal the boolean value for
#word we have chosen(if the word is in the document, this will show true, otherwise it shows false)
    return features

save_featuresets=open("featuresets.pickle", "rb")
featuresets=pickle.load(save_featuresets)
save_featuresets.close()


# #### Because the order of examples can affect the way the algorithms train, it is important to shufffle them, as shown below:

# In[6]:


random.shuffle(featuresets)
print(len(featuresets))


# #### Create the testing and training sets:

# In[7]:


testing_set=featuresets[10000:]
training_set=featuresets[:10000]


# #### Below I pickle all the classifiers I have trained: Naive Bayes,Multinomial Naive Bayes, Bernoulli,Logistic Regression and SGD classifier.

# In[8]:


open_file=open("NB.pickle","rb")
classifier=pickle.load(open_file)
open_file.close()


# In[9]:


open_file=open("MNB.pickle","rb")
MNB=pickle.load(open_file)
open_file.close()


# In[10]:


open_file=open("Bernoulli_.pickle","rb")
Bernoulli_=pickle.load(open_file)
open_file.close()


# In[11]:


open_file=open("LR.pickle","rb")
LogisticRegression_=pickle.load(open_file)
open_file.close()


# In[12]:


open_file=open("SGD.pickle","rb")
SGDClassifier_=pickle.load(open_file)
open_file.close()


# #### The voted classifier below inherits the class above and returns the most common sentiment (negative or positive) among the five classifiers that we trainied:

# In[13]:


voted_classifier=Algo_votes(LogisticRegression_,SGDClassifier_,Bernoulli_,MNB,classifier)


# #### The function below takes in text: A word or a sentence and tells us if that piece of text is positive or negative

# In[16]:


def sentiment(text):
    feats=find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

sentiment("genome") # The word genome is classified as negative, and all the classifiers agree that it is negative:


# In[ ]:




