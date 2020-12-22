#!/usr/bin/env python
# coding: utf-8

# # NLTK Sentiment Analysis Pipeline

# #### In this module, I walk you through all the methods available for sentiment analysis: The algorithms, the methods available, the ensemble, tokenizing and basically what we need to take the model from scratch to completion. I use the Natural Language Took Kit(NLTK) that inherits from the SKlearn module:

# In[1]:


# import modules
import nltk
import random #used to shuffle the pre labelled corpus of words
#from nltk.corpus import movie_reviews #has 1000 positive and 1000 negative movie reviews
import pickle
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier #includes sklearn's algorithms within the nltk module itself,its
#a wrapper to inclune the scikit learn algorithms within the NLTK classifier itself
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC,NuSVC
from nltk.classify import ClassifierI #so we can inherit from the NLTK classifier class
from statistics import mode #this is how we choose who got the most votes
# from matplotlib import style
# style.use("ggplot")


# In[2]:


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
        confidence_=choice_votes/len(votes) #how many chosen out of the totatl number of categories, this gives you a confidence
        #level
        return confidence_
        
        
            


# In[3]:


#we read in the new corpus that we want to train our algorithm on
s_pos_reviews=open("positive.txt","r").read()
s_neg_reviews=open("negative.txt","r").read()
all_words=[]
documents=[]


# In[5]:


allowed_words=["J"]


for p in s_pos_reviews.split("\n"):
    documents.append((p,"pos")) #this documents is a tuple, consisting of a review and whether ist's negative or positive
    words=word_tokenize(p)
    pos=nltk.pos_tag(words)
    for w in pos:
         if w[1][0] in allowed_words:
                all_words.append(w[0].lower())
        
for p in s_neg_reviews.split("\n"):     
    documents.append((p,"neg")) #this documents is a tuple, consisting of a review and whether ist's negative or positive
    words=word_tokenize(p)
    pos=nltk.pos_tag(words)
    for w in pos:
         if w[1][0] in allowed_words:
                all_words.append(w[0].lower())
        
# 
# pos_words=word_tokenize(s_pos_reviews)
# neg_words=word_tokenize(s_neg_reviews)
# for w in pos_words:
#     all_words.append(w.lower())
# for w in neg_words:
#     all_words.append(w.lower())
save_documents=open("pickled_docs.pickle","wb")
pickle.dump(documents,save_documents)
save_documents.close()


# In[6]:


#this part is for creating training and testing sets
#documents= [(list(movie_reviews.words(fileid)),category) for category in movie_reviews.categories() for fileid in movie_reviews
           #.fileids(category)]
#random.shuffle(documents)
#print(documents[1]) #this is just a list of words, and then it gives you the ratings of the words as either negative or positive
#we will take very word in every review and compile them and take the list of words and find the most common appearing
#in each category
#or we can do the below 
# documents=[]
# for category in movie_reviews.categories():
#     for fileid in movie_reviews.fileids(category):
#         documents.append(list(movie_reviews.words(fileid),category)
# x=random.shuffle(documents) #we do this to shuffle the documents for the train-test part of the data analysis to avoid extreme
#                          #bias
# print(x[0])


# In[7]:


#these are all the words from your corpus, that we will use to compare against the labelled set that we have generated above
#all_words=[]
# for w in movie_reviews.words():
#     all_words.append(w.lower())
all_words=nltk.FreqDist(all_words) # converts all words to an an NLTK frequency distsribution
print(all_words.most_common(15)) #shows you the most common words in a set that you have specified.
#print(all_words["stupid"]) #shows you how many times a word appears in a corpus
# #we use Naive Bayes Algorithm for  sentiment analysis
#we train based on the words in each document and how it is categorised....


# In[8]:


# we limit the list of words that we use for training our model, because as you see above some of them are quite useless for
#training the model
word_features=list(all_words.keys())[:5000]
save_word_features=open("word_features.pickle","wb")
pickle.dump(word_features,save_word_features)
save_word_features.close()


def find_features(document):
    words=word_tokenize(document)
    features={}
    for w in word_features:
        features[w]=(w in words) #the key in the number of words we have chosen is going to equal the boolean value for
#word we have chosen(if the word is in the document, this will show true, otherwise it shows false)
    return features
#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets=[(find_features(rev), category) for (rev, category) in documents ] #this converts the words to a dictionary where
#the word is the key and the and value is a true or false feature
random.shuffle(featuresets)
print(len(featuresets))
#sentiment analysis is based on the existence of words arounds a word and whether or not they convey a positive or negative
#feel
save_featuresets=open("featuresets.pickle", "wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()


# In[9]:


#creating the training and testing set portions of the data
training_set=featuresets[:10000]
testing_set=featuresets[10000:]
# posterior =prior occurences*likelihood/evidence, so likelihhod for something to be positive or negative
classifier=nltk.classify.NaiveBayesClassifier.train(training_set)
#classifier_f=open('naivebayes.pickle',"rb")
#classifier=pickle.load(classifier_f)
#classifier_f.close()
print("Algorithm Accuracy:", nltk.classify.accuracy(classifier,testing_set))
classifier.show_most_informative_features(20)

#pickle the algorithm
save_classifier=open("NB.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()


# In[6]:


# import nltk
# # nltk.download()
# import regex #this is a word tokenizer but the NLTK word tokenizer does a better job than the regex tokenizer.
# #the snowball stemmer supports multiple languages.
# import word2vec


# In[7]:


# conda install -c conda-forge word2vec


# In[8]:


# conda update -n base -c defaults conda


# In[9]:


#pickle is a way to save your trained algorithm so any time you wanna use the trained algorithm it's there for your use, so you
#don't have to retrain it
#pickle is a way we can save python objects and it's actually extremely powerful to use
# saved_classifier=open("naivebayes.pickle", "wb") #wb is write in bytes, open in with the intention to write in bytes
# pickle.dump(classifier, saved_classifier) #so we save our classifier defined above to a pickle object that we have defined using 
# #dump
# saved_classifier.close()


# In[10]:


#NLTK is a natural language took kit. It's not a machine learning toolkit


# In[10]:


#creating instances of the SklearnCalssifier and wrapping the Sklearn modules.
MNB=SklearnClassifier(MultinomialNB())#this creates a multinomialNB classifier for the NTLK module
MNB.train(training_set)
print("MNB Algorithm Accuracy:", nltk.classify.accuracy(MNB,testing_set))

#pickle MNB
save_MNB=open("MNB.pickle","wb")
pickle.dump(MNB,save_MNB)
save_MNB.close()

# Gaussian=SklearnClassifier(GaussianNB())#this creates a GaussianNB classifier for the NLTK module
# Gaussian.train(training_set)
# print("Gaussian Algorithm Accuracy:", nltk.classify.accuracy(GaussianNB,testing_set))

# Bernoulli=SklearnClassifier(BernoulliNB()).train(training_set)#this creates a BernoullilNB classifier for the NLTK module
# print("Bernoulli Algorithm Accuracy:", nltk.classify.accuracy(BernoulliNB,testing_set))


# In[11]:


Bernoulli_=SklearnClassifier(BernoulliNB())#this creates a BermoulliNB classifier for the NTLK module
Bernoulli_.train(training_set)
print("MNB Algorithm Accuracy:", nltk.classify.accuracy(MNB,testing_set))

#pickle Bernoulli
save_Bernoulli=open("Bernoulli_.pickle","wb")
pickle.dump(Bernoulli_,save_Bernoulli)
save_Bernoulli.close()


# In[12]:


#NLTK classifers, the ones with an underscore at the end are inheriting from SK learn classifiers
LogisticRegression_=SklearnClassifier(LogisticRegression())#this creates a Logistic classifier for the NTLK module
LogisticRegression_.train(training_set)
print("LogisticReg Algorithm Accuracy:", nltk.classify.accuracy(LogisticRegression_,testing_set))


#pickle LR
save_LR=open("LR.pickle","wb")
pickle.dump(LogisticRegression_,save_LR)
save_LR.close()


# In[ ]:


#######################################################################################################
# NuSVC_=SklearnClassifier(NuSVC())#this creates a NuSVC classifier for the NTLK module
# NuSVC_.train(training_set)
# print("NuSVC Algorithm Accuracy:", nltk.classify.accuracy(NuSVC_,testing_set))

# #pickle LSVC
# save_NUSVC=open("NUSVC.pickle","wb")
# pickle.dump(NuSVC_,save_NULSVC)
# save_NUSVC.close()


# In[ ]:


#####################################################################################################
# LinearSVC_=SklearnClassifier(LinearSVC())#this creates a LinearSVC classifier for the NTLK module
# LinearSVC_.train(training_set)
# print("SVC Algorithm Accuracy:", nltk.classify.accuracy(LinearSVC_,testing_set))

# #pickle LSVC
# save_LSVC=open("LSVC.pickle","wb")
# pickle.dump(LinearSVC_,save_LSVC)
# save_LSVC.close()


# In[ ]:


#######################################################################################################
# SVC_=SklearnClassifier(SVC())#this creates a SVC classifier for the NTLK module
# SVC_.train(training_set)
# print("LinearSVC_ Algorithm Accuracy:", nltk.classify.accuracy(SVC_,testing_set))

# #pickle SVC
# save_SVC=open("SVC.pickle","wb")
# pickle.dump(SVC_,save_SVC)
# save_SVC.close()


# In[13]:


###################################################################################################
SGDClassifier_=SklearnClassifier(SGDClassifier())#this creates a SGD classifier for the NTLK module
SGDClassifier_.train(training_set)
print("SGDClassifier Algorithm Accuracy:", nltk.classify.accuracy(SGDClassifier_,testing_set))


#pickle SGD
save_SGD=open("SGD.pickle","wb")
pickle.dump(SGDClassifier_,save_SGD)
save_SGD.close()


# In[14]:


# X=[LogisticRegression,SGDClassifier, SVC, LinearSVC,NuSVC,MultinomialNB]
# def Class_Algo(training_set_, testing_set_):
#     for i in X:
#         i_=SklearnClassifier(i())
#         i_.train(training_set_)
#         print(i , nltk.classify.accuracy(i_,testing_set_))
# Class_Algo(training_set, testing_set)


# In[1]:


# #combining Algos with a vote, Ensemble methods.Raises accuracy by a few points but also raises the reliability and allows us
# #to introduce a confidence measure
voted_classifier=Algo_votes(LogisticRegression_,SGDClassifier_,Bernoulli_,MNB,classifier)
print("voted_classifier accuracy percent:",(nltk.classify.accuracy(voted_classifier,testing_set))*100)
print()
print("classification:", voted_classifier.classify(testing_set[0][0]),"confidence%:",voted_classifier.confidence(testing_set[0][0]))
#so basically all algorithms agree that there is no positive sentiment from the words we've chosen


# In[16]:


#we can pass in the words through find-features function as along as we know about word features, we can connect to  twitter feed,
#and anlyze some sort of key word
#what is the distribution between our accuracy on positive information and accuracy on negative information
#we can stop shuffling data to achieve this
#negative data example
# training_set1=featuresets[100:]
# testing_set1=featuresets[:1900]
# X=[LogisticRegression,SGDClassifier, SVC, LinearSVC,NuSVC,MultinomialNB]
# def Class_Algo(training_set_, testing_set_):
#     for i in X:
#         i_=SklearnClassifier(i())
#         i_.train(training_set_)
#         print(i , nltk.classify.accuracy(i_,testing_set_))
# Class_Algo(training_set1, testing_set1)


# In[1]:


#the sentiment function......
def sentiment(text):
    feats=find_features(text)
    return LogisticRegression_.classify(feats), LogisticRegression_.confidence(feats)


# In[ ]:




