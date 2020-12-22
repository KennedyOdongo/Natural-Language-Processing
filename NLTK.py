#!/usr/bin/env python
# coding: utf-8

# # Natural Language Tool Kit

# In[1]:


#importing the modules.
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer#for tokenizing
from nltk.corpus import stopwords, state_union,WordListCorpusReader,wordnet #imports stopwords. This allows you to import the specific corpus(body of words that
#you can use for sentiment analysis) #imports wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer #for stemming, for lemmatizing
 


# In[2]:


# nltk.download()


# In[3]:


# lemmatizer=WordNetLemmatizer()
# print(lemmatizer.lemmatize('cats'))


# In[4]:


# #shows you where the module you intsalled is located
#print(nltk.__file__)


# In[5]:


# from nltk.corpus import gutenberg
# sample= gutenberg.raw('bible-kjv.txt')
# tok=sent_tokenize(sample)
# print(tok[5:15])


# In[9]:


#let's say we want synonyms for a word we can use the synonyms in wordnet
syns=wordnet.synsets('good') #this prints a list and you can reference elements in that list.
print(syns)
synset= print(syns[0].name())
# just the word print(syns[0].lemmas[0].name())
#we can use a 'good.n.01' as search criteria as well


# In[7]:


# print(syns[0].lemmas()[1].name())


# In[8]:


# #get all the words from the list of lemmas
# for i in syns[0].lemmas():
#     print(i.name())


# In[9]:


# #definition
# print(syns[0].definition())
# #examples
# print(syns[0].examples())


# In[10]:


# #getting synonyms and antonyms:
# synonyms=[]
# antonyms=[]
# for word in wordnet.synsets('good'): #these are just synonyms
#     for l in word.lemmas():
#         synonyms.append(l.name())
#         if l.antonyms():
#             antonyms.append(l.antonyms()[0].name())
# print(set(synonyms))
# print(set(antonyms))
#wordnet.synsets gives you all of the synonyms of a word


# In[11]:


# #semantic similarities
# word1=wordnet.synset("ship.n.01")
# word2=wordnet.synset("boat.n.01")
# #to compare semantic similarity we use the method from wu and palmer paper form the 1990's
# print(word1.wup_similarity(word2))


# In[12]:


# #semantic similarities
# word3=wordnet.synset("man.n.01")
# word4=wordnet.synset("dog.n.01")
# #to compare semantic similarity we use the method from wu and palmer paper form the 1990's
# print(word3.wup_similarity(word4)) #compare the similarity of word 1 to word 2


# In[13]:


#print stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)


# In[24]:


#chunking: Groups of nouns, and modifiers sorrounding that noun.The way we chunk is that we use the POS tags and regular 
#expressions
train_text=state_union.raw('2005-GWBush.txt')
sample_text=state_union.raw('2006-GWBush.txt')
custom_sent_tokenizer= PunktSentenceTokenizer(train_text) #trains the tokenizer on the training text
tokenized=custom_sent_tokenizer.tokenize(sample_text) #we want to see how it performs
# def fin_sent_tokenize():
#     try:
#         for w in tokenized:
#             words=nltk.word_tokenize(w)
#             tagged=nltk.pos_tag(words)
#             print(tagged)
#     except Exception as e:
#         print(str(e))
# fin_sent_tokenize()


# In[15]:


# #example from Sentdex
# # def fin_sent_tokenize():
# #     for w in tokenized:
# #         words=nltk.word_tokenize(w)
# #         tagged=nltk.pos_tag(words)
#             #the r infront is to say that this is a form of regular expressions. Put the chunk that you wanna find inside
#             #curly braces, to mention any part of speech tag use the angle brackets,add thev part of speech you wanna find,
#             #then followed by the criteria of how you wanna find it for example "any character after it we use a fullstop(.)"
#             #for example<RB.?>* we are looking for an adverb and we want to find 0 or more of those
#         Chunk_=r"""Chunk: {<RB.?>*<VB.?>*<NNP><NN>} """
#         chunkParser=nltk.RegexParser(Chunk_)
#         chunked=chunkParser.parse(tagged)
#         print(chunked.draw())


# In[18]:


#chinking: You chink something form a chunk. It basically is the removal of something
#the way you denote a chink is by using opposite signs curly brackets }{, whatever we wanna keep out goes in between the chunk
#brackets


# In[ ]:


# #named Entity recognition, Using the same function above
# def fin_sent_tokenize():
#     try:
#         for w in tokenized:
#             words=nltk.word_tokenize(w)
#             tagged=nltk.pos_tag(words)
        
#             named_ent=nltk.ne_chunk(tagged)
#             named_ent.draw()
#     except Exception as e:
#         print(str(e))
#fin_sent_tokenize()


# In[ ]:


#new_word=wordnet.synset("genome")


# In[ ]:




