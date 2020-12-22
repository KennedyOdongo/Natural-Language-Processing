#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import modules.
import pandas as pd 
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from nltk import sent_tokenize
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import re
import warnings
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA


# #### read the text files and convert them into a data file

# In[2]:



path=r'C:\Users\Rodgers\Downloads\Text Analysis'
dirs = os.listdir(path)
df = pd.DataFrame(columns=['Name', 'State', 'party', 'year','text'])
for i in range(len(dirs)):
    components=dirs[i].split('_')
    Name=components[0]
    State=components[1]
    party=components[2]
    year=components[3].split('.')[0]
    
    df.loc[i,'year'] = year
    df.loc[i,'Name'] = Name
    df.loc[i,'party'] = party
    df.loc[i,'State'] = State 
    
    filename=os.path.join(path, dirs[i])
    text_file=open(filename, "r",encoding="mbcs")
    lines=text_file.read()
    lines=lines.replace('\n', ' ')
    df.loc[i, 'text'] = lines.lower()
df.year = df.year.astype(int) 
df.Name = df.Name.astype(str)
df.party = df.party.astype(str)
df.State = df.State.astype(str)
df.text = df.text.astype(str)
print('Shape: ', df.shape)    


# #### Let's take a look at what the data frame looks like: Name, state,party, and the text of that year's state of the state

# In[3]:


print(df.tail())
print(df.head())


# #### remove the punctuation and make all the letters lower case.

# In[4]:


df["text_processed"]=df["text"].map(lambda x: re.sub('[,\.!?]', '', x))
df['text_processed'] = df['text_processed'].map(lambda x: x.lower())


# In[5]:


all_stopwords = stopwords.words('english')
all_stopwords.append('State')
all_stopwords.append("state")


# In[6]:


text_tokens= [sent_tokenize(text) for text in df.text]
sentences=[word for word in text_tokens if not word in all_stopwords]
# remove the first and last sentences (meaningless intro/closing statements)
#for i in range(len(sentences)):
 #   del sentences[i][5]
  #  del sentences[i][-1]
    
    
sentence_lengths = [len(sent) for sent in sentences]
df['sentences'] = sentences
df['sentence_length'] = [len(sent) for sent in sentences]

# now need to "unstack" the above list of lists of sentences
sentences_all = []
for sentences in sentences:
    for sent in sentences:
        sentences_all.append(sent)


# In[7]:


plt.figure(figsize=(10,6))
sns.lineplot(x='year',y='sentence_length',hue='party',data=df.reset_index())
plt.xlabel('year')
plt.ylabel('Number of Sentences')
sns.despine()
plt.show()


# In[8]:


df.tail()


# In[10]:


df_democrats=df[df["party"]=="Democratic"]
df_republican=df[df["party"]=="Republican"]
df_democrats.to_csv(r'C:\Users\Rodgers\Downloads\Text Analysis\dems.csv')
df_republican.to_csv(r'C:\Users\Rodgers\Downloads\Text Analysis\reps.csv')
df.to_csv(r'C:\Users\Rodgers\Downloads\Text Analysis\all.csv')


# In[11]:


long_string = ','.join(list(df_democrats['text_processed'].values))
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[12]:


long_string1 = ','.join(list(df_republican['text_processed'].values))
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string1)
wordcloud.to_image()


# In[13]:


df.shape


# #### Lets plot the 10 most common words before doing a simple LDA model.

# In[14]:


def ten_most_common(count_data,count_vectorizer):
    import matplotlib.pyplot as plt
    words=count_vectorizer.get_feature_names()
    total_counts=np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    count_dict=(zip(words,total_counts))
    count_dict=sorted(count_dict,key=lambda x:x[1], reverse=True)[0:10]
    
    words=[w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
    
#initialize the count vectorizer with English stop words
count_vectorizer=CountVectorizer(stop_words='english')
#Fit and transform the text from the data you have created
count_data = count_vectorizer.fit_transform(df['text_processed'])
count_data1 = count_vectorizer.fit_transform(df_democrats['text_processed'])
count_data2 = count_vectorizer.fit_transform(df_republican['text_processed'])


# ### Republican

# In[15]:


ten_most_common(count_data2,count_vectorizer)


# ### LDA

# In[17]:


warnings.simplefilter("ignore", DeprecationWarning)
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below
number_topics = 10
number_words = 10
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data2)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)


# In[12]:


dems_common=['state','year','work','new','people','make','jobs']
reps_common=['state','year','south','people','governor','years','new','just','work','education']
for word in dems_common:
    if word in reps_common:
        print(word)


# In[13]:


with open('dems_topics.txt', 'r') as f:
    dems= [line.strip() for line in f]
len(dems)


# In[24]:


vocab_d=[]
for line in dems :
    word=line.split() #this is a list of lists containing the all the words in the text
    vocab_d.append(word)
#create a list that contains only the words
vocabs_d=[]
for i in vocab_d:
    for j in i:
        vocabs_d.append(j)


# #### All the words in the democratic list of topics

# In[25]:


vocabs_d


# In[26]:


for word in dems_common:
    if word in vocabs_d:
        print(word)


# In[27]:


with open('reps_text.txt', 'r') as f:
    reps= [line.strip() for line in f]
len(reps)


# In[28]:


vocab_r=[]
for line in dems :
    word=line.split() #this is a list of lists containing the all the words in the text
    vocab_r.append(word)
#create a list that contains only the words
vocabs_r=[]
for i in vocab_r:
    for j in i:
        vocabs_r.append(j)


# In[30]:


for word in reps_common:
    if word in vocabs_r:
        print(word)


# In[ ]:




