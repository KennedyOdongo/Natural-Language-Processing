#!/usr/bin/env python
# coding: utf-8

# ### Data

# In[1]:


#import modules.
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import os
import glob
import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import seaborn as sns
from nltk.corpus import stopwords
rand_state = 42
np.random.seed(rand_state)
import re

warnings.simplefilter('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# #### In the code block below, you supply a file path to the code and it will convert all the pdf files to text files. I found it much easier to work with text files rather than pdf files because of encoding issues.

# In[2]:


directory = r'C:\Users\Rodgers\Downloads\State of the State\Washington' #path
pdfFiles = glob.glob(os.path.join(directory, '*.pdf'))

resourceManager = PDFResourceManager()
returnString = StringIO()
codec = 'utf-8'
laParams = LAParams()
device = TextConverter(resourceManager, returnString, laparams=laParams)
interpreter = PDFPageInterpreter(resourceManager, device)


# ##### create a custom function to convert all the pdf's to text

# In[3]:


password = ""
maxPages = 0
caching = True
pageNums=set()

for one_pdf in pdfFiles:
    print("Processing file: " + str(one_pdf))
    fp = open(one_pdf, 'rb')
    for page in PDFPage.get_pages(fp, pageNums, maxpages=maxPages, password=password,caching=caching, check_extractable=True):
            interpreter.process_page(page)
    text = returnString.getvalue()
    filenameString = str(one_pdf) + ".txt"
    text_file = open(filenameString, "w",encoding="utf-8")
    text_file.write(text)
    text_file.close()
    fp.close()

device.close()
returnString.close()


# #### I copied all the tecxt files and put them in a folder "S_O_S Text". From this folder, I built a  dataframe: Like shown below.

# In[23]:


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


# #### Let's take a look at what the data frame looks like: Name, state,party, and the text of that year's state of the state.

# In[24]:


df.tail()


# #### Converted the dataframe into a CSV. It looks messed up but I think it should hold up well to any analysis.

# In[6]:


#df[["Name","State","party","year"]].to_csv(r'C:\Users\Rodgers\Downloads\governors.csv')


# #### Taking a look at how the text column looks like.

# In[25]:


df.text.tail()


# #### Grouping by party, approximately how many governors do we have.

# In[26]:


df.groupby('party').size()


# In[27]:


all_stopwords = stopwords.words('english')
all_stopwords.append('State')
all_stopwords.append("state")


# #### Tokenizing

# In[28]:


df["text_processed"]=df["text"].map(lambda x: re.sub('[,\.!?]', '', x))
df['text_processed'] = df['text_processed'].map(lambda x: x.lower())


# In[30]:


from nltk import sent_tokenize

text_tokens= [sent_tokenize(text) for text in df.text]
sentences=[word for word in text_tokens if not word in all_stopwords]
# remove the first and last sentences (meaningless intro/closing statements)
for i in range(len(sentences)):
    del sentences[i][0]
    #del sentences[i][-1]
    
    
sentence_lengths = [len(sent) for sent in sentences]
df['sentences'] = sentences
df['sentence_length'] = [len(sent) for sent in sentences]

# now need to "unstack" the above list of lists of sentences
sentences_all = []
for sentences in sentences:
    for sent in sentences:
        sentences_all.append(sent)


# #### By count of sentences in the speeches by party affiliation.

# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(10,6))
sns.lineplot(x='year',y='sentence_length',hue='party',data=df.reset_index())
plt.xlabel('')
plt.ylabel('Number of Sentences')
sns.despine()
plt.show()


# In[32]:


from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(list(df['text_processed'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()


# In[ ]:




