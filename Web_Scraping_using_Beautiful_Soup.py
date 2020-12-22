#!/usr/bin/env python
# coding: utf-8

# # Webscraping with BeautifulSoup& requests or requests_html

# In[1]:


#import modules
from bs4 import BeautifulSoup
import requests
#You can use the above two modules together or you can use the one below as a stand alone.
from requests_html import HTML, HTMLSession # this is for a HTML  file that is already on you local computer, for a website we need a
#HTMLSession
import csv
#requests-HTML does not have a prettify functionality, but you can use an online HTML prettifer
import urllib.request
import io #The io module provides Python’s main facilities for dealing with various types of I/O, the IO mean input output


# In[2]:


#HTML
#HTML is structured in a way that all the information is contained in a certai tags.
#When referring to HTML, the <div> tag is used to divide or section off content on a web page.
#An anchor tag is an HTML tag. It is used to define the beginning and end of a hypertext link.
#The href attribute specifies the link's destination
#paragraph tags are the text summary of the page you are looking at
#the li tag is for an element contained in a list.
#td tag deines a standard HTML cell in a td table
#tr is the table row tag
#classes are used for CSS styliing or in Javascript to identify specific elements
#what's in the body tag is what gets displayed on the website


# In[3]:


get_ipython().run_line_magic('autosave', '120 #this just tells jupyter to save the notebook every 120 seconds, nothing too fancy')


# In[4]:


# starting from a simple HTML file in the same directory, open up the file and pass it into beautifulsoup
with open('index.html') as html_file:
    soup=BeautifulSoup(html_file,'lxml') #pass the file into beautiful soup, specify the parser you want to use
print(soup)
print()
#because the HTML file is not formatted i.e. it is pushed over to the left, we can use the prettify function to make it 
#look more like a HTML file. This helps to see which tags are nested within each other
print(soup.prettify()) # this is a method: A function in a python class


# In[5]:


# the easiest way to get information from a tag is to access it like an attribute, this gets only the first item on the page
title=soup.title# I like naming these variables the same as tag to it's easir to track but you can name them whatever you 
#choose
print(title) #this prints out the title plus the tags, this will also print out all the child tags of the tag you choose,see
#below
head=soup.head
print(head)
print(title.text) #this prints only the title
#accessing the text in the paragraph
parag=soup.p
print(parag.text)
#We can use the find method to find the exact information that we want, this is only possible with bs4 and requests,not
#requests_html
#more of this with a real website.
#tag=soup.find('tag',class_="whatever class you want") -> for one tag
#instead of using the find for one tag we can use the find_all method


# In[7]:


# #THIS IS FOR SAMPLE HTML FILE THAT IS IN THE SAME DIRECTORY AS YOUR SCRIPT
# with open(' simple.html') as html_file:
#     source=html_file.read()
#     html=HTML(html=source) # instance of the html attribute.
     
# accesing the html obejct by prinitng out the html attribute
# print(html.html) #prints the html attribute of the instance we have created above
# if we only wanted the text without the tags, we use the text attribute 
# print(html.html.text) #instance.attribute.text
# # to find something within a html file we use the find method as shown below
# match=html.find('title') #we want to find the title
# #to get the HTML of that title tag we can use
# print(match[0].html) #accesing the first element and printing out its html, if we wanted the text we'd use print(match[0].text)
# # match=html.find('title',first=True) prints the same thing as the .text method above.
# # if you want a div with the id of footer "x=html.find('#footer', first=True)"

# # grab the first div, its article and print out its text
# article=html.find('div.article', first=True)
# print(article.text)
# articles=html.find('div.article') #this will find all articles in the div class, then we can loop over to collect all the data.
# for article in articles:


# In[8]:


#use the requests module, to convert your webpage to an object similar to an HTML file.
source=requests.get('http://safaricom.co.ke').text #returns a response object, to get the source code from that object we add
#.text to it. The source variable should be equal to the HTML of the website you are scraping
soup1=BeautifulSoup(source,'lxml') # converting to a BeautifulSoup object
print(soup1.prettify()) #prettify so that it looks like a HTML file


# In[9]:


#start by getting one of the things you are scraping: could be a headline, summary etc .
div=soup1.find('div', class_='mobile-sec--nav')
print(div.prettify())


# In[10]:


#so say we want go get only the word voice from this HTML we can access it like an attribute:
V=div.li.a.text
print(V)


# In[11]:


#open csv file before the for loop:
#csv_file=open('safaricom_scrape.csv', 'w')
#csv_writer=csv.writer(csv_file)
#csv_writer.writerow(['services'])


# In[12]:


#open csv file before the for loop:
csv_file=open('safaricom_scrape.csv', 'w')
csv_writer=csv.writer(csv_file)
csv_writer.writerow(['services'])
# get all the texts form you div tag...
for item in div.find_all('li'):
    voice=item.a.text
    print(voice)
    print()
    csv_writer.writerow([voice])
csv_file.close()
    


# Scraping wikipedia for the list of corporate scandals

# In[13]:


#start with the URL
url='https://en.wikipedia.org/wiki/List_of_corporate_collapses_and_scandals'
req=urllib.request.urlopen(url)
article = req.read().decode()
with open('List_of_corporate_collapses_and_scandals.html','w',encoding="utf-8") as scandals:
    scandals.write(article)
print(scandals) #doesn't show anything because it's all been coverted to bytes


# In[14]:


# Load article, turn into Beautifulsoup and get the table tags, amake sure you specify the encoding otherwise it will throw 
#errors
article = open('List_of_corporate_collapses_and_scandals.html',encoding="utf8").read()
soup2 = BeautifulSoup(article, 'html.parser')
tables = soup2.find_all('table', class_='sortable')


# In[15]:


for table in tables:
    ths = table.find_all('th')
    headings = [th.text.strip() for th in ths]
    if headings[:5] == ['Name', 'HQ', 'Date', 'Business', 'Causes']:
        break


# In[16]:


with open('List_scandals.csv', 'w',encoding="utf8") as scandals:
    for tr in table.find_all('tr'):
        tds = tr.find_all('td')
        if not tds:
            continue
        Name, HQ, Date, Business, Causes= [td.text.strip() for td in tds[:5]]
        # Wikipedia does something funny with country names containing
        # accented characters: extract the correct string form.
        if '!' in HQ:
            HQ= HQ[HQ.index('!')+1:]
        print('; '.join([Name, HQ, Date, Business, Causes]), file=scandals)
#print a txt file to your dircetory and then you can open that in excel      


# # Everything below uses requests_HTML, same idea but you use just one module

# In[17]:


#scrapping data from a realm website using HTMLSessions for the requests-HTML website
#make an instance of the HTMLSessions class
session=HTMLSession()
r=session.get('http://quotes.toscrape.com/') #uses the library to get a response from a website, response is now a html object
print(r.html) #prints the HTML attribute


# In[18]:


#start by grabbing one entity's information first then we will loop through all of them to find the rest
#find the tag for one post that you are looking for
#name your variables after the tags that contain a whole entity of what you are looking for
div=r.html.find('.quote',first=True)
print(div.html) #gives you all of the HTML of that first div tage for one entity
# div_=r.html.find('.quote',first=True).text
# print(div_)


# In[19]:


#within that div class we want to find the quote
quote=div.find('.text', first=True).text
print(quote)


# In[20]:


#find the author
author=div.find('.author', first=True).text
print(author)


# In[21]:


#find the tags
tags=div.find('.tag', first=True).text
print(tags)


# In[22]:


# #loop over all of the articles and get the information for all of them....
# # Go back to the top where we found our first div tag and replace
# divs=r.html.find_all('div',class_='quote') # the class option is only available in beautiful soup
# for div in divs:
#     quote=div.find('.text', first=True).text
#     print(quote)
#     author=div.find('.author', first=True).text
#     print(author)
#     tags=div.find('.tags', first=True).text
#     print(tags)


# In[23]:


#the creation of the csv file should come before the for loop, looping through all the various parts of the html file
csv_file=open('bks_scrape.csv','w') #open is a context manager
csv_writer=csv.writer(csv_file)
#writing headers of the csv file, and pass in a list of values that we want to write to this row
csv_writer.writerow(['quotes', 'author', 'tags'])


# In[24]:


#scraper 
divs=r.html.find('.quote')
for div in divs:
    quote=div.find('.text', first=True).text
    print(quote)
    author=div.find('.author', first=True).text
    print(author)
    tags=div.find('.tag', first=True).text
    print(tags)
    print()
    
    #at the very bottom of our for loop we can write that data to the global csv file that we just created
    csv_writer.writerow([quote, author,tags]) #the values that we pass in are a list of the variables that we created above
#outside of our for loop we need to close the csv file, because we have not used a context manager
csv_file.close()

