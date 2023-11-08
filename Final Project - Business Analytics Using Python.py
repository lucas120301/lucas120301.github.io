#!/usr/bin/env python
# coding: utf-8

# # Final project - Group 4

# ## Predicting a "viral" article on Medium
# We are a brand new startup providing SEO consulting services for digital native vertical brands. So far, we have focused on mainstream social media such as Twitter, Instagram, Facebook... But we have realized that we were setting an important commmunication channel aside: Medium. Medium is the trendy website for content creators and people with disruptive ideas, wishing to share their thoughts and experiences to build a strong community. Inbound marketing is key nowadays, and we want to predict the popularity of a given Medium article using well-chosen features to better advice our clients. 

# ## Logistic regression
# We have chosen the most efficient/explainable regression method; logistic regression is rather adapted to classification algorithms with a large amount of categorical data. We will try to predict whether the article is viral (1) or not (0) setting a specific threshold of "claps". We will need to dive into our data to set a proper threshold. 

# ## 1 - The method

# We uploaded a dataset of random articles on Medium, willing to better understand what makes an article viral. First of all, we will randomly split the dataset in two ("train" and "test" datasets), to structure our "train" dataset. 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


medium_csv = r"C:\Users\zargo\OneDrive\Documents\HEC M1\Business analytics using python\medium_data.csv\medium_data.csv"
df = pd.read_csv(medium_csv, header=0, sep=',', parse_dates=['date']) 
df.info()


# ## 1.1. Cleaning data and feature elaboration
# Our feature of interest (which we want to predict) will be the number of claps. Let's go through our dataset to know what each column title stands for. 
# - id is the number of the article
# - url is the url of the article on the Medium website
# - image is the volume of the image uploaded in the article. If there aren't any image, the cell will be empty. 
# - responses is the number of "comments" on the article.
# - publication is the topic of the article. We will generate a list of all the topics later in this notebook. 
# The rest of the columns titles speak for themselves... 
# 
# The url feature is not relevant for our case. We will just drop it. Then we will change the types of certain columns to types that are more suited for our study. Eventually, we will explore which numerical features we could integrate to our model based on the categorical features we initially have. 

# In[2]:


df.drop(['url'],axis=1,inplace=True)
df


# ### 1.1.1 - Setting proper data types for each column. 
# - The 'response' column data type shouldn't be an 'object' but an integer. 
# Changing a column date type is impossible when the column contains NaN values. Fortunately, it is not the case for our column here. 

# In[3]:


#Identifying the rows where 'responses' is not a number. 
df.sort_values("responses", inplace = True)
#The last two rows displayed contain 'Read' instead of an actual number in the 'responses' column. Let's remove them. 
df.drop([6392,3977],axis=0,inplace=True)
df.astype({'responses': 'int64'}).dtypes


# ### 1.1.2 - Is the presence of an image in the article a good feature? 

# In[4]:


without_image = str(round(100 * df['image'].isna().sum()/6508))
print("Articles without an image represent " + "\033[1m" + without_image + "%" + "\033[0m" + " of our sample. ")


# In[5]:


df.drop(['image'],axis=1,inplace=True)
df


# Since the number of articles without an image **is below 5%** of total articles, this feature is **irrelevant**. 

# ### 1.1.3 - Popularity based on topic

# In[6]:


topics = list(df['publication'].unique())
topics


# In[7]:


df = pd.get_dummies(df, columns =['publication'])


# ### 1.1.4 - Textual analysis

# In this part we will analyze what we consider to be the most important feature of our model: the title. We will take time to reflect on what makes a good title for a Medium article, in order to highlight eventual relevant features for our model. 
# 1. removing stopwords and lemmatizing the titles
# 2. highlighting titles which contain 'wh-words' and interrogative titles - they are likely to be well referenced on Google
# 3. highlighting titles which directly address the reader (using second person or first person plural) - they are likely to be more appealing to a reader as they convey a sense of community
# 4. highlighting titles which use first person - they emphasize on story-telling
# 5. highlighting titles which contain digits - the old Buzzfeed clickbait method of listing stuff

# In[8]:


wh_words = ['what', 'how', 'where', 'who', 'which', 'when']
you_us_words = ['yours','your', 'you', "you're",'we',"we're","us","ours", "our"]
me_words = ['I','me','my','mine']



def digit(title):
    list1=[]
    for i in title.lower().split():
        list1.append(i.isdigit())
    return int(list1.count(True)>0)

def wh(title):
    list1=[i for i in title.lower().split() if i in wh_words]
    return int(list1 != [])

def you_us(title):
    list1=[i for i in title.lower().split() if i in you_us_words]
    return int(list1 != [])

def me(title):
    list1=[i for i in title.lower().split() if i in me_words]
    return int(list1 != [])

def q_mark(title) :
    list1=[char for char in title if char=="?"]
    return int(list1 != [])

df['digits']= df['title'].apply(digit)
df['wh_words']= df['title'].apply(wh)
df['you_us_words']= df['title'].apply(you_us)
df['me_words']= df['title'].apply(me)
df['question_mark']=df['title'].apply(q_mark)

# What we could also do is determine whether the article has a subtitle or not. 
def is_subtitle(subtitle):
    return int(pd.isna(subtitle))

df['is_subtitle']=df['subtitle'].apply(is_subtitle)
print(list(df.columns))


# ## 1.1.5 - What about the date?
# We could imagine that few people read work-related articles in the summer? Maybe that is too far-fetched. Let's discard this column later.
# 

# ## 1.1.6 What is viral, after all?
# We decided to set the threshold of claps at the 3rd quartile, above which an article is considered as 'viral'. In order to be absolutely rigourous, we will set this threshold after train/test splitting our dataset, the threshold being based on the train dataset.
# 

# ## 1.2.1 - Let's select the features we will integrate to our model!

# In[9]:


columns_X = [ 'responses', 'reading_time', 'publication_Better Humans', 'publication_Better Marketing', 'publication_Data Driven Investor', 'publication_The Startup', 'publication_The Writing Cooperative', 'publication_Towards Data Science', 'publication_UX Collective', 'digits', 'wh_words', 'you_us_words', 'me_words', 'question_mark', 'is_subtitle']
X = df[columns_X]
print(X.head())

Y = df['claps']
print(Y.head())


# ## 1.2.2 - Of the importance of normalization
# Our X dataframe contains mostly one-hot encoded vectors... But the 'responses' and 'reading_time' column contain continuous - numerical - values. To train our model properly, we will have to scale our numerical data. 

# In[10]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[['responses','reading_time']]=scaler.fit_transform(X[['responses','reading_time']])
print(X)


# ## 1.2.3 - Split the dataset!
# Everything is in order now, so we will randomly split the data into a training set and a testing set

# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# In[12]:


#Now that we have our X_train, we can set the threshold about which an article is considered to be viral. 
threshold = round(Y_train.quantile(0.75))
print(threshold)
def thresh(clap):
    return int(clap>threshold)


Y_train = Y_train.apply(thresh)
Y_test = Y_test.apply(thresh)
print(Y_train)


# We will now train our data
classifier = LogisticRegression()

classifier.fit(X_train, Y_train)

# testing
predicted_y = classifier.predict(X_test)

for yi in range(len(predicted_y)):
    if (predicted_y[yi] == 1):
        print(yi, end="\t")

# assess the classifier, accuracy
print('Accuracy: {:.4f}'.format(classifier.score(X_test, Y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, predicted_y)
print(confusion_matrix)


from sklearn.metrics import classification_report
print(classification_report(Y_test, predicted_y))
# precision, recall, f1-score, support


# # 2 - Interpretation and observations

# ## 2.1 - Confusion matrix analysis
# 
# We can draw a few conclusions from the latter metrics: 
# - Our accuracy (0.87) is satisfying. The features selected for this model were on point!
# - Since we are a company providing consulting services for brands wishing to expand their SEO strategy, there are only two elements of the confusion matrix which matter to us: False positive, and false negative. Indeed, out of 100 articles which were predicted to be viral, 86 were actually viral. Reversely, 97% data points that were actually negative were labeled as negative. We managed to prove that our best practices are efficient, and that not following these best practices can be detrimental for your SEO strategy... So we succeded in our task!
# - Out of 100 viral articles, our model could only predict the sucess of 53 of them. Business-wise, this isn't an issue since our activity is centered around *making* articles viral.
# - Standardization of numerical values didn't change much in terms of performance of the model, but it enabled the algorithm to run faster. 

# ## 2.2 - Descriptive statistics and further feature elaboration
# We did not want to be biased in our choice of feature (our model would have been over-fitted), so we did not measure the proportion of articles without an image, or without subtitles, but we can now determine all these parameters: 
# - A threshold for claps: it will be the third quartile of our whole sample. 
# - The number of articles without subtitles or images
# - The median reading time 

# In[13]:


data_descriptive = pd.read_csv(medium_csv, header=0, sep=',', parse_dates=['date']) 


# In[14]:


threshold = round(df['claps'].quantile(0.75))
threshold


# In[15]:


without_subtitle = str(round(100 * data_descriptive['subtitle'].isna().sum()/6508))
print("The number of articles without subtitle represents " + "\033[1m" + without_subtitle + "%" + "\033[0m" + " of our sample. ")


# In[16]:


without_image = str(round(100 * data_descriptive['image'].isna().sum()/6508))
print("The number of articles without an image represents " + "\033[1m" + without_image + "%" + "\033[0m" + " of our sample. \n This feature was indeed irrelevant to our study.")


# In[17]:


median_reading_time = int(data_descriptive['reading_time'].median())
print("The median reading time is " + "\033[1m" + str(median_reading_time) + "min" + "\033[0m" + " in our sample. ")


# # 3 - Bonus : Linear Regression?
# Are we able to predict the exact number of "claps" an article can have, based on all our previous features? We tried to run a linear regression model, but it wasn't very performant. *To run this model, do not run all cells (otherwise you will not use the right Y_train nor Y_test). Instead, run every cell but the one starting with #Now that we have our X_train, we can set the threshold...)*

# In[18]:


#build and train a linear regression model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, Y_train)
plt.scatter(lr.predict(X_train), Y_train)
plt.xlabel('y_pred')
plt.ylabel('y_true')
plt.show()


# In[ ]:




