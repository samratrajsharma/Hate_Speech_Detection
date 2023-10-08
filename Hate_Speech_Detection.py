#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dataset = pd.read_csv("labeled_data.csv")


# In[3]:


dataset


# In[4]:


dataset.isnull()


# In[5]:


dataset.isnull().sum()


# In[6]:


dataset.describe()


# In[7]:


dataset["labels"] = dataset["class"].map({0: "Hate Speech",
                                         1: "Offensive Language",
                                         2: "No hate or offensive language"})


# In[8]:


dataset


# In[9]:


data = dataset[["tweet", "labels"]]


# In[10]:


data


# In[11]:


import re
import nltk
nltk.download("stopwords")
import string


# In[12]:


from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))


# In[13]:


stemmer = nltk.SnowballStemmer("english")


# In[14]:


def clean_data(text):
    text = str(text).lower()
    text = re.sub('https?://\S+|www\.S+','',text)
    text = re.sub('\[,*?\]', '',text)
    text = re.sub('<,"?>+','',text)
    text = re.sub('[%s]' %re.escape(string.punctuation),'', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*','',text)
    text = [word for word in text.split('  ') if word not in stopwords]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text


# In[15]:


data["tweet"] = data['tweet'].apply(clean_data)


# In[16]:


data


# In[17]:


X = np.array(data["tweet"])
Y = np.array(data["labels"])


# In[18]:


X


# In[19]:


Y


# In[20]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# In[21]:


cv = CountVectorizer()
X = cv.fit_transform(X)


# In[22]:


X


# In[23]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[24]:


X_train


# In[25]:


from sklearn.tree import DecisionTreeClassifier


# In[26]:


dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)


# In[27]:


Y_pred = dt.predict(X_test)


# In[28]:


Y_pred


# In[29]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# In[30]:


cm


# In[31]:


import seaborn as sns
import matplotlib.pyplot as ply
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


sns.heatmap(cm, annot = True, fmt =".1f", cmap="YlGnBu")


# In[33]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_pred)


# In[34]:


sample = "Let's unite and make this world a better place"
sample = clean_data(sample)


# In[35]:


sample


# In[36]:


data1 = cv.transform([sample]).toarray()


# In[37]:


data1


# In[38]:


dt.predict(data1)


# In[ ]:




