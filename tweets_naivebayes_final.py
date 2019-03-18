#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
#import library numpy dan pandas


# In[2]:


df=pd.read_csv('tweets_csv_siap.csv',sep=';')
#read csv file


# In[3]:


df.head()
#verifikasi 5 row data pertama saja


# In[4]:


df.tail()
#verifikasi 5 row data terakhir saja


# In[5]:


df.columns
#verifikasi nama kolom


# In[6]:


df.isnull().sum()
#verifikasi kelengkapan data


# In[7]:


len(df)
#verdifikasi panjang row data


# In[8]:


df['kelas'].value_counts()
#melihat unik value dari kolom label nya


# In[9]:


import sklearn


# In[10]:


from sklearn.model_selection import train_test_split
#library untuk split data test dan data training


# In[11]:


X = df['tweet']


# In[12]:


y = df['kelas']
#label 'keluhan', 'respon' atau 'bukan keluhan atau respon'


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state=42)
#dipakai 30% untuk test size, 70% nya training size


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[15]:


from sklearn.naive_bayes import MultinomialNB
#dipakai algoritma naive bayes


# In[16]:


from sklearn.pipeline import Pipeline
#Pipeline kelas perform kombinasi vectorizer dan tfidf dalam satu syntax


# In[17]:


text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',MultinomialNB())])
#semua step dalam satu single step


# In[18]:


text_clf.fit(X_train,y_train)


# In[19]:


predictions = text_clf.predict(X_test)


# In[20]:


from sklearn.metrics import confusion_matrix,classification_report


# In[21]:


print(confusion_matrix(y_test,predictions))


# In[22]:


print(classification_report(y_test,predictions))
#statistik hasil test dari training


# In[23]:


from sklearn import metrics


# In[24]:


metrics.accuracy_score(y_test,predictions)
#tingkat akurasi nya


# In[25]:


text_clf.predict(["Buka twitter muncul promote ginian. Gimana atuh kalo anak kecil yg liat? Cc: @Menkominfo @ridwankamil https://t.co/3okfeZIsap"])
#masukkan tweet uji di sini - belum pakai GUI


# In[ ]:




