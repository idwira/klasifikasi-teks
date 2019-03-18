#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd
#import library numpy dan pandas


# In[3]:


df=pd.read_csv('tweets_csv_siap.csv',sep=';')
#read csv file


# In[4]:


df.head()
#verifikasi 5 row data pertama saja


# In[5]:


df.tail()
#verifikasi 5 row data terakhir saja


# In[6]:


df.columns
#verifikasi nama kolom


# In[7]:


df.isnull().sum()
#verifikasi kelengkapan data


# In[8]:


len(df)
#verdifikasi panjang row data


# In[9]:


df['kelas'].value_counts()
#melihat unik value dari kolom label nya


# In[10]:


import sklearn


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X = df['tweet']
#isi tweets


# In[13]:


y = df['kelas']
#label 'keluhan', 'respon' dan 'bukan keluhan atau respon'


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state=42)
#test data 30%, traing data 70%


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[16]:


from sklearn.svm import LinearSVC
#Menggunakan algoritma SVM


# In[17]:


from sklearn.pipeline import Pipeline
#Pipeline kelas perform kombinasi semua nya tfidf vectorizer dan model yg dipakai


# In[18]:


text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])
#semua step dalam satu single step


# In[19]:


text_clf.fit(X_train,y_train)


# In[20]:


predictions = text_clf.predict(X_test)


# In[21]:


from sklearn.metrics import confusion_matrix,classification_report


# In[22]:


print(confusion_matrix(y_test,predictions))


# In[23]:


print(classification_report(y_test,predictions))
#statistik hasil test dari training data


# In[24]:


from sklearn import metrics


# In[25]:


metrics.accuracy_score(y_test,predictions)
#akurasi untuk 30% data test


# In[26]:


text_clf.predict(["RT @infobdg: #suaraBDG via @aldlansyah: Tombol sama lampu penyebrangannya engga berfungsi nih, pak cc: @dishub_kotabdg https://t.co/NvtS3Eq…"])
#paste syntax disini untuk di uji - belum pakai GUI


# In[ ]:




