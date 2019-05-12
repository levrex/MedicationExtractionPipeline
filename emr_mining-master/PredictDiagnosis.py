
# coding: utf-8

# # Predict Diagnosis notebook

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, model_selection, tree, preprocessing, metrics
import sklearn.ensemble as ske


# In[11]:


radiag_df = pd.read_csv(r'output_files/DF_studentOpdracht.csv', index_col=0, sep="|", na_values=['NA'])
radiag_df['Outcome'] = radiag_df['Outcome'].apply(lambda x : binarize(x)) 
radiag_df.head()


# ### Data exploration

# In[3]:


radiag_df[['Outcome']]


# ### Pipeline

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import tree

naive_bayes = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MultinomialNB()),
            ])

text_clf = naive_bayes.fit(X_train, y_train)

pred = text_clf.predict(X_test)

