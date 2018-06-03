
# coding: utf-8

# In[2]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Scipy: {}'.format(scipy.__version__))
print('Sklearn: {}'.format(sklearn.__version__))


# In[5]:


#import the necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


#load the dataset from the csv file using pandas
data = pd.read_csv('creditcard.csv')


# In[9]:


#explore the dataset
print(data.columns)


# In[10]:


print (data.shape)


# In[11]:


print (data.describe())


# In[12]:


data = data.sample (frac = 0.1, random_state = 1)

print (data.shape)


# In[13]:


#plot histogram with each parameter

data.hist (figsize = (20, 20))
plt.show()


# In[14]:


#determine fraud cases

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/ float(len(Valid))

print(outlier_fraction)

print('Fraud Cases: {}'.format(len(Fraud))) 
print('Valid Cases: {}'.format(len(Valid))) 


# In[15]:


#build correlation matrix
cormat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(cormat, vmax = .8, square = True)


# In[16]:


#get all the columns from the dataframe

columns = data.columns.tolist()

#filter the columns to remove data we donot want

columns = [c for c in columns if c not in ["Class"]]

#store the variable we will be predicting on

target = "Class"

x = data[columns]
y = data[target]

#print the shapes of x and y

print(x.shape)
print(y.shape)


# In[24]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


#define a random state
state = 1

#define a outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(x),
                                       contamination = outlier_fraction,
                                       random_state = state),
    "Local Outlier Factor": LocalOutlierFactor(
    n_neighbors = 20,
    contamination = outlier_fraction)
}


# In[29]:


# Fit the model
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    #fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(x)
        scores_pred = clf.negative_outlier_factor_
    else: 
            clf.fit(x)
            scores_pred = clf.decision_function(x)
            y_pred = clf.predict(x)
            
            #Reshape the prediction values to 0 for valid, 1 for fraud
            y_pred[y_pred == 1] = 0
            y_pred[y_pred == -1] = 1
            
            n_errors = (y_pred != y).sum()
            
            #Run classificaiton metrics 
            print ('{}: {}'. format(clf_name, n_errors))
            print(accuracy_score(y, y_pred))
            print(classification_report(y, y_pred))

