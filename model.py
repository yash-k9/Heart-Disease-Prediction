#!/usr/bin/env python
# coding: utf-8

# //These are the following attributes present in the dataset
# //dataset contains the information of the previous patient's health records.
# 
# age: The person's age in years
# 
# sex: The person's sex (1 = male, 0 = female)
# 
# cp: The chest pain experienced (Value 0: typical angina, Value 1: atypical angina, Value 2: non-anginal pain, Value 3: asymptomatic)
# 
# trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
# 
# chol: The person's cholesterol measurement in mg/dl
# 
# fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# 
# restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
# 
# thalach: The person's maximum heart rate achieved
# 
# exang: Exercise induced angina (1 = yes; 0 = no)
# 
# oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
# 
# slope: the slope of the peak exercise ST segment (Value 0: upsloping, Value 1: flat, Value 2: downsloping)
# 
# ca: The number of major vessels (0-3)
# 
# thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
# 
# target: Heart disease (0 = no, 1 = yes)
#  

# In[1]:


import pandas as pd
import seaborn as sns


# first we read the data from the csv file. Pandas is the library for performing operations on the dataset.
# describe and head are used to see the information.

# In[ ]:


df = pd.read_csv('heart.csv')
df.describe()


# In[3]:


df.head()


# In[4]:


df.rename(columns = {'sex' : 'gender', 'cp' : 'chestPain', 'trestbps' : 'restBp', 'fbs' : 'bloodSugar', 'restecg' : 'restEcg', 'thalach' : 'maxHeart'}, inplace = True)


# In[5]:


df.describe()


# The command drop is used to delete the column from the table. In the step pre-processing we remove the columns that are not needed.

# In[6]:


df = df.drop(['ca'], axis = 1)

After Changing the data and values. These have to one hot encoded and pickled.
# In[7]:


df.rename(columns = {'oldpeak' : 'oldPeak', 'chol' : 'cholestrol'}, inplace = True)
list(df.columns)


# One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction. The categorical value represents the numerical value of the entry in the dataset. We have performed one hot encoding. 
# 
# EX: In the data set, male is represented as 1 and female as 0
#     when we try to make a model that predict the chances because of this the model will be biased.
#     so we make separate columns for men and women
#     
#     Gender  after one hot encoding  Male Female
#     1                                1     0
#     0                                0     1  

# In[8]:


ohe_col = ['chestPain', 'restEcg', 'slope', 'thal', 'gender']
df = pd.get_dummies(df, columns = ohe_col)
print(df.columns)


# In[9]:


df.head(10)


# In[10]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# Target in the data represents whether there the heart disease occured or not.
# BY using this as a final result we train and test data

# In[11]:


X = df.drop(['target'], axis = 1)
y = df['target']


# we have split data into two 80 for training 20 for testing
# Random Forest Regessor is a algorithm for prediction. 
# 
# A random forest is an ensemble model that consists of many decision trees. Predictions are made by averaging the predictions of each decision tree. Or, to extend the analogy—much like a forest is a collection of trees, the random forest model is also a collection of decision tree models. This makes random forests a strong modeling technique that’s much more powerful than a single decision tree.
# 
# 
# Each tree in a random forest is trained on the subset of data provided. The subset is obtained both with respect to rows and columns. This means each random forest tree is trained on a random data point sample, while at each decision node, a random set of features is considered for splitting.

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[13]:


ml = RandomForestRegressor(n_estimators = 190, random_state = 0)
ml.fit(X_train, y_train)
y_pred = ml.predict(X_test)


# In[14]:


y_pred = y_pred > 0.53


# In[15]:


y_pred


# In[16]:


newdf=pd.DataFrame(y_pred, columns=['Status'])
newdf


# In[17]:


cols = list(X.columns.values)


# In[18]:


cols


# In[19]:


filename = 'allcol.pkl'
joblib.dump(cols, filename)


# In[20]:


filename = 'finalmodel.pkl'
joblib.dump(ml, filename)


# In[24]:


cm = confusion_matrix(y_test, y_pred.round())
sns.heatmap(cm, annot = True, cmap = 'Blues', cbar = False)


# In[25]:


acc = accuracy_score(y_test, y_pred.round())
print(acc)


# In[ ]:




