#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#loading dataset
df = pd.read_csv(r'D:\Purity\ML_Zoomcamp\endterm\survey.csv')
print(df.shape)
df.head()


# Data cleaning

# In[3]:


#setting columns to lower case
df.columns = df.columns.str.lower() 
df.head()


# In[4]:


#checking variable types
df.dtypes


# In[5]:


#getting the number of uniques for each column
df.nunique()


# In[6]:


#checking for missing values
df.isnull().sum()


# In[7]:


# plotting missing values 
ax = df.isna().sum().sort_values().plot(kind = 'barh', figsize = (9, 10))
plt.title('Percentage of Missing Values Per Column in Train Set', fontdict={'size':15})
for p in ax.patches:
    percentage ='{:,.0f}%'.format((p.get_width()/df.shape[0])*100)
    width, height =p.get_width(),p.get_height()
    x=p.get_x()+width+0.02
    y=p.get_y()+height/2
    ax.annotate(percentage,(x,y))


# In[8]:


#dropping state and comments due to the large numbers of missing values
df.drop(['comments','state'], axis=1, inplace=True)
df.head()


# In[9]:


#filling up the missing values for the columns 'work_interfere' and self_employed
df['work_interfere'] = df['work_interfere'].fillna("Unknown") #filling with the unknown class for column work_interfere
df = df.apply(lambda x: x.fillna(x.value_counts().index[0])) #filling with most common value for column self_employed
df.isnull().sum()


# In[10]:


df.shape


# In[11]:


#filtering out ages less than 16 and greater than 100
df = df[df.age>16]
df = df[df.age<100]
df.shape


# Exploratory Data Analytics (EDA)

# Descriptive Analysis

# In[12]:


df.describe()


# In[13]:


#histogram of the age column
sns.histplot(df.age, bins=50)


# In[14]:


#getting log of age and plotting the histogram
df['log_age'] = np.log(df['age'])
sns.histplot(df.log_age, bins=50)


# Correlational Analysis

# In[15]:


#plotting numerical columns against the target variable
def boxplot(df, cols):
    for col in cols:
        sns.set_style('whitegrid')
        sns.boxplot(x='treatment', y=col, data=df)
        plt.title('Boxplot of ' + col)
        plt.ylabel(col) #setting text for y axis
        plt.show()
boxplot(df, ['age','log_age'])


# In[16]:


#creating a list for categorical variables
cat_cols = ['country', 'self_employed', 'family_history', 'work_interfere', 'no_employees', 
            'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program', 
            'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence',
             'coworkers', 'supervisor', 'mental_health_interview','phys_health_interview', 
             'mental_vs_physical', 'obs_consequence']


# In[17]:


#visualizing class separation by categorical variables
df['dummy'] = np.ones(shape = df.shape[0])
for col in cat_cols:
    print(' ')
    print(col)
    print(' ')
    counts = df[['dummy', 'treatment', col]].groupby(['treatment', col], as_index = False).count()
    _ = plt.figure(figsize = (10,4))
    plt.subplot(1, 2, 1)
    temp1 = counts[counts['treatment'] == 'No'][[col, 'dummy']]
    plt.bar(temp1[col], temp1.dummy)
    plt.xticks(rotation=90)
    plt.title('Counts for ' + col + '\n not seeking treatment')
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    temp2 = counts[counts['treatment'] == 'Yes'][[col, 'dummy']]
    plt.bar(temp2[col], temp2.dummy)
    plt.xticks(rotation=90)
    plt.title('Counts for ' + col + '\nseeking treatment')
    plt.ylabel('count')
    plt.show()


# In[18]:


#mutual information scores for the categorical variables
from sklearn.metrics import mutual_info_score
#creating a function
def mutual_score(series):
    return mutual_info_score(series, df.treatment)


# In[19]:


mi = df[cat_cols].apply(mutual_score) #getting mutual info scores
pd.set_option('display.max_rows', None) #setting to print all rows
print(mi.sort_values(ascending=False)) #sorting in descending order


# In[20]:


#Categorical columns to be used for training the model
cat_cols = ['family_history', 'work_interfere','benefits', 'care_options']


# Splitting the dataset

# In[21]:


#importing train_test_split library
from sklearn.model_selection import train_test_split


# In[22]:


#split to get test set
df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
#forming the label
y_train = df_train.treatment.values
y_test = df_test.treatment.values


# In[23]:


df_train.head()


# In[24]:


print(df_train.shape)


# In[25]:


print(df_test.shape)


# Training the selected model

# In[26]:


#importing libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


# In[27]:


df.treatment = (df.treatment == 'Yes').astype(int)#turning target variable to int


# In[28]:


#function for training random forest
def RF_train(X,y):
    model = RandomForestClassifier(max_depth=10,n_estimators=90,random_state=1)
    model.fit(X, y)
    return model


# In[29]:


#function for predicting random forest
def RF_pred(df, model):
    y_pred = model.predict_proba(X_val)[:, 1]
    return y_pred


# In[31]:


n_splits=10
dv = DictVectorizer(sparse=False)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
#creating a score list
scores = []
#creating loop for cross validation
for train_idx, val_idx in kfold.split(df_train):
    train = df_train.iloc[train_idx]
    val = df_train.iloc[val_idx]
    y_train = train.treatment.values
    y_val = val.treatment.values
    del train['treatment']
    del val['treatment']
    #encoding train dataset
    train_dict = train[cat_cols].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    #encoding validation set
    val_dict = val[cat_cols].to_dict(orient='records')
    X_val = dv.transform(val_dict)
    #training model
    model = RF_train(X_train,y_train)
    #predicting
    y_pred = RF_pred(X_val, model)
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
print(f'Mean= {np.mean(scores)}')
print(f'Standard Deviation= {np.std(scores)}')


# Saving the model

# In[32]:


import pickle
with open('capstone_model.pkl', 'wb') as f:
    pickle.dump((dv, model), f)


# In[ ]:




