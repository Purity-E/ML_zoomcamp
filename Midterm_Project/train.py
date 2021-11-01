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
df = pd.read_csv(r'D:\Purity\ML_Zoomcamp\midterm project dataset\googleplaystore.csv')
print(df.shape) #getting the shape
df.head() #reading first five rows


# Data Cleaning

# In[3]:


#setting columns to lower case
df.columns = df.columns.str.lower() 
#replacing space with _
df.columns= df.columns.str.replace(' ', '_')
df.head()


# In[4]:


#checking for duplicates
print(df.app.unique().shape)
print(df.app.nunique())


# In[5]:


#dropping duplicates
df.drop_duplicates(subset= 'app', inplace=True)
print(df.shape)


# In[6]:


#checking variable types
df.dtypes


# In[7]:


#getting number of uniques for each column
df.nunique()


# Dealing with missing values

# In[8]:


#checking for missing values
df.isnull().sum()


# In[9]:


# plotting missing values 
ax = df.isna().sum().sort_values().plot(kind = 'barh', figsize = (9, 10))
plt.title('Percentage of Missing Values Per Column in Train Set', fontdict={'size':15})
for p in ax.patches:
    percentage ='{:,.0f}%'.format((p.get_width()/df.shape[0])*100)
    width, height =p.get_width(),p.get_height()
    x=p.get_x()+width+0.02
    y=p.get_y()+height/2
    ax.annotate(percentage,(x,y))


# In[10]:


#filling the missing values for column 'rating' with mean
df['rating'] = df['rating'].fillna(df['rating'].mean())


# In[11]:


df.isnull().sum()


# In[12]:


#dropping missing values for the rest of the columns
df.dropna(inplace=True)
df.isnull().sum()


# Value counts

# In[13]:


#creating list for categorical columns
cat_cols = list(df.select_dtypes(['object']).columns)
print(len(cat_cols))


# In[14]:


# value_counts for categorical columns
for col in cat_cols:
    print(col)
    print(df[col].value_counts())
    print('\n')


# Converting price and reviews to numerical columns

# In[15]:


#removing $ character from price
value_list = []
col_list = list(df['price'].values)
for val in col_list:
    val = val.replace('$','')
    value_list.append(val)
df['price'] = value_list


# In[16]:


#converting reviews and price to float
for col in ['reviews',  'price']:
    value_list = []
    col_list = list(df[col].values)
    for val in col_list:
        val = float(val)
        value_list.append(val)
    df[col] = value_list
df.dtypes


# Creating a datetime column  and extracting features

# In[17]:


#creating a column for datetime from the last updated column
from datetime import datetime
dates = []
date_list = list(df['last_updated'].values)
for date in date_list:
    date_object = datetime.strptime(date, "%B %d, %Y")#converting string to python datetime
    dates.append(date_object)
df['update_date'] = dates
df.head()


# In[18]:


#extracting features from datetime column
for date_feature in ['year', 'quarter', 'month']:
    df[date_feature] = getattr(df['update_date'].dt, date_feature)
df.head()


# In[19]:


df.dtypes


# In[20]:


#converting the variable types for date columns
cols = ['year', 'quarter', 'month']
for col in cols:
    df[col] = df[col].astype('object')


# EDA

# Descriptive analytics

# In[21]:


#creating list for categorical columns and numerical columns
cat_cols = ['category', 'size', 'installs', 'type', 'content_rating', 'genres',
            'last_updated','current_ver','android_ver', 'year','quarter', 'month']
num_cols = ['price', 'reviews']


# In[22]:


#summary of the numerical columns
df.describe()


# In[23]:


#histogram of target column
sns.histplot(df.rating, bins=50)


# In[24]:


#histogram of reviews column
sns.histplot(df.reviews, bins=50)


# In[25]:


#histogram of price column
sns.histplot(df.price, bins=50)


# In[26]:


#applying log to the ratings column
rating_logs = np.log1p(df.rating)
#ploting histogram of log rating
sns.histplot(rating_logs, bins=50)


# In[27]:


#creating column for log of rating
df['log_rating'] = rating_logs
df.head()


# Correlation Analysis and 
# Feature Importance

# In[28]:


#correlation of numeric columns with target variable
df[num_cols].corrwith(df.rating)


# In[29]:


#correlation of numeric column with log target variable
df[num_cols].corrwith(rating_logs)


# In[30]:


#turning the numerical column to categorical column
cat_rating = pd.qcut(df.rating, q=10)
cat_rating


# In[31]:


#mutual information scores for the categorical variables
from sklearn.metrics import mutual_info_score
#creating a function
def mutual_score(series):
    return mutual_info_score(series, cat_rating)
mi = df[cat_cols].apply(mutual_score) #getting mutual info scores
pd.set_option('display.max_rows', None) #setting to print all rows
print(mi.sort_values(ascending=False)) #sorting in descending order


# Splitting the dataset

# In[32]:


#importing train_test_split library
from sklearn.model_selection import train_test_split


# In[33]:


#split to get test set
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
#split to get validation set
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
#forming the label
y_train = df_train.log_rating.values
y_val = df_val.log_rating.values
y_test = df_test.log_rating.values


# In[34]:


print(df_train.shape), print(df_val.shape), print(df_test.shape)


# In[35]:


print(y_train.shape), print(y_val.shape), print(y_test.shape)


# One-Hot encoding of categorical variables

# In[36]:


#importing dictvectorizer
from sklearn.feature_extraction import DictVectorizer


# In[37]:


dv = DictVectorizer(sparse=False)
train_dict = df_train[['current_ver','installs'] + ['reviews']].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[['current_ver','installs'] + ['reviews']].to_dict(orient='records')
X_val = dv.transform(val_dict)

test_dict = df_test[['current_ver','installs'] + ['reviews']].to_dict(orient='records')
X_test = dv.transform(test_dict)




#importing libraries
from sklearn.metrics import mean_squared_error
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text


# Training the selected model

# In[ ]:


#training selected model
model = DecisionTreeRegressor(max_depth=6, min_samples_leaf=5)
model.fit(X_train, y_train)
model_pred = model.predict(X_val)
rmse = math.sqrt(mean_squared_error(y_val, model_pred))
print(f'The rmse is {rmse}')
rating = np.exp(model_pred)
print(f'The rating is {rating}')


# Saving the model using pickle

# In[ ]:


#saving model
import pickle
with open('app_rating_model.pkl', 'wb') as f:
    pickle.dump((dv, model), f)

