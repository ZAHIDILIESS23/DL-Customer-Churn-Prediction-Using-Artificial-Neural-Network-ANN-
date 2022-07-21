#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Prediction Using Artificial Neural Network (ANN)
# 
# Customer churn prediction is to measure why customers are leaving a business. In this tutorial we will be looking at customer churn in telecom business. We will build a deep learning model to predict the churn and use precision,recall, f1-score to measure performance of our model

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the data

# In[2]:


df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.sample(5)


# ## First of all, drop customerID column as it is of no use

# In[3]:


df.drop('customerID',axis='columns',inplace=True)


# In[4]:


df.dtypes


# Quick glance at above makes me realize that TotalCharges should be float but it is an object. Let's check what's going on with this column

# In[5]:


df.TotalCharges.values


# Ahh... it is string. Lets convert it to numbers

# In[6]:


pd.to_numeric(df.TotalCharges,errors='coerce').isnull()


# In[7]:


df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]


# In[8]:


df.shape


# In[9]:


df.iloc[488].TotalCharges


# In[10]:


df[df.TotalCharges!=' '].shape


# ## Remove rows with space in TotalCharges

# In[11]:


df1 = df[df.TotalCharges!=' ']
df1.shape


# In[12]:


df1.dtypes


# In[13]:


df1.TotalCharges = pd.to_numeric(df1.TotalCharges)


# In[14]:


df1.TotalCharges.values


# In[15]:


df1[df1.Churn=='No']


# ## Data Visualization

# In[21]:


tenure_churn_no = df1[df1.Churn=='No'].tenure
tenure_churn_yes = df1[df1.Churn=='Yes'].tenure

plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")
blood_sugar_men = [113, 85, 90, 150, 149, 88, 93, 115, 135, 80, 77, 82, 129]
blood_sugar_women = [67, 98, 89, 120, 133, 150, 84, 69, 89, 79, 120, 112, 100]


plt.hist([tenure_churn_yes, tenure_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()


# In[19]:


mc_churn_no = df1[df1.Churn=='No'].MonthlyCharges      
mc_churn_yes = df1[df1.Churn=='Yes'].MonthlyCharges      

plt.xlabel("Monthly Charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")

blood_sugar_men = [113, 85, 90, 150, 149, 88, 93, 115, 135, 80, 77, 82, 129]
blood_sugar_women = [67, 98, 89, 120, 133, 150, 84, 69, 89, 79, 120, 112, 100]

plt.hist([mc_churn_yes, mc_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()


# Many of the columns are yes, no etc. Let's print unique values in object columns to see data values

# In[22]:


def print_unique_col_values(df):
       for column in df:
            if df[column].dtypes=='object':
                print(f'{column}: {df[column].unique()}') 


# In[23]:


print_unique_col_values(df1)


# Some of the columns have no internet service or no phone service, that can be replaced with a simple No

# In[24]:


df1.replace('No internet service','No',inplace=True)
df1.replace('No phone service','No',inplace=True)


# In[25]:


print_unique_col_values(df1)


# ## Convert Yes and No to 1 or 0

# In[26]:


yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes': 1,'No': 0},inplace=True)


# In[27]:


for col in df1:
    print(f'{col}: {df1[col].unique()}') 


# In[28]:


df1['gender'].replace({'Female':1,'Male':0},inplace=True)


# In[29]:


df1.gender.unique()


# ## One hot encoding for categorical columns

# In[31]:


df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])
df2.columns


# In[32]:


df2.sample(5)


# In[33]:


df2.dtypes


# ## transforming everything to the same scaling

# In[35]:


cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])


# In[36]:


for col in df2:
    print(f'{col}: {df2[col].unique()}')


# ## Train test split

# In[37]:


X = df2.drop('Churn',axis='columns')
y = df2['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)


# In[38]:


X_train.shape


# In[39]:


X_test.shape


# In[40]:


X_train[:10]


# In[41]:


len(X_train.columns)


# ## Build a model (ANN) in tensorflow/keras

# In[42]:


import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)


# In[43]:


model.evaluate(X_test, y_test)


# In[44]:


yp = model.predict(X_test)
yp[:5]


# In[45]:


y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


# In[46]:


y_pred[:10]


# In[47]:


y_test[:10]


# In[48]:


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_pred))


# In[49]:


import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[50]:


y_test.shape


# ## Accuracy

# In[51]:


round((862+229)/(862+229+137+179),2)


# Precision for 0 class. i.e. Precision for customers who did not churn

# In[52]:



round(862/(862+179),2)


# Precision for 1 class. i.e. Precision for customers who actually churned

# In[53]:


round(229/(229+137),2)


# Recall for 0 class

# In[55]:


round(862/(862+137),2)


# In[56]:


round(229/(229+179),2)


# In[ ]:




