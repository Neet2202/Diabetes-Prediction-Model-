#!/usr/bin/env python
# coding: utf-8

# # Program 4:Implement Decision Tree

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv(r"C:\Users\HP\Downloads\diabetes.csv")
df


# In[3]:


df.dtypes


# In[4]:


df.columns


# In[5]:


df.isnull().sum()


# In[6]:


print('N0. of zero values in Glucose',df[df['Glucose']==0].shape[0])
print('N0. of zero values in BloodPressure',df[df['BloodPressure']==0].shape[0])
print('N0. of zero values in SkinThickness',df[df['SkinThickness']==0].shape[0])
print('N0. of zero values in Insulin',df[df['Insulin']==0].shape[0])
print('N0. of zero values in BMI',df[df['BMI']==0].shape[0])


# In[7]:


#replace zero values with mean
df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())
df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())
df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].mean())
df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())
df['BMI']=df['BMI'].replace(0,df['BMI'].mean())


# In[8]:


print('N0. of zero values in Glucose',df[df['Glucose']==0].shape[0])
print('N0. of zero values in BloodPressure',df[df['BloodPressure']==0].shape[0])
print('N0. of zero values in SkinThickness',df[df['SkinThickness']==0].shape[0])
print('N0. of zero values in Insulin',df[df['Insulin']==0].shape[0])
print('N0. of zero values in BMI',df[df['BMI']==0].shape[0])


# In[9]:


x=df[['Pregnancies','Insulin','SkinThickness','BMI','Age','Glucose','BloodPressure','DiabetesPedigreeFunction']]
y=df['Outcome']


# In[10]:


x


# In[11]:


y.values


# In[12]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x)
X=scaler.transform(x)


# In[13]:


# Splitting Dataset into Training and Test Set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[14]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(max_depth=2,random_state=1)
clf=clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)


# In[15]:


from sklearn.metrics import plot_confusion_matrix
fig,ax=plt.subplots(figsize=(8,8))
disp=plot_confusion_matrix(clf,x_test,y_test,labels=np.unique(y),cmap=plt.cm.Blues,ax=ax)


# In[16]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


# In[17]:


from sklearn.metrics import accuracy_score,classification_report
print("Accuracy score of training data using gini index)" ,accuracy_score(y_true=y_train,y_pred=clf.predict(x_train)))
print("Accuracy score of testing data using gini index)",accuracy_score(y_true=y_test,y_pred=y_pred))


# In[18]:


print("classification Report")
print(classification_report(y_test,y_pred))


# In[20]:


target_names=['0','1']
feature_cols = x.columns
from sklearn import tree
fig = plt.figure(figsize=(25,20))
v= tree.plot_tree(clf, 
                   feature_names = feature_cols,class_names=target_names,
                   filled=True)


# In[21]:


#using entropy
clf = DecisionTreeClassifier(criterion='entropy',max_depth=5)
clf = clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)


# In[22]:


from sklearn.metrics import accuracy_score 
print("Accuracy score on training data(Using gini index)", accuracy_score(y_true = y_train,y_pred = clf.predict(x_train)))
print("Accuracy score on testing data(Using gini index)", accuracy_score(y_true = y_test,y_pred = y_pred))


# In[23]:


clf.predict([[22,55,78,20,1,25,0.523,30]])


# In[24]:


target_names=['0','1']
feature_cols = x.columns
from sklearn import tree
fig = plt.figure(figsize=(25,20))
v= tree.plot_tree(clf, 
                   feature_names = feature_cols,class_names=target_names,
                   filled=True)


# In[25]:


from sklearn import tree
import graphviz
# DOT data
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names = feature_cols,class_names=target_names,
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph


# In[26]:


#  Create Decision Tree classifer object using Entropy
clf = DecisionTreeClassifier(criterion='entropy',max_depth=6)
# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)


# In[27]:


from sklearn.metrics import accuracy_score 
print("Accuracy score on training data(Using gini index)", accuracy_score(y_true = y_train,y_pred = clf.predict(x_train)))
print("Accuracy score on testing data(Using gini index)", accuracy_score(y_true = y_test,y_pred = y_pred))


# In[28]:


clf.predict([[22,55,78,20,1,25,0.523,30]])


# In[ ]:




