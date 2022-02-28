#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#importing datasets
data_set = pd.read_csv(r"C:\Users\User\Downloads\Loan_Approval_Prediction.csv")

data_set.head(5)


# In[2]:


data_set.loc[data_set['Loan_Status'] == 'Y', 'Loan_approval'] = 1
data_set.loc[data_set['Loan_Status'] == 'N', 'Loan_approval'] = 0
data_set.loc[data_set['Education'] == 'Graduate', 'Edu'] = 1
data_set.loc[data_set['Education'] == 'Not Graduate', 'Edu'] = 0
data_set.loc[data_set['Self_Employed'] == 'No', 'Self_Emp'] = 1
data_set.loc[data_set['Self_Employed'] == 'Yes', 'Self_Emp'] = 0
data_set.head()


# In[3]:


#Making a new column based on summation of salaries of applicants
data_set['Total Income'] = (data_set['ApplicantIncome'] + data_set['CoapplicantIncome'])
data_set.head()


# In[4]:


##Handling the missing data with 0 value
data_set = data_set.fillna(0)


# In[5]:


#Extracting independant and dependant variable
x = data_set.iloc[:, [14,15,16]].values
y = data_set.iloc[:, 13].values


# In[6]:


#Splitting the dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)


# In[7]:


#feature scaling
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)
y_test


# In[8]:


knn = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean')
knn.fit(x_train, y_train)


# In[9]:


y_pred = knn.predict(x_test)


# In[10]:


confusion_matrix(y_test, y_pred)


# In[11]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[12]:


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred)


# In[13]:


from sklearn.metrics import precision_score
precision_score(y_test, y_pred)


# In[14]:


from sklearn.metrics import recall_score
recall_score(y_test, y_pred)


# In[15]:


from sklearn.metrics import f1_score
f1_score(y_test, y_pred)


# In[16]:


error_rate = []
for i in range(1,40):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(x_train, y_train)
 pred_i = knn.predict(x_test)
 #print (pred_i)
 #print (1-accuracy_score(y_test, pred_i))
 error_rate.append(1-accuracy_score(y_test, pred_i))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate))+1)


# In[17]:


knn = KNeighborsClassifier(n_neighbors=17, metric='euclidean')
knn.fit(x_train, y_train)


# In[18]:


y_pred = knn.predict(x_test)
accuracy_score(y_test, y_pred)


# In[19]:


import pandas as pd
import numpy as np
# 1.2 For OS related operations
import os
import matplotlib.pyplot as plt

# 1.3 Class for applying multiple data transformation jobs
from sklearn.compose import ColumnTransformer as ct

# 1.4 Scale numeric data
from sklearn.preprocessing import StandardScaler as ss

# 1.5 One hot encode data--Convert to dummy
from sklearn.preprocessing import OneHotEncoder as ohe

# 1.6 for data splitting
from sklearn.model_selection import train_test_split

# 1.7 Modeler
from sklearn.tree import DecisionTreeClassifier as dt
###################### Random Forest ############################
from sklearn.ensemble import RandomForestClassifier 
#use RandomForestRegressor for regression problem
from sklearn.metrics import accuracy_score


# In[20]:


feature_cols = ['Loan_approval', 'Edu', 'Self_Emp']
x = data_set[feature_cols] # Features
y = data_set.Loan_Status # Target variable


# In[21]:


# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)


# In[22]:


#Feature Scaling
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[23]:


#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm

classifier = dt(criterion = 'entropy', random_state = 0)
#classifier = dt(criterion = 'entropy', random_state = 0, max_depth=4)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
#Y_pred, Y_test


# In[24]:


print("Predicted Values : ",Y_pred[1:50])


# In[25]:


print("Accuracy:",accuracy_score(Y_test, Y_pred))


# In[26]:


from sklearn import tree
plt.figure(figsize=(12,7))
tree.plot_tree(classifier, filled=True, fontsize=12)


# In[27]:


X_dataframe = x

grr = pd.plotting.scatter_matrix(X_dataframe, figsize=(25, 25), marker='o', hist_kwds={'bins': 20}, s=10, alpha=.8)


# In[28]:


#Plotting the Correlation Map

import seaborn as sns
plt.figure(figsize=(25,25))
sns.heatmap(x.corr(), center=0, annot=True)
plt.title("Correlation Map")
plt.show()


# In[29]:


print("Accuracy:",accuracy_score(Y_test, Y_pred))


# In[32]:


from sklearn.ensemble import RandomForestClassifier

#Create a Random Forest Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

# prediction on test set
y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[34]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
metrics.confusion_matrix(y_test, y_pred)


# In[35]:


print(classification_report(y_test, y_pred))


# In[ ]:




