#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# In[60]:


train_df = pd.read_csv(r'C:\Users\Avinash\Desktop\ml\train.csv')
test_df = pd.read_csv(r'C:\Users\Avinash\Desktop\ml\test.csv')


# In[61]:


train_df.isnull().sum()
test_df.isnull().sum()


# In[62]:


sns.countplot(x = 'Survived', hue = 'Sex',data = train_df)


# In[63]:


x = train_df.drop(['PassengerId','Name','Ticket'],axis = 1).copy(deep = True)
x_test = test_df.drop(['PassengerId','Name','Ticket'],axis = 1).copy(deep = True)


# In[64]:


names = train_df['Name'].str.split(',')
names = names.apply(lambda x: x[1]).str.split(" ")
names = names.apply(lambda x: x[1])
names1 = test_df['Name'].str.split(',')
names1 = names1.apply(lambda x_test: x_test[1]).str.split(" ")
names1 = names1.apply(lambda x_test: x_test[1])


# In[67]:


names.value_counts()
names1.value_counts()


# In[69]:


y = x.Survived.copy(deep = True)
x = x.drop('Survived',axis = 1)


# In[70]:


x_test.head()


# In[73]:


x['Cabin'] = x['Cabin'].fillna('U')
x['Cabin'] = x['Cabin'].apply(lambda x:x[0])
x_test['Cabin'] = x_test['Cabin'].fillna('U')
x_test['Cabin'] = x_test['Cabin'].apply(lambda x_test:x_test[0])
x['Cabin'],_ = pd.factorize(x['Cabin'], sort = True)
x_test['Cabin'],_ = pd.factorize(x_test['Cabin'], sort = True)


# In[75]:


x['Cabin'].value_counts()


# In[80]:


x['Family'] = x['SibSp']+x['Parch']+1
x_test['Family'] = x_test['SibSp']+x_test['Parch']+1
x['Title'] = names
x_test['Title'] = names1


# In[83]:


x.head()


# In[110]:


le = LabelEncoder()
x['Sex'],_ = pd.factorize(x['Sex'], sort = True)
x['Embarked'],_ = pd.factorize(x['Embarked'],sort = True)
x['Title'],_ = pd.factorize(x['Title'], sort = True)
x_test['Sex'],_ = pd.factorize(x_test['Sex'], sort = True)
x_test['Embarked'],_ = pd.factorize(x_test['Embarked'],sort = True)
x_test['Title'],_ = pd.factorize(x_test['Title'], sort = True)


# In[111]:


x_test.head()


# In[112]:


x.fillna(x.mean(),inplace = True)
x_test.fillna(x_test.mean(),inplace = True)


# In[113]:


x_test.head()


# In[114]:


x['Alone'] = 1
x['Alone'].loc[x['Family']>1] = 0
x['FareBin'] = pd.qcut(x['Fare'],4)
x['AgeBin'] = pd.cut(x['Age'].astype(int),5)
x_test['Alone'] = 1
x_test['Alone'].loc[x_test['Family']>1] = 0
x_test['FareBin'] = pd.qcut(x_test['Fare'],4)
x_test['AgeBin'] = pd.cut(x_test['Age'].astype(int),5)


# In[118]:


x['FareBin'] = le.fit_transform(x['FareBin'])
x['AgeBin'] = le.fit_transform(x['AgeBin'])
x_test['FareBin'] = le.fit_transform(x_test['FareBin'])
x_test['AgeBin'] = le.fit_transform(x_test['AgeBin'])


# In[122]:


from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
mm.fit(x)
x = mm.transform(x)
x_test = mm.transform(x_test)


# In[123]:


X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# In[128]:


Y_test


# In[124]:


print('X_train: {}, x_test {}, y_train {}, y_test {}'.format(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape))


# In[125]:


model = RandomForestClassifier(max_depth= 10, n_estimators= 100, warm_start= True, random_state=42, 
                                    criterion= 'gini', max_features= 'auto')


# In[135]:


print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[134]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x, y, cv=5)


# In[137]:


model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(Y_test, y_pred)
print("Accuracy: {}".format(acc))


# In[142]:


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(Y_test, y_pred))


# In[143]:


y_pred


# In[ ]:




