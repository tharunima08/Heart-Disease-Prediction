#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction Using Machine Learning

# ### Steps
# 1. Data gathering
# 2. Data preperation
# 3. Data Preprocessing
# 4. Data Transformation
# 5. Model Building
# 6. Model Evaluation

# In[48]:


# Importing Libraries Which are required for our Project.

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from matplotlib import rcParams
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# # Data Preperation

# In[49]:


df = pd.read_csv('HeartDisease.csv')
df.info()


# In[50]:


df.tail()


# In[51]:


pd.set_option("display.float", "{:.2f}".format)
df.describe()


# # Data Exploration

# In[52]:


df.target.value_counts().plot(kind="bar", color=["salmon", "lightblue"])
plt.xlabel('Patient has heart disease')
plt.ylabel('counts')
plt.title('Histogram of Patient has heart disease') 


# In[53]:


# Checking whether there are any null values or not.

df.isna().sum()


# In[54]:


categorical_val = []
continous_val = []
for column in df.columns:
    print('==============================')
    print(f"{column} : {df[column].unique()}")
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)


# In[55]:


plt.figure(figsize=(15, 15))

for i, column in enumerate(categorical_val, 1):
    plt.subplot(3, 3, i)
    df[df["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    df[df["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# Observations from the above plot:
# 
# - cp {Chest pain}: People with cp 1, 2, 3 are more likely to have heart disease than people with cp 0.
# - restecg {resting EKG results}: People with a value of 1 (reporting an abnormal heart rhythm, which can range from mild symptoms to severe problems) are more likely to have heart disease.
# - exang {exercise-induced angina}: people with a value of 0 (No ==> angina induced by exercise) have more heart disease than people with a value of 1 (Yes ==> angina induced by exercise)
# - slope {the slope of the ST segment of peak exercise}: People with a slope value of 2 (Downslopins: signs of an unhealthy heart) are more likely to have heart disease than people with a slope value of 2 slope is 0 (Upsloping: best heart rate with exercise) or 1 (Flatsloping: minimal change (typical healthy heart)).
# - ca {number of major vessels (0-3) stained by fluoroscopy}: the more blood movement the better, so people with ca equal to 0 are more likely to have heart disease.
# - thal {thalium stress result}: People with a thal value of 2 (defect corrected: once was a defect but ok now) are more likely to have heart disease.

# In[56]:


# Let's make our correlation matrix a little prettier
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 15))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# In[57]:


df.corrwith(df.target).plot(kind='bar', grid=True, figsize=(12, 8), 
                                                   title="Correlation with target")


# Observations from correlation:
# 
# - fbs and chol are the least correlated with the target variable.
# - All other variables have a significant correlation with the target variable.

# In[58]:


num_val = df[['age','rest_bps', 'cholestrol', 'thalach', 'old_peak']]
sns.pairplot(num_val)


# # Data Preprocessing

# In[59]:


target_var = df['target']
independent_features = df.drop(columns = ['target'])


# In[60]:


df = pd.get_dummies(independent_features, columns = ['gender', 'chest_pain', 'fasting_blood_sugar', 'rest_ecg', 'exer_angina', 'slope', 'ca', 'thalassemia'])


# In[61]:


df.head()


# # Data Transformation

# In[62]:


sc = StandardScaler()
col_to_scale = ['age', 'rest_bps', 'cholestrol', 'thalach', 'old_peak']
df[col_to_scale] = sc.fit_transform(df[col_to_scale])


# In[63]:


df.head()


# In[64]:


#df.describe()


# # Model Training and train Test Split

# In[65]:


X = df
y = target_var

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[66]:


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# # Logistic Regression

# In[67]:


lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)

print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)


# # Support Vector Machine

# In[68]:


svc_clf = SVC()
svc_clf.fit(X_train, y_train)

print_score(svc_clf, X_train, y_train, X_test, y_test, train=True)
print_score(svc_clf, X_train, y_train, X_test, y_test, train=False)


# # Naive Bayes

# In[69]:


GaussianNB_clf = GaussianNB()
GaussianNB_clf.fit(X_train, y_train)

print_score(GaussianNB_clf, X_train, y_train, X_test, y_test, train=True)
print_score(GaussianNB_clf, X_train, y_train, X_test, y_test, train=False)


# # Decision Tree

# In[70]:


dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

print_score(dt_clf, X_train, y_train, X_test, y_test, train=True)
print_score(dt_clf, X_train, y_train, X_test, y_test, train=False)


# In[ ]:





# In[71]:


aa=( 0.29046364,  0.47839125, -0.10172985, -1.16528085, -0.7243226 ,
         1.        ,  0.        ,  1.        ,  0.        ,  0.        ,
         0.        ,  1.        ,  0.        ,  0.        ,  1.        ,
         0.        ,  0.        ,  1.        ,  0.        ,  1.        ,
         0.        ,  1.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  1. )

ab=(0.95, 0.76, -0.26, 0.02,    1.09,   0.00,   1.00,    0.00,   
    0.00,    0.00,   1.00,  0.00,  1.00,   1.00,   0.00,  0.00,  
    1.00,  0.00, 1.00,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00,
    0.00,  0.00,   1.00,  0.00, 0.00)
a = np.asarray(aa)
a = a.reshape(1,-1)
p = svc_clf.predict(a)


# In[72]:


X_train.iloc[19]


# In[73]:


new = X_train.iloc[193]


# In[74]:


a = np.asarray(new)
a = a.reshape(1,-1)
p = svc_clf.predict(a)


# In[75]:


p[0]


# In[ ]:





# In[76]:


p[0]


# In[78]:


if (p[0] == 1):
    print("Person has heart disease")
else:
    print("Great! the results are normal")


# ### Conclusion
# 
# So, In this project, We have used Machine Learning to predict whether a person is suffering from a heart disease or not. 
# Steps which were involved along the project.
# 1. Data Collection.
# 2. Data Preperation(Importing, Exploratory Data Analysis), After importing the data we have used some basic pandas fucntions to get to know more about the data, such as, Head(), tail(), Descibe() -> for statistical analysis, info(), 
# 3. Data Exploration, we have used some plots to get an understanding of what our data is telling to us, like count of our target variable, Histogram to check whetehr the variables are normally distributed or not, Unique values present in a variable.
# 4. Data preprocessing, here we have checked whether their are any null vlaues, outliers or unwanted values, fortunaltely we didnt have any, so after that we have created dummy variables using pd.get_dummies(), to transform our categorical variables to numerical(0/1).
# 5. Data Transformation: this will help us to trasform the variables/ used to bring all the variables to the same scale, using StandardScaler()/Normalisation().
# 6. Data Modelling: Before modelling our data, we are splitting our data to trian_test_split(80/20), After which we have used 2 Machine Learning algorithms, `Logistic Regression` and  `Support Vector Classifier`. I varied parameters across each model to improve their scores.
# In the end, we can see that  `Support Vector Classifier`  has achieved better scores compared to other model.

# In[ ]:




