#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries that will be working with for the intial calculations and visualization
import numpy as np  # Numerical and linear algebra
import pandas as pd # data processing, CSV file (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import seaborn as sns # Data visualization


# In[2]:


# Importing dataset in csv to pandas data frame.
data = pd.read_csv("C:/Users/User/Desktop/Heart Rate/heart.csv")

print('Data Shape Show (On the left side represents the rows, and on the right side represents the columns):\n')
data.shape


# **From the collected data, we can see that there are 303 rows and 14 attributes that will be used for this data analysis.**

# In[3]:


#Now, our data is loaded. We're writing the following code to see some information from the data. 
#The purpose here is to see the top ten (or as many as needed) of the rows and all the attributes that are existed from
#this loaded data.

print('The first 10 rows of Data:\n')
data.head(10)


# In[4]:


# Here, we would like to see the full mathematical description from our data that we are about to analyze. 

print('Description of the Data:\n')
data.describe()


# In[5]:


#Getting all related column information from the data used. 

data.info()


# In[6]:


#Checking for the null values

print('Sum of Null Values of the Data \n')
data.isnull().sum()


# **As we can see, no null or blank information is presented for this dataset.**

# In[7]:


# Extracting brief descriprion of how the columns appear in the dataset

print('Column Names of the Data:\n')
data.columns


# ## Dataset Columns Feature Explanation 
# 
# * **Age:** The person's *Age* in years
# 
# 
# * **Sex:** The person's *Sex* (1 = **Male**, 0 = **Female**)
# 
# 
# * **CP:** The *Chest Pain* experienced (1: **Typical Angina**, 2: **Atypical Angina**, 3: **Non-Anginal Pain**, 4: **Asymptomatic**)
# 
# 
# * **TRESTBPS:** The person's *Resting Blood Pressure* (mmHg on admission to the hospital)
# 
# 
# * **CHOL:** The person's *Cholesterol* measurement (in mg/dL)
# 
# 
# * **FBS:** The person's *Fasting Blood Sugar* (> 120 mg/dL, 1 = **True**; 0 = **False**)
# 
# 
# * **REST-ECG:** *Resting Electrocardiographic* measurement (0 = **Normal**, 1 = **Having ST-T wave Abnormality**, 2 = **Showing probable or definite left ventricular hypertrophy**)
# 
# 
# * **THALACH:** The person's *Maximum Heart Rate* achieved
# 
# 
# * **EXANG:** *Exercise Induced Angina* (1 = **Yes**; 0 = **No**)
# 
# 
# * **OLDPEAK:** *ST Depression Induced* by exercise relative to rest ('ST' relates to positions on the ECG plot)
# 
# 
# * **SLOPE:** The *Slope* of the peak exercise ST segment (1: **psloping**, 2: **Flat**, 3: **Downsloping**)
# 
# 
# * **CA:** The number of *Major Vessels* (0-3)
# 
# 
# * **THAL:** A blood disorder called *Thalassemia* (3 = **Normal**; 6 = **Fixed defect**; 7 = **Reversable Defect**)
# 
# 
# * **TARGET:** *Heart Disease* (0 = **No**, 1 = **Yes**)

# In[8]:


#Reshaping names for the existing columns for better visualization and understanding.

data=data.rename(columns={'age':'Age','sex':'Sex','cp':'Chest_Pain_Type','trestbps':
                          'Resting_Blood_Pressure','chol':'Serum_Cholesterol','fbs':'Fasting_Blood_Sugar',
                          'restecg':'Rest_ECG','thalach':'Max_Heart_Rate','exang':'Exercise_Induced_Angina',
                          'oldpeak':'St_Depression','slope':'St_Slope','ca':'Number_Major_Vessels',
                          'thal':'Thalassemia','target':'Target'})
data.columns


# In[9]:


# For easier analysis and clean results, listing the actual meanings of each parameter 

data.Target = data.Target.replace({0:'Heart Disease', 1:'No Heart Disease'})
data.Sex = data.Sex.replace({0:'Female', 1:'Male'})
data.Chest_Pain_Type = data.Chest_Pain_Type.replace({1:'Agina Pectoris', 2:'Atypical Agina', 3:'Non-Anginal Pain',
                                                     0:'Absent'})
data.St_Slope = data.St_Slope.replace({1:'Upsloping', 2:'Horizontal', 3:'Downsloping', 0:'Absent'})
data.Fasting_Blood_Sugar = data.Fasting_Blood_Sugar.replace({0:'Greater than 120mg/dL', 1:'Lower than 120mg/dL'})
data.Exercises_Induced_Angina = data.Exercise_Induced_Angina.replace({0:'No', 1:'Yes'})
data.Thalassemia = data.Thalassemia.replace({1:'Normal', 2:'Fixed defect', 3:'Reversable defect', 0:'Absent'})


# In[10]:


#Getting to know our data with number of males and females and the Target measurement

fig, axs = plt.subplots(ncols=2)
sns.countplot(data=data, x="Target", palette="YlOrBr", ax=axs[0])
sns.countplot(data=data, x="Sex", palette="cividis", ax=axs[1])
plt.gcf().set_size_inches(15, 5)

print('Number of people with no heart disease (165), heart disease (138). There are 207 Male and 96 Female\n')
types = np.unique(data.Target)
count_ = [0,0]
for j in data.Target:
    for idx,val in enumerate(types):
        if val == j: count_[idx] += 1 
count_


# **The results are showing that there are 165 people who don't have heart disease and 138 who have. Also, from the right side, the data showing that there been collecting data among 207 men and 96 women.**

# In[12]:


#Ratio between men with heart disease and men without heart disease and same for women

data = data.sort_values('Target', ascending=0)
sns.countplot(data=data, x="Target", hue="Sex", palette="cividis")

print('Number of male and female with heart disease and without\n')
types = np.unique(data.Sex)
count_ = [0,0]
for i in data.Sex:
    for idx,val in enumerate(types):
        if val == i: count_[idx] += 1 
count_


# **Here we can see the ratios between men and women with and without having heart disease. From this data we can see that male are in great incidence of having heart disease.**

# In[13]:


#Looking on age distribution for men and women to see the age where the incident of getting heart disease 
#is most likely happening 

print('Age distribution for Male and Female and likelihood to get heart disease\n')

sns.distplot(data[data.Sex=="Male"].Age, color="y")
sns.distplot(data[data.Sex=="Female"].Age, color="r")
plt.xlabel("Age Distribution (Yellow = Male, Pink = Female)")


# **From here we see that the likelihood for getting heart disease for male and female is around age of 60.**

# **Correlation coefficient** 
# 
# Mostly used is the Pearson Correlation Coefficient. If we are interested to observe linear relationship between the dataset that we have, Pearson Correlation Coefficient would be the best option. Correlation matrix from our dataset is measured between 1 to -1 where the value is close to 1 or -1 then we can assume there is strong positive correlation or strong negative correlation respectively. While the correlation is close to 0, we can conclude there is a weak correlation between the measurements.

# In[14]:


corr_matrix = data.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, 
            annot=True, 
            linewidths=0.1, 
            fmt= ".2f", 
            cmap="BrBG");


# # Data Information Comparison 

# **For the comparison part, we would like to investigate some causes and their effect on the Target (having or not having a heart condition), to evaluate their role of influencing the severity or easiness of the Target.** 

# In[15]:


#Chest pain type and Heart Disease association to both genders (Agina Pectoris, Atypical Agina, 
#Non-Anginal Pain or Absent)

fig, axs = plt.subplots(ncols=2)
sns.countplot(x="Chest_Pain_Type", hue="Target", data=data, palette="CMRmap", ax=axs[0])
sns.countplot(x="Chest_Pain_Type", hue="Sex", data=data, palette="Accent", ax=axs[1])
plt.gcf().set_size_inches(15, 5)


# **For this comparison, we wanted to know the target (having heart disease) among people who are having different types of Angina. From the absent information, the highest amount of people who are getting heart disease don't come from angina, but probably from different factors. On the right side, we can clearly see again that men are higher in all the areas of being most vulnerable for getting heart conditions.**

# In[16]:


# Fasting blood sugar count between Male and Female 

fig, axs = plt.subplots(ncols=2)
sns.countplot(x="Fasting_Blood_Sugar", hue="Target", data=data, palette="CMRmap", ax=axs[0])
sns.countplot(x="Fasting_Blood_Sugar", hue="Sex", data=data, palette="Accent", ax=axs[1])
plt.gcf().set_size_inches(15, 5)


# **Using fasting blood sugar indicating that male are again can get diagnosed with heart conditions with extimated blood sugar count of higher than 120mg/dL.**

# # Association Analysis

# **The conditional probability for this association rule will test to see Exercise Induced Angina. This will test if a person who just completed his/her exercise, what is the confidence for that person to experience chest pain that is caused by reduced blood flow to the heart due to his/her exercise?**

# In[17]:


angina =pd.crosstab(data['Target'],data['Exercise_Induced_Angina'],margins=True)
angina


# In[18]:


prob_margin = (99/303) #The marginal probability for excercise induced angina
suppo = (76/303)
confidence = (suppo/prob_margin)
print(confidence)


# **The confidence that is obtained showing ~77% which means that if the person have angina after exercise (Exercise_Induced_Angina = True) then the probabilty for that person to experience a heart condition is ~77%.**

# ## *Predictive Modeling* 

# **Here we are going to compare some of the algorithms together.**
# 
# **1. Logistic Regression**
# 
# Is a machine learning categorization algorithm that is to predict the possibility of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that has data coded as 1 (considered to be 'yes', or success, etc.) or 0 (which is considered to be no, or failure, etc.). In other words, the logistic regression model predicts the probability of P(Y=1) as a function of using f(x).
# 
# **2. k-nearest neighbors (KNN)** 
# 
# k-Nearest Neighbors Classifier is a supervised machine learning algorithm or an instance-based classifier that can be used to determine both data classification and data regression problems. The common underlying idea is that the probability for the two occuring examples of the instance space that will be part of the same category or class will get higher with the closeness of the instance. This kind of closeness instance can be explained with a distance or similarity function.
# 
# **3. Decision Tree**
# 
# This predictive algorithm builds regression or classification models in the form of a tree branch structure. The main function of it is to break down a dataset into smaller and smaller subsets while at the same time an associated decision tree is gradually categorized. The results at the end that is achieved for the decision tree comes with nodes and leaf nodes. A decision node has at least two branches, that each representing values that have been used for the attribute has been tested. Leaf node represents a decision on the numerical target (has a heart disease or don't in our case). Climbing all the way up to the topmost decision node in a tree which corresponds to the best predictor called root node. Decision trees can handle both categorical and numerical data.
# 
# **4. Gaussian Naive Bayes**
# 
# Is supervised machine learning classification model that is following the Gaussian normal distribution and also supporting continuous data. This type of machine learing algorithm is commonly used to estimate the mean and the standard deviation from our training data.  

# In[19]:


from sklearn.decomposition import PCA
from sklearn import linear_model, decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# In[20]:


data = pd.read_csv("C:/Users/User/Desktop/Heart Rate/heart.csv", dtype={'sex':float})


# In[21]:


# Assigning test and train for our data for predictions

X = data.drop('target', axis=1)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[22]:


#Assigning the predictive models that will be used for this experiment

Algo = [('LoRe', LogisticRegression()),('KNeCla', KNeighborsClassifier()),('DeTrClas', DecisionTreeClassifier()),
        ('GauNB', GaussianNB())]


# In[23]:


mod = []
category = []
for name, algorithm in Algo:
    kfold = KFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(algorithm, X_train, y_train, cv=kfold, scoring='accuracy')
    mod.append(cv_results)
    category.append(name)
    print(name, cv_results.mean(), cv_results.std())


# ### For easier and functional information, we will be creating a function that calculate everything with train and test we used

# In[24]:


def results(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        prediction = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, prediction, output_dict=True))
        print("Train Result:\n**********************************************")
        print(f"Score for Accuracy: {accuracy_score(y_train, prediction) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Report for Classification:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, prediction)}\n")
        
    elif train==False:
        prediction = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, prediction, output_dict=True))
        print("Test Result:\n************************************************")        
        print(f"Score for Accuracy: {accuracy_score(y_test, prediction) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Report for Classification:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, prediction)}\n")


# **1. Logistic Regression**

# In[25]:


LoRe = LogisticRegression(solver='saga',penalty='elasticnet',l1_ratio=0.6,max_iter=1000)
LoRe.fit(X_train, y_train)

results(LoRe, X_train, y_train, X_test, y_test, train=True)
results(LoRe, X_train, y_train, X_test, y_test, train=False)


# **2. k-nearest neighbors (KNN)**

# In[26]:


KNeCla = KNeighborsClassifier(n_neighbors=5)
KNeCla.fit(X_train, y_train)
results(KNeCla, X_train, y_train, X_test, y_test, train=True)
results(KNeCla, X_train, y_train, X_test, y_test, train=False)


# **3. Decision Tree**

# In[27]:


DeTrClas = DecisionTreeClassifier()
DeTrClas.fit(X_train, y_train)
results(DeTrClas, X_train, y_train, X_test, y_test, train=True)
results(DeTrClas, X_train, y_train, X_test, y_test, train=False)


# **4. Gaussian Naive Bayes**

# In[28]:


GauNB = GaussianNB()
GauNB.fit(X_train, y_train)
results(GauNB, X_train, y_train, X_test, y_test, train=True)
results(GauNB, X_train, y_train, X_test, y_test, train=False)


# ## Cross Validation

# **Cross validation is the one of best choice to predict from the unseen data because the model can be trained with the many folds during the training. It is way better choice than a random selection. The most common type of cross-validation is k-fold. It involves splitting data into k-fold's and then testing a model on each.**

# In[29]:


from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = GauNB, X = X_train, y = y_train, cv = 10)
print("Cross validation of Gaussian Naive Bayes model = ",cross_validation)
print("Cross validation of Gaussian Naive Bayes model (in mean) = ",cross_validation.mean())


# In[30]:


from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = DeTrClas, X = X_train, y = y_train, cv = 10)
print("Cross validation of Decision Tree Machine model = ",cross_validation)
print("Cross validation of Decision Tree Machine model (in mean) = ",cross_validation.mean())


# # To conclude
# ### K-Fold Cross validation is performed: 
# 
# **From Gaussian Naive Bayes model and Decision Tree Machine model we can conclude that Gaussian Naive Bayes regression is performing well in K-Fold Cross Validation**

# In[ ]:




