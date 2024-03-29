```python
# Importing libraries that will be working with for the intial calculations and visualization
import numpy as np  # Numerical and linear algebra
import pandas as pd # data processing, CSV file (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import seaborn as sns # Data visualization
```


```python
# Importing dataset in csv to pandas data frame.
data = pd.read_csv("C:/Users/User/Desktop/Heart Rate/heart.csv")

print('Data Shape Show (On the left side represents the rows, and on the right side represents the columns):\n')
data.shape
```

    Data Shape Show (On the left side represents the rows, and on the right side represents the columns):
    
    




    (303, 14)



**From the collected data, we can see that there are 303 rows and 14 attributes that will be used for this data analysis.**


```python
#Now, our data is loaded. We're writing the following code to see some information from the data. 
#The purpose here is to see the top ten (or as many as needed) of the rows and all the attributes that are existed from
#this loaded data.

print('The first 10 rows of Data:\n')
data.head(10)
```

    The first 10 rows of Data:
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>5</td>
      <td>57</td>
      <td>1</td>
      <td>0</td>
      <td>140</td>
      <td>192</td>
      <td>0</td>
      <td>1</td>
      <td>148</td>
      <td>0</td>
      <td>0.4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>6</td>
      <td>56</td>
      <td>0</td>
      <td>1</td>
      <td>140</td>
      <td>294</td>
      <td>0</td>
      <td>0</td>
      <td>153</td>
      <td>0</td>
      <td>1.3</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>7</td>
      <td>44</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>263</td>
      <td>0</td>
      <td>1</td>
      <td>173</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <td>8</td>
      <td>52</td>
      <td>1</td>
      <td>2</td>
      <td>172</td>
      <td>199</td>
      <td>1</td>
      <td>1</td>
      <td>162</td>
      <td>0</td>
      <td>0.5</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <td>9</td>
      <td>57</td>
      <td>1</td>
      <td>2</td>
      <td>150</td>
      <td>168</td>
      <td>0</td>
      <td>1</td>
      <td>174</td>
      <td>0</td>
      <td>1.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Here, we would like to see the full mathematical description from our data that we are about to analyze. 

print('Description of the Data:\n')
data.describe()
```

    Description of the Data:
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>54.366337</td>
      <td>0.683168</td>
      <td>0.966997</td>
      <td>131.623762</td>
      <td>246.264026</td>
      <td>0.148515</td>
      <td>0.528053</td>
      <td>149.646865</td>
      <td>0.326733</td>
      <td>1.039604</td>
      <td>1.399340</td>
      <td>0.729373</td>
      <td>2.313531</td>
      <td>0.544554</td>
    </tr>
    <tr>
      <td>std</td>
      <td>9.082101</td>
      <td>0.466011</td>
      <td>1.032052</td>
      <td>17.538143</td>
      <td>51.830751</td>
      <td>0.356198</td>
      <td>0.525860</td>
      <td>22.905161</td>
      <td>0.469794</td>
      <td>1.161075</td>
      <td>0.616226</td>
      <td>1.022606</td>
      <td>0.612277</td>
      <td>0.498835</td>
    </tr>
    <tr>
      <td>min</td>
      <td>29.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>94.000000</td>
      <td>126.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>71.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>47.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>120.000000</td>
      <td>211.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>133.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>55.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>130.000000</td>
      <td>240.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>153.000000</td>
      <td>0.000000</td>
      <td>0.800000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>61.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>140.000000</td>
      <td>274.500000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>166.000000</td>
      <td>1.000000</td>
      <td>1.600000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>77.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>200.000000</td>
      <td>564.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>202.000000</td>
      <td>1.000000</td>
      <td>6.200000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Getting all related column information from the data used. 

data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 303 entries, 0 to 302
    Data columns (total 14 columns):
    age         303 non-null int64
    sex         303 non-null int64
    cp          303 non-null int64
    trestbps    303 non-null int64
    chol        303 non-null int64
    fbs         303 non-null int64
    restecg     303 non-null int64
    thalach     303 non-null int64
    exang       303 non-null int64
    oldpeak     303 non-null float64
    slope       303 non-null int64
    ca          303 non-null int64
    thal        303 non-null int64
    target      303 non-null int64
    dtypes: float64(1), int64(13)
    memory usage: 33.3 KB
    


```python
#Checking for the null values

print('Sum of Null Values of the Data \n')
data.isnull().sum()
```

    Sum of Null Values of the Data 
    
    




    age         0
    sex         0
    cp          0
    trestbps    0
    chol        0
    fbs         0
    restecg     0
    thalach     0
    exang       0
    oldpeak     0
    slope       0
    ca          0
    thal        0
    target      0
    dtype: int64



**As we can see, no null or blank information is presented for this dataset.**


```python
# Extracting brief descriprion of how the columns appear in the dataset

print('Column Names of the Data:\n')
data.columns
```

    Column Names of the Data:
    
    




    Index(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
           'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
          dtype='object')



## Dataset Columns Feature Explanation 

* **Age:** The person's *Age* in years


* **Sex:** The person's *Sex* (1 = **Male**, 0 = **Female**)


* **CP:** The *Chest Pain* experienced (1: **Typical Angina**, 2: **Atypical Angina**, 3: **Non-Anginal Pain**, 4: **Asymptomatic**)


* **TRESTBPS:** The person's *Resting Blood Pressure* (mmHg on admission to the hospital)


* **CHOL:** The person's *Cholesterol* measurement (in mg/dL)


* **FBS:** The person's *Fasting Blood Sugar* (> 120 mg/dL, 1 = **True**; 0 = **False**)


* **REST-ECG:** *Resting Electrocardiographic* measurement (0 = **Normal**, 1 = **Having ST-T wave Abnormality**, 2 = **Showing probable or definite left ventricular hypertrophy**)


* **THALACH:** The person's *Maximum Heart Rate* achieved


* **EXANG:** *Exercise Induced Angina* (1 = **Yes**; 0 = **No**)


* **OLDPEAK:** *ST Depression Induced* by exercise relative to rest ('ST' relates to positions on the ECG plot)


* **SLOPE:** The *Slope* of the peak exercise ST segment (1: **psloping**, 2: **Flat**, 3: **Downsloping**)


* **CA:** The number of *Major Vessels* (0-3)


* **THAL:** A blood disorder called *Thalassemia* (3 = **Normal**; 6 = **Fixed defect**; 7 = **Reversable Defect**)


* **TARGET:** *Heart Disease* (0 = **No**, 1 = **Yes**)


```python
#Reshaping names for the existing columns for better visualization and understanding.

data=data.rename(columns={'age':'Age','sex':'Sex','cp':'Chest_Pain_Type','trestbps':
                          'Resting_Blood_Pressure','chol':'Serum_Cholesterol','fbs':'Fasting_Blood_Sugar',
                          'restecg':'Rest_ECG','thalach':'Max_Heart_Rate','exang':'Exercise_Induced_Angina',
                          'oldpeak':'St_Depression','slope':'St_Slope','ca':'Number_Major_Vessels',
                          'thal':'Thalassemia','target':'Target'})
data.columns
```




    Index(['Age', 'Sex', 'Chest_Pain_Type', 'Resting_Blood_Pressure',
           'Serum_Cholesterol', 'Fasting_Blood_Sugar', 'Rest_ECG',
           'Max_Heart_Rate', 'Exercise_Induced_Angina', 'St_Depression',
           'St_Slope', 'Number_Major_Vessels', 'Thalassemia', 'Target'],
          dtype='object')




```python
# For easier analysis and clean results, listing the actual meanings of each parameter 

data.Target = data.Target.replace({0:'Heart Disease', 1:'No Heart Disease'})
data.Sex = data.Sex.replace({0:'Female', 1:'Male'})
data.Chest_Pain_Type = data.Chest_Pain_Type.replace({1:'Agina Pectoris', 2:'Atypical Agina', 3:'Non-Anginal Pain',
                                                     0:'Absent'})
data.St_Slope = data.St_Slope.replace({1:'Upsloping', 2:'Horizontal', 3:'Downsloping', 0:'Absent'})
data.Fasting_Blood_Sugar = data.Fasting_Blood_Sugar.replace({0:'Greater than 120mg/dL', 1:'Lower than 120mg/dL'})
data.Exercises_Induced_Angina = data.Exercise_Induced_Angina.replace({0:'No', 1:'Yes'})
data.Thalassemia = data.Thalassemia.replace({1:'Normal', 2:'Fixed defect', 3:'Reversable defect', 0:'Absent'})
```

    C:\Users\User\Anaconda3\lib\site-packages\ipykernel_launcher.py:9: UserWarning: Pandas doesn't allow columns to be created via a new attribute name - see https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access
      if __name__ == '__main__':
    


```python
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
```

    Number of people with no heart disease (165), heart disease (138). There are 207 Male and 96 Female
    
    




    [138, 165]




![png](output_12_2.png)


**The results are showing that there are 165 people who don't have heart disease and 138 who have. Also, from the right side, the data showing that there been collecting data among 207 men and 96 women.**


```python
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
```

    Number of male and female with heart disease and without
    
    




    [96, 207]




![png](output_14_2.png)


**Here we can see the ratios between men and women with and without having heart disease. From this data we can see that male are in great incidence of having heart disease.**


```python
#Looking on age distribution for men and women to see the age where the incident of getting heart disease 
#is most likely happening 

print('Age distribution for Male and Female and likelihood to get heart disease\n')

sns.distplot(data[data.Sex=="Male"].Age, color="y")
sns.distplot(data[data.Sex=="Female"].Age, color="r")
plt.xlabel("Age Distribution (Yellow = Male, Pink = Female)")
```

    Age distribution for Male and Female and likelihood to get heart disease
    
    




    Text(0.5, 0, 'Age Distribution (Yellow = Male, Pink = Female)')




![png](output_16_2.png)


**From here we see that the likelihood for getting heart disease for male and female is around age of 60.**

**Correlation coefficient** 

Mostly used is the Pearson Correlation Coefficient. If we are interested to observe linear relationship between the dataset that we have, Pearson Correlation Coefficient would be the best option. Correlation matrix from our dataset is measured between 1 to -1 where the value is close to 1 or -1 then we can assume there is strong positive correlation or strong negative correlation respectively. While the correlation is close to 0, we can conclude there is a weak correlation between the measurements.


```python
corr_matrix = data.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, 
            annot=True, 
            linewidths=0.1, 
            fmt= ".2f", 
            cmap="BrBG");
```


![png](output_19_0.png)


# Data Information Comparison 

**For the comparison part, we would like to investigate some causes and their effect on the Target (having or not having a heart condition), to evaluate their role of influencing the severity or easiness of the Target.** 


```python
#Chest pain type and Heart Disease association to both genders (Agina Pectoris, Atypical Agina, 
#Non-Anginal Pain or Absent)

fig, axs = plt.subplots(ncols=2)
sns.countplot(x="Chest_Pain_Type", hue="Target", data=data, palette="CMRmap", ax=axs[0])
sns.countplot(x="Chest_Pain_Type", hue="Sex", data=data, palette="Accent", ax=axs[1])
plt.gcf().set_size_inches(15, 5)
```


![png](output_22_0.png)


**For this comparison, we wanted to know the target (having heart disease) among people who are having different types of Angina. From the absent information, the highest amount of people who are getting heart disease don't come from angina, but probably from different factors. On the right side, we can clearly see again that men are higher in all the areas of being most vulnerable for getting heart conditions.**


```python
# Fasting blood sugar count between Male and Female 

fig, axs = plt.subplots(ncols=2)
sns.countplot(x="Fasting_Blood_Sugar", hue="Target", data=data, palette="CMRmap", ax=axs[0])
sns.countplot(x="Fasting_Blood_Sugar", hue="Sex", data=data, palette="Accent", ax=axs[1])
plt.gcf().set_size_inches(15, 5)
```


![png](output_24_0.png)


**Using fasting blood sugar indicating that male are again can get diagnosed with heart conditions with extimated blood sugar count of higher than 120mg/dL.**

# Association Analysis

**The conditional probability for this association rule will test to see Exercise Induced Angina. This will test if a person who just completed his/her exercise, what is the confidence for that person to experience chest pain that is caused by reduced blood flow to the heart due to his/her exercise?**


```python
angina =pd.crosstab(data['Target'],data['Exercise_Induced_Angina'],margins=True)
angina
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Exercise_Induced_Angina</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Target</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Heart Disease</td>
      <td>62</td>
      <td>76</td>
      <td>138</td>
    </tr>
    <tr>
      <td>No Heart Disease</td>
      <td>142</td>
      <td>23</td>
      <td>165</td>
    </tr>
    <tr>
      <td>All</td>
      <td>204</td>
      <td>99</td>
      <td>303</td>
    </tr>
  </tbody>
</table>
</div>




```python
prob_margin = (99/303) #The marginal probability for excercise induced angina
suppo = (76/303)
confidence = (suppo/prob_margin)
print(confidence)
```

    0.7676767676767676
    

**The confidence that is obtained showing ~77% which means that if the person have angina after exercise (Exercise_Induced_Angina = True) then the probabilty for that person to experience a heart condition is ~77%.**

## *Predictive Modeling* 

**Here we are going to compare some of the algorithms together.**

**1. Logistic Regression**

Is a machine learning categorization algorithm that is to predict the possibility of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that has data coded as 1 (considered to be 'yes', or success, etc.) or 0 (which is considered to be no, or failure, etc.). In other words, the logistic regression model predicts the probability of P(Y=1) as a function of using f(x).

**2. k-nearest neighbors (KNN)** 

k-Nearest Neighbors Classifier is a supervised machine learning algorithm or an instance-based classifier that can be used to determine both data classification and data regression problems. The common underlying idea is that the probability for the two occuring examples of the instance space that will be part of the same category or class will get higher with the closeness of the instance. This kind of closeness instance can be explained with a distance or similarity function.

**3. Decision Tree**

This predictive algorithm builds regression or classification models in the form of a tree branch structure. The main function of it is to break down a dataset into smaller and smaller subsets while at the same time an associated decision tree is gradually categorized. The results at the end that is achieved for the decision tree comes with nodes and leaf nodes. A decision node has at least two branches, that each representing values that have been used for the attribute has been tested. Leaf node represents a decision on the numerical target (has a heart disease or don't in our case). Climbing all the way up to the topmost decision node in a tree which corresponds to the best predictor called root node. Decision trees can handle both categorical and numerical data.

**4. Gaussian Naive Bayes**

Is supervised machine learning classification model that is following the Gaussian normal distribution and also supporting continuous data. This type of machine learing algorithm is commonly used to estimate the mean and the standard deviation from our training data.  


```python
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
```


```python
data = pd.read_csv("C:/Users/User/Desktop/Heart Rate/heart.csv", dtype={'sex':float})
```


```python
# Assigning test and train for our data for predictions

X = data.drop('target', axis=1)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


```python
#Assigning the predictive models that will be used for this experiment

Algo = [('LoRe', LogisticRegression()),('KNeCla', KNeighborsClassifier()),('DeTrClas', DecisionTreeClassifier()),
        ('GauNB', GaussianNB())]
```


```python
mod = []
category = []
for name, algorithm in Algo:
    kfold = KFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(algorithm, X_train, y_train, cv=kfold, scoring='accuracy')
    mod.append(cv_results)
    category.append(name)
    print(name, cv_results.mean(), cv_results.std())
```

    C:\Users\User\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\User\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\User\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\User\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\User\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\User\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\User\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\User\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\User\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\User\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    

    LoRe 0.8389610389610389 0.08073447998418022
    KNeCla 0.6785714285714286 0.11562271363176417
    DeTrClas 0.7406926406926406 0.10908231945843898
    GauNB 0.806060606060606 0.0730448128630229
    

### For easier and functional information, we will be creating a function that calculate everything with train and test we used


```python
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
```

**1. Logistic Regression**


```python
LoRe = LogisticRegression(solver='saga',penalty='elasticnet',l1_ratio=0.6,max_iter=1000)
LoRe.fit(X_train, y_train)

results(LoRe, X_train, y_train, X_test, y_test, train=True)
results(LoRe, X_train, y_train, X_test, y_test, train=False)
```

    Train Result:
    **********************************************
    Score for Accuracy: 71.70%
    _______________________________________________
    Report for Classification:
                       0           1  accuracy   macro avg  weighted avg
    precision   0.717647    0.716535  0.716981    0.717091      0.717044
    recall      0.628866    0.791304  0.716981    0.710085      0.716981
    f1-score    0.670330    0.752066  0.716981    0.711198      0.714668
    support    97.000000  115.000000  0.716981  212.000000    212.000000
    _______________________________________________
    Confusion Matrix: 
     [[61 36]
     [24 91]]
    
    Test Result:
    ************************************************
    Score for Accuracy: 81.32%
    _______________________________________________
    Report for Classification:
                       0          1  accuracy  macro avg  weighted avg
    precision   0.785714   0.836735  0.813187   0.811224      0.813747
    recall      0.804878   0.820000  0.813187   0.812439      0.813187
    f1-score    0.795181   0.828283  0.813187   0.811732      0.813369
    support    41.000000  50.000000  0.813187  91.000000     91.000000
    _______________________________________________
    Confusion Matrix: 
     [[33  8]
     [ 9 41]]
    
    

    C:\Users\User\Anaconda3\lib\site-packages\sklearn\linear_model\sag.py:337: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      "the coef_ did not converge", ConvergenceWarning)
    

**2. k-nearest neighbors (KNN)**


```python
KNeCla = KNeighborsClassifier(n_neighbors=5)
KNeCla.fit(X_train, y_train)
results(KNeCla, X_train, y_train, X_test, y_test, train=True)
results(KNeCla, X_train, y_train, X_test, y_test, train=False)
```

    Train Result:
    **********************************************
    Score for Accuracy: 76.89%
    _______________________________________________
    Report for Classification:
                       0           1  accuracy   macro avg  weighted avg
    precision   0.779070    0.761905  0.768868    0.770487      0.769759
    recall      0.690722    0.834783  0.768868    0.762752      0.768868
    f1-score    0.732240    0.796680  0.768868    0.764460      0.767196
    support    97.000000  115.000000  0.768868  212.000000    212.000000
    _______________________________________________
    Confusion Matrix: 
     [[67 30]
     [19 96]]
    
    Test Result:
    ************************************************
    Score for Accuracy: 65.93%
    _______________________________________________
    Report for Classification:
                       0          1  accuracy  macro avg  weighted avg
    precision   0.631579   0.679245  0.659341   0.655412      0.657769
    recall      0.585366   0.720000  0.659341   0.652683      0.659341
    f1-score    0.607595   0.699029  0.659341   0.653312      0.657834
    support    41.000000  50.000000  0.659341  91.000000     91.000000
    _______________________________________________
    Confusion Matrix: 
     [[24 17]
     [14 36]]
    
    

**3. Decision Tree**


```python
DeTrClas = DecisionTreeClassifier()
DeTrClas.fit(X_train, y_train)
results(DeTrClas, X_train, y_train, X_test, y_test, train=True)
results(DeTrClas, X_train, y_train, X_test, y_test, train=False)
```

    Train Result:
    **********************************************
    Score for Accuracy: 100.00%
    _______________________________________________
    Report for Classification:
                  0      1  accuracy  macro avg  weighted avg
    precision   1.0    1.0       1.0        1.0           1.0
    recall      1.0    1.0       1.0        1.0           1.0
    f1-score    1.0    1.0       1.0        1.0           1.0
    support    97.0  115.0       1.0      212.0         212.0
    _______________________________________________
    Confusion Matrix: 
     [[ 97   0]
     [  0 115]]
    
    Test Result:
    ************************************************
    Score for Accuracy: 75.82%
    _______________________________________________
    Report for Classification:
                       0          1  accuracy  macro avg  weighted avg
    precision   0.711111   0.804348  0.758242   0.757729      0.762340
    recall      0.780488   0.740000  0.758242   0.760244      0.758242
    f1-score    0.744186   0.770833  0.758242   0.757510      0.758827
    support    41.000000  50.000000  0.758242  91.000000     91.000000
    _______________________________________________
    Confusion Matrix: 
     [[32  9]
     [13 37]]
    
    

**4. Gaussian Naive Bayes**


```python
GauNB = GaussianNB()
GauNB.fit(X_train, y_train)
results(GauNB, X_train, y_train, X_test, y_test, train=True)
results(GauNB, X_train, y_train, X_test, y_test, train=False)
```

    Train Result:
    **********************************************
    Score for Accuracy: 83.02%
    _______________________________________________
    Report for Classification:
                       0           1  accuracy   macro avg  weighted avg
    precision   0.827957    0.831933  0.830189    0.829945      0.830114
    recall      0.793814    0.860870  0.830189    0.827342      0.830189
    f1-score    0.810526    0.846154  0.830189    0.828340      0.829853
    support    97.000000  115.000000  0.830189  212.000000    212.000000
    _______________________________________________
    Confusion Matrix: 
     [[77 20]
     [16 99]]
    
    Test Result:
    ************************************************
    Score for Accuracy: 83.52%
    _______________________________________________
    Report for Classification:
                       0          1  accuracy  macro avg  weighted avg
    precision   0.782609   0.888889  0.835165   0.835749      0.841004
    recall      0.878049   0.800000  0.835165   0.839024      0.835165
    f1-score    0.827586   0.842105  0.835165   0.834846      0.835564
    support    41.000000  50.000000  0.835165  91.000000     91.000000
    _______________________________________________
    Confusion Matrix: 
     [[36  5]
     [10 40]]
    
    

## Cross Validation

**Cross validation is the one of best choice to predict from the unseen data because the model can be trained with the many folds during the training. It is way better choice than a random selection. The most common type of cross-validation is k-fold. It involves splitting data into k-fold's and then testing a model on each.**


```python
from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = GauNB, X = X_train, y = y_train, cv = 10)
print("Cross validation of Gaussian Naive Bayes model = ",cross_validation)
print("Cross validation of Gaussian Naive Bayes model (in mean) = ",cross_validation.mean())
```

    Cross validation of Gaussian Naive Bayes model =  [0.81818182 0.90909091 0.81818182 0.90909091 0.77272727 0.76190476
     0.80952381 0.9        0.7        0.7       ]
    Cross validation of Gaussian Naive Bayes model (in mean) =  0.8098701298701299
    


```python
from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = DeTrClas, X = X_train, y = y_train, cv = 10)
print("Cross validation of Decision Tree Machine model = ",cross_validation)
print("Cross validation of Decision Tree Machine model (in mean) = ",cross_validation.mean())
```

    Cross validation of Decision Tree Machine model =  [0.86363636 0.68181818 0.68181818 0.59090909 0.68181818 0.71428571
     0.80952381 0.85       0.85       0.75      ]
    Cross validation of Decision Tree Machine model (in mean) =  0.7473809523809523
    

# To conclude
### K-Fold Cross validation is performed: 

**From Gaussian Naive Bayes model and Decision Tree Machine model we can conclude that Gaussian Naive Bayes regression is performing well in K-Fold Cross Validation**


```python

```
