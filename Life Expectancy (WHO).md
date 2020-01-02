
<p style="font-family: Arial; font-size:2.75em;color:black; font-style:bold" align="center""><br>
World Life Expectancy (WHO)
</p><br>                                                                                              

<br><br><center><h1 style="font-size:2em;color:black">World's Population - 7.7 Billion and Growing...</h1></center>
<br>
<table>
<col width="750">
<col width="50">
<tr>
<td><img src="http://www.mapsnworld.com/blog/wp-content/uploads/2012/01/world-map.jpg
" align="center" style="width:1550px;height:460px;"/></td>
<td>
</td>
</tr>
</table>

<p style="font-family: Arial; font-size:2.75em;color:black; font-style:bold"><br>
Introduction
</p><br>

* It was found that the effect of immunization and human development index between developing and developed counties was not taken into account in the past. Most of the life expectancy studies were considered by demographic variables, income composition and mortality rates. For this srudy we will be considering dataset aspects from a period of 2000 to 2014 for all the countries (Developed and Developing). Important immunization like Hepatitis B, Polio, Measles and Diphtheria will be considered in addition to other health factors that can possibly affect life expectancy rates. 

* In addition to immunization factors, we will also focus on mortality, economic, social and a health related factors. Since the dataset is based on different countries, it will be easier for a country to determine the predicting factor which is contributing to lower value of life expectancy. The main idea of this project is to  suggest a country which area should be given importance in order to efficiently improve its life expectancy.

* Based on the results that were gathered from WHO in the past 15 years, there has been a huge development and awareness in health sector resulting in improvement of human mortality rates especially in the developing nations in comparison to the past 30 years. Therefore, in this project we have obtained data from year 2000-2014 for 134 countries for further analysis to analyse these developments. All predicting variables from the dataset were divided into several broad categories: Immunization related factors, Mortality factors, Economical factors and Social factors.

<p style="font-family: Arial; font-size:2.75em;color:black; font-style:bold"><br>
Part 1: Initial Exploration of the Dataset
</p><br>


```python
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
# Importing dataset in csv to pandas data frame.
data = pd.read_csv("C:/Users/Albert/Desktop/Data Science course/Life Expectancy WHO/Life Expectancy.csv")

# Testing the shape of the data
data.shape
```




    (2655, 16)




```python
# Droping rows with Null values.
data = data.dropna(axis=0)
```


```python
# To get more clear and easy to read data, we removing spaces from the column names.
data.columns = data.columns.str.replace(' ','')
```


```python
data.head(16)
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
      <th></th>
      <th>Country</th>
      <th>Year</th>
      <th>Status</th>
      <th>Lifeexpectancy</th>
      <th>AdultMortality</th>
      <th>infantdeaths</th>
      <th>Alcohol</th>
      <th>HepatitisB</th>
      <th>Measles</th>
      <th>under-fivedeaths</th>
      <th>Polio</th>
      <th>Diphtheria</th>
      <th>HIV/AIDS</th>
      <th>GDP</th>
      <th>Population</th>
      <th>Schooling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2014</td>
      <td>Developing</td>
      <td>59.9</td>
      <td>271.0</td>
      <td>64</td>
      <td>0.01</td>
      <td>62.0</td>
      <td>492</td>
      <td>86</td>
      <td>58.0</td>
      <td>62.0</td>
      <td>0.1</td>
      <td>612.696514</td>
      <td>327582.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2013</td>
      <td>Developing</td>
      <td>59.9</td>
      <td>268.0</td>
      <td>66</td>
      <td>0.01</td>
      <td>64.0</td>
      <td>430</td>
      <td>89</td>
      <td>62.0</td>
      <td>64.0</td>
      <td>0.1</td>
      <td>631.744976</td>
      <td>31731688.0</td>
      <td>9.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2012</td>
      <td>Developing</td>
      <td>59.5</td>
      <td>272.0</td>
      <td>69</td>
      <td>0.01</td>
      <td>67.0</td>
      <td>2787</td>
      <td>93</td>
      <td>67.0</td>
      <td>67.0</td>
      <td>0.1</td>
      <td>669.959000</td>
      <td>3696958.0</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2011</td>
      <td>Developing</td>
      <td>59.2</td>
      <td>275.0</td>
      <td>71</td>
      <td>0.01</td>
      <td>68.0</td>
      <td>3013</td>
      <td>97</td>
      <td>68.0</td>
      <td>68.0</td>
      <td>0.1</td>
      <td>63.537231</td>
      <td>2978599.0</td>
      <td>9.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>Developing</td>
      <td>58.8</td>
      <td>279.0</td>
      <td>74</td>
      <td>0.01</td>
      <td>66.0</td>
      <td>1989</td>
      <td>102</td>
      <td>66.0</td>
      <td>66.0</td>
      <td>0.1</td>
      <td>553.328940</td>
      <td>2883167.0</td>
      <td>9.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Afghanistan</td>
      <td>2009</td>
      <td>Developing</td>
      <td>58.6</td>
      <td>281.0</td>
      <td>77</td>
      <td>0.01</td>
      <td>63.0</td>
      <td>2861</td>
      <td>106</td>
      <td>63.0</td>
      <td>63.0</td>
      <td>0.1</td>
      <td>445.893298</td>
      <td>284331.0</td>
      <td>8.9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Afghanistan</td>
      <td>2008</td>
      <td>Developing</td>
      <td>58.1</td>
      <td>287.0</td>
      <td>80</td>
      <td>0.03</td>
      <td>64.0</td>
      <td>1599</td>
      <td>110</td>
      <td>64.0</td>
      <td>64.0</td>
      <td>0.1</td>
      <td>373.361116</td>
      <td>2729431.0</td>
      <td>8.7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Afghanistan</td>
      <td>2007</td>
      <td>Developing</td>
      <td>57.5</td>
      <td>295.0</td>
      <td>82</td>
      <td>0.02</td>
      <td>63.0</td>
      <td>1141</td>
      <td>113</td>
      <td>63.0</td>
      <td>63.0</td>
      <td>0.1</td>
      <td>369.835796</td>
      <td>26616792.0</td>
      <td>8.4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Afghanistan</td>
      <td>2006</td>
      <td>Developing</td>
      <td>57.3</td>
      <td>295.0</td>
      <td>84</td>
      <td>0.03</td>
      <td>64.0</td>
      <td>1990</td>
      <td>116</td>
      <td>58.0</td>
      <td>58.0</td>
      <td>0.1</td>
      <td>272.563770</td>
      <td>2589345.0</td>
      <td>8.1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Afghanistan</td>
      <td>2005</td>
      <td>Developing</td>
      <td>57.3</td>
      <td>291.0</td>
      <td>85</td>
      <td>0.02</td>
      <td>66.0</td>
      <td>1296</td>
      <td>118</td>
      <td>58.0</td>
      <td>58.0</td>
      <td>0.1</td>
      <td>25.294130</td>
      <td>257798.0</td>
      <td>7.9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Afghanistan</td>
      <td>2004</td>
      <td>Developing</td>
      <td>57.0</td>
      <td>293.0</td>
      <td>87</td>
      <td>0.02</td>
      <td>67.0</td>
      <td>466</td>
      <td>120</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>0.1</td>
      <td>219.141353</td>
      <td>24118979.0</td>
      <td>6.8</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Afghanistan</td>
      <td>2003</td>
      <td>Developing</td>
      <td>56.7</td>
      <td>295.0</td>
      <td>87</td>
      <td>0.01</td>
      <td>65.0</td>
      <td>798</td>
      <td>122</td>
      <td>41.0</td>
      <td>41.0</td>
      <td>0.1</td>
      <td>198.728544</td>
      <td>2364851.0</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Afghanistan</td>
      <td>2002</td>
      <td>Developing</td>
      <td>56.2</td>
      <td>3.0</td>
      <td>88</td>
      <td>0.01</td>
      <td>64.0</td>
      <td>2486</td>
      <td>122</td>
      <td>36.0</td>
      <td>36.0</td>
      <td>0.1</td>
      <td>187.845950</td>
      <td>21979923.0</td>
      <td>6.2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Afghanistan</td>
      <td>2001</td>
      <td>Developing</td>
      <td>55.3</td>
      <td>316.0</td>
      <td>88</td>
      <td>0.01</td>
      <td>63.0</td>
      <td>8762</td>
      <td>122</td>
      <td>35.0</td>
      <td>33.0</td>
      <td>0.1</td>
      <td>117.496980</td>
      <td>2966463.0</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>Developing</td>
      <td>54.8</td>
      <td>321.0</td>
      <td>88</td>
      <td>0.01</td>
      <td>62.0</td>
      <td>6532</td>
      <td>122</td>
      <td>24.0</td>
      <td>24.0</td>
      <td>0.1</td>
      <td>114.560000</td>
      <td>293756.0</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Albania</td>
      <td>2014</td>
      <td>Developing</td>
      <td>77.5</td>
      <td>8.0</td>
      <td>0</td>
      <td>4.51</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>98.0</td>
      <td>98.0</td>
      <td>0.1</td>
      <td>4575.763787</td>
      <td>288914.0</td>
      <td>14.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Number of countries the data was collected from 
country = data['Country'].unique().tolist()
len(country)
```




    134




```python
# Getting the total number of years for each counry
years = data['Year'].unique().tolist()
len(years)
```




    15




```python
# Range period of the years the data was collected
print(min(years)," to ",max(years))
```

    (2000L, ' to ', 2014L)
    

* For the first part, we carefully prepared and studied all the set of information our dataset has. Next part will be to extract the information from the data to find the causes for life expectancy and its abnormalities.

<p style="font-family: Arial; font-size:2.75em;color:black; font-style:bold"><br>
Part 2: World's Immunized Population Rates 
</p><br>


```python
# Calculating total Immunized population between years 2000 - 2014
data.groupby('Year').Population.sum()
```




    Year
    2000    6.316683e+08
    2001    6.352206e+08
    2002    5.209537e+08
    2003    1.038549e+09
    2004    2.141666e+09
    2005    2.519828e+09
    2006    2.123688e+09
    2007    2.593902e+09
    2008    1.286632e+09
    2009    1.268106e+09
    2010    1.650681e+09
    2011    1.586026e+09
    2012    1.755421e+09
    2013    1.761761e+09
    2014    2.954990e+09
    Name: Population, dtype: float64



* From this data we can clearly observe constant increase of population awareness and importance of immunization from developing and developed countries. There is a slight decrease at year 2008, but then gradual increase where the immunization level was doubled in 2014.  


```python
# Immunized life Expectancy from Developed and Developing Countries between 2000 - 2014
plt.figure(figsize=(6,6))

plt.bar(data.groupby('Status')['Status'].count().index,data.groupby('Status')['Lifeexpectancy'].mean())
plt.xlabel("Development Status",fontsize=15)
plt.ylabel("Average Life Expectancy",fontsize=15)
plt.title("Life Expectancy Comparison")

plt.show()
```


![png](output_18_0.png)


* From this plot we observe that there is still slight difference between developing and developed countries in terms of life expectancy. We will try to investigate what are the reasons for this difference.


```python
# Life expectancy Through Years
plt.figure(figsize=(7,5))
sns.set(style='whitegrid')

life_expectancy = sns.barplot(data.groupby('Year')['Year'].count().index,data.groupby('Year')['Lifeexpectancy'].sum(),color='black')
plt.xlabel("Year",fontsize=15), plt.ylabel("Life Expectancy",fontsize=15), plt.title("Life Expectancy Through Years")

plt.show()
```


![png](output_20_0.png)


* Life expectancy is constantly going up for the past 15 years. This indication can relate into implemented actions the countries are making for healthier and longer life.  


```python
# Contry/ies with the highest life expectancy level
data[data['Lifeexpectancy'] == data['Lifeexpectancy'].max()]
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
      <th></th>
      <th>Country</th>
      <th>Year</th>
      <th>Status</th>
      <th>Lifeexpectancy</th>
      <th>AdultMortality</th>
      <th>infantdeaths</th>
      <th>Alcohol</th>
      <th>HepatitisB</th>
      <th>Measles</th>
      <th>under-fivedeaths</th>
      <th>Polio</th>
      <th>Diphtheria</th>
      <th>HIV/AIDS</th>
      <th>GDP</th>
      <th>Population</th>
      <th>Schooling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>224</th>
      <td>Belgium</td>
      <td>2014</td>
      <td>Developed</td>
      <td>89.0</td>
      <td>76.0</td>
      <td>0</td>
      <td>12.60</td>
      <td>98.0</td>
      <td>70</td>
      <td>1</td>
      <td>99.0</td>
      <td>99.0</td>
      <td>0.1</td>
      <td>47439.39684</td>
      <td>112957.0</td>
      <td>16.3</td>
    </tr>
    <tr>
      <th>830</th>
      <td>France</td>
      <td>2008</td>
      <td>Developed</td>
      <td>89.0</td>
      <td>88.0</td>
      <td>3</td>
      <td>11.90</td>
      <td>47.0</td>
      <td>604</td>
      <td>3</td>
      <td>98.0</td>
      <td>98.0</td>
      <td>0.1</td>
      <td>45413.65710</td>
      <td>6437499.0</td>
      <td>16.1</td>
    </tr>
    <tr>
      <th>831</th>
      <td>France</td>
      <td>2007</td>
      <td>Developed</td>
      <td>89.0</td>
      <td>89.0</td>
      <td>3</td>
      <td>12.20</td>
      <td>42.0</td>
      <td>39</td>
      <td>3</td>
      <td>99.0</td>
      <td>98.0</td>
      <td>0.1</td>
      <td>416.58397</td>
      <td>6416229.0</td>
      <td>16.1</td>
    </tr>
    <tr>
      <th>884</th>
      <td>Germany</td>
      <td>2014</td>
      <td>Developed</td>
      <td>89.0</td>
      <td>69.0</td>
      <td>2</td>
      <td>11.03</td>
      <td>88.0</td>
      <td>443</td>
      <td>3</td>
      <td>94.0</td>
      <td>95.0</td>
      <td>0.1</td>
      <td>4792.65288</td>
      <td>89825.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>1149</th>
      <td>Italy</td>
      <td>2004</td>
      <td>Developed</td>
      <td>89.0</td>
      <td>66.0</td>
      <td>2</td>
      <td>8.98</td>
      <td>96.0</td>
      <td>599</td>
      <td>3</td>
      <td>97.0</td>
      <td>94.0</td>
      <td>0.1</td>
      <td>31174.56118</td>
      <td>57685327.0</td>
      <td>15.6</td>
    </tr>
    <tr>
      <th>1860</th>
      <td>Portugal</td>
      <td>2014</td>
      <td>Developed</td>
      <td>89.0</td>
      <td>78.0</td>
      <td>0</td>
      <td>9.88</td>
      <td>98.0</td>
      <td>0</td>
      <td>0</td>
      <td>98.0</td>
      <td>98.0</td>
      <td>0.1</td>
      <td>2277.53613</td>
      <td>14162.0</td>
      <td>16.8</td>
    </tr>
    <tr>
      <th>2182</th>
      <td>Spain</td>
      <td>2007</td>
      <td>Developed</td>
      <td>89.0</td>
      <td>72.0</td>
      <td>2</td>
      <td>11.05</td>
      <td>96.0</td>
      <td>267</td>
      <td>2</td>
      <td>96.0</td>
      <td>96.0</td>
      <td>0.1</td>
      <td>3279.41400</td>
      <td>4522683.0</td>
      <td>16.0</td>
    </tr>
  </tbody>
</table>
</div>



* This chart shows that even though the life expectancy is constantly going up, the countries which are at the top of the chart are from developed regions. We need to understand whether developed countries are more aware of the immunization benefits or maybe there are additional factors. 


```python
# Contry/ies with the lowest life expectancy level
data[data['Lifeexpectancy'] == data['Lifeexpectancy'].min()]
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
      <th></th>
      <th>Country</th>
      <th>Year</th>
      <th>Status</th>
      <th>Lifeexpectancy</th>
      <th>AdultMortality</th>
      <th>infantdeaths</th>
      <th>Alcohol</th>
      <th>HepatitisB</th>
      <th>Measles</th>
      <th>under-fivedeaths</th>
      <th>Polio</th>
      <th>Diphtheria</th>
      <th>HIV/AIDS</th>
      <th>GDP</th>
      <th>Population</th>
      <th>Schooling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1421</th>
      <td>Malawi</td>
      <td>2002</td>
      <td>Developing</td>
      <td>44.0</td>
      <td>67.0</td>
      <td>46</td>
      <td>1.1</td>
      <td>64.0</td>
      <td>92</td>
      <td>75</td>
      <td>79.0</td>
      <td>64.0</td>
      <td>24.7</td>
      <td>29.979898</td>
      <td>1213711.0</td>
      <td>10.4</td>
    </tr>
  </tbody>
</table>
</div>



* Not surprisingly in this chart we see that the countries who have the lowest life expectancy are from developing regions. 

<p style="font-family: Arial; font-size:2.75em;color:black; font-style:bold"><br>
Part 3: Mortality Rates and Immunization
</p><br>

#### In this part we will investigate groups of mortality between developed and developing countries. We will extract and plot from our dataset the immunization factors that are being used to explore whether they made any impact of world's life expectancy. 


```python
# Measuring the average life expectancy and mortality between 2000 - 2014.
plt.figure(figsize=(30,20))
sns.set(style='whitegrid')

plt.subplot(3,2,1)
life_expectancy = sns.barplot(data=data,x=data.Country[1:150],y=data['Lifeexpectancy'],hue='Status')
life_expectancy.set_title("Life Expectancy Across Each Country",fontsize=20)

plt.subplot(3,2,2)
adult_mortality = sns.barplot(data=data,x=data.Country[1:150],y=data['AdultMortality'],hue='Status')
adult_mortality.set_title("AdultMortality Across Each Country",fontsize=20)

plt.subplot(3,2,3)
infant_death = sns.barplot(data=data,x=data.Country[1:150],y=data['infantdeaths'],hue='Status')
infant_death.set_title("Infant Deaths Across Each Country",fontsize=20)

plt.subplot(3,2,4)
under_five = sns.barplot(data=data,x=data.Country[1:150],y=data['under-fivedeaths'],hue='Status')
under_five.set_title("Under 5-Years Old Deaths Across Each Country",fontsize=20)


plt.show()
```


![png](output_28_0.png)


These group of graphs inicate the following:
   * From these set of plots we can observe that hight percentage of mortality are coming from developing countries. However, since life expectancy between developed and developing countries are very close it is hard to predict whether immunization is the main factor for the higher developing countries mortality. There might be additional reasons why developing countries are not correlated with the developed countries mortality rates


```python
# Immunization Coverage across each country between 2000 - 2014
plt.figure(figsize=(30,20))
sns.set(style='whitegrid')

plt.subplot(3,2,1)
hepatitisb = sns.barplot(data=data,x=data.Country[1:150],y=data['HepatitisB'],hue='Status')
hepatitisb.set_title("HepatitisB Across Each Country",fontsize=20)

plt.subplot(3,2,2)
measles = sns.barplot(data=data,x=data.Country[1:150],y=data['Measles'],hue='Status')
measles.set_title("Measles Across Each Country",fontsize=20)

plt.subplot(3,2,3)
polio = sns.barplot(data=data,x=data.Country[1:150],y=data['Polio'],hue='Status')
polio.set_title("Polio Across Each Country",fontsize=20)

plt.subplot(3,2,4)
diphtheria = sns.barplot(data=data,x=data.Country[1:150],y=data['Diphtheria'],hue='Status')
diphtheria.set_title("Diphtheria Across Each Country",fontsize=20)

plt.show()
```


![png](output_30_0.png)


These group of graphs inicate the following:
   * It's clearly seen that developed and developing countries are fully aware of being vaccinated. So, from these set of plots we can estimate that there might be correlation between immunization and life expectancy. However,  it's not clear for Measles if we can rule it out, since there is no record that developed countries are immuned and very low percentage of developing countries are being vaccinated. Perhaps developing countries are not fully aware of Measles and therefore not many of them considering the importance to immune themselves.   


```python
# 3 countries with lowest percentange of immunization coverage for HepatitisB
data = data.groupby('Country').mean().nsmallest(3,'HepatitisB').reset_index()
data
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
      <th></th>
      <th>Country</th>
      <th>Year</th>
      <th>Lifeexpectancy</th>
      <th>AdultMortality</th>
      <th>infantdeaths</th>
      <th>Alcohol</th>
      <th>HepatitisB</th>
      <th>Measles</th>
      <th>under-fivedeaths</th>
      <th>Polio</th>
      <th>Diphtheria</th>
      <th>HIV/AIDS</th>
      <th>GDP</th>
      <th>Population</th>
      <th>Schooling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Equatorial Guinea</td>
      <td>2014.0</td>
      <td>57.900000</td>
      <td>32.000000</td>
      <td>3.000000</td>
      <td>0.010000</td>
      <td>2.000000</td>
      <td>13.000000</td>
      <td>4.000000</td>
      <td>24.000000</td>
      <td>2.000000</td>
      <td>4.400000</td>
      <td>192.597330</td>
      <td>1.129424e+06</td>
      <td>9.200000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chad</td>
      <td>2011.0</td>
      <td>52.285714</td>
      <td>322.142857</td>
      <td>46.000000</td>
      <td>0.464286</td>
      <td>27.571429</td>
      <td>1527.571429</td>
      <td>79.142857</td>
      <td>31.857143</td>
      <td>27.857143</td>
      <td>3.814286</td>
      <td>712.233898</td>
      <td>7.677455e+06</td>
      <td>6.800000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>India</td>
      <td>2009.0</td>
      <td>66.000000</td>
      <td>116.636364</td>
      <td>1268.818182</td>
      <td>2.264545</td>
      <td>30.272727</td>
      <td>43188.545455</td>
      <td>1681.818182</td>
      <td>72.545455</td>
      <td>68.090909</td>
      <td>0.245455</td>
      <td>900.009910</td>
      <td>5.943872e+08</td>
      <td>10.409091</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 3 countries with lowest percentange of immunization coverage for Polio
data = data.groupby('Country').mean().nsmallest(3,'Polio').reset_index()
data
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
      <th></th>
      <th>Country</th>
      <th>Year</th>
      <th>Lifeexpectancy</th>
      <th>AdultMortality</th>
      <th>infantdeaths</th>
      <th>Alcohol</th>
      <th>HepatitisB</th>
      <th>Measles</th>
      <th>under-fivedeaths</th>
      <th>Polio</th>
      <th>Diphtheria</th>
      <th>HIV/AIDS</th>
      <th>GDP</th>
      <th>Population</th>
      <th>Schooling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Equatorial Guinea</td>
      <td>2014.0</td>
      <td>57.900000</td>
      <td>32.000000</td>
      <td>3.000000</td>
      <td>0.010000</td>
      <td>2.000000</td>
      <td>13.000000</td>
      <td>4.000000</td>
      <td>24.000000</td>
      <td>2.000000</td>
      <td>4.400000</td>
      <td>192.597330</td>
      <td>1.129424e+06</td>
      <td>9.200000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chad</td>
      <td>2011.0</td>
      <td>52.285714</td>
      <td>322.142857</td>
      <td>46.000000</td>
      <td>0.464286</td>
      <td>27.571429</td>
      <td>1527.571429</td>
      <td>79.142857</td>
      <td>31.857143</td>
      <td>27.857143</td>
      <td>3.814286</td>
      <td>712.233898</td>
      <td>7.677455e+06</td>
      <td>6.800000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>India</td>
      <td>2009.0</td>
      <td>66.000000</td>
      <td>116.636364</td>
      <td>1268.818182</td>
      <td>2.264545</td>
      <td>30.272727</td>
      <td>43188.545455</td>
      <td>1681.818182</td>
      <td>72.545455</td>
      <td>68.090909</td>
      <td>0.245455</td>
      <td>900.009910</td>
      <td>5.943872e+08</td>
      <td>10.409091</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 3 countries with lowest percentange of immunization coverage for Diphtheria
data = data.groupby('Country').mean().nsmallest(3,'Diphtheria').reset_index()
data
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
      <th></th>
      <th>Country</th>
      <th>Year</th>
      <th>Lifeexpectancy</th>
      <th>AdultMortality</th>
      <th>infantdeaths</th>
      <th>Alcohol</th>
      <th>HepatitisB</th>
      <th>Measles</th>
      <th>under-fivedeaths</th>
      <th>Polio</th>
      <th>Diphtheria</th>
      <th>HIV/AIDS</th>
      <th>GDP</th>
      <th>Population</th>
      <th>Schooling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Equatorial Guinea</td>
      <td>2014.0</td>
      <td>57.900000</td>
      <td>32.000000</td>
      <td>3.000000</td>
      <td>0.010000</td>
      <td>2.000000</td>
      <td>13.000000</td>
      <td>4.000000</td>
      <td>24.000000</td>
      <td>2.000000</td>
      <td>4.400000</td>
      <td>192.597330</td>
      <td>1.129424e+06</td>
      <td>9.200000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chad</td>
      <td>2011.0</td>
      <td>52.285714</td>
      <td>322.142857</td>
      <td>46.000000</td>
      <td>0.464286</td>
      <td>27.571429</td>
      <td>1527.571429</td>
      <td>79.142857</td>
      <td>31.857143</td>
      <td>27.857143</td>
      <td>3.814286</td>
      <td>712.233898</td>
      <td>7.677455e+06</td>
      <td>6.800000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>India</td>
      <td>2009.0</td>
      <td>66.000000</td>
      <td>116.636364</td>
      <td>1268.818182</td>
      <td>2.264545</td>
      <td>30.272727</td>
      <td>43188.545455</td>
      <td>1681.818182</td>
      <td>72.545455</td>
      <td>68.090909</td>
      <td>0.245455</td>
      <td>900.009910</td>
      <td>5.943872e+08</td>
      <td>10.409091</td>
    </tr>
  </tbody>
</table>
</div>



* Even though both developed and developing countries are immunizing themselves against HepatitisB, Polio and Diphtheria. We can still see that the countries who have the lowest percentage of immunizaion coverage are coming from developing regions. 


```python
#countries with the lowest Adult Mortalities
data[data['AdultMortality'] == data['AdultMortality'].min()]
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
      <th></th>
      <th>Country</th>
      <th>Year</th>
      <th>Lifeexpectancy</th>
      <th>AdultMortality</th>
      <th>infantdeaths</th>
      <th>Alcohol</th>
      <th>HepatitisB</th>
      <th>Measles</th>
      <th>under-fivedeaths</th>
      <th>Polio</th>
      <th>Diphtheria</th>
      <th>HIV/AIDS</th>
      <th>GDP</th>
      <th>Population</th>
      <th>Schooling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Equatorial Guinea</td>
      <td>2014.0</td>
      <td>57.9</td>
      <td>32.0</td>
      <td>3.0</td>
      <td>0.01</td>
      <td>2.0</td>
      <td>13.0</td>
      <td>4.0</td>
      <td>24.0</td>
      <td>2.0</td>
      <td>4.4</td>
      <td>192.59733</td>
      <td>1129424.0</td>
      <td>9.2</td>
    </tr>
  </tbody>
</table>
</div>



* For this part we wanted to focuse on adult mortality since by looking on the chart we can see that most of the countries who have the lowest adult mortallty are from developing regions. So, perhaps there are other reasons why life expectancy in developing countries are lower and maybe being immunized don't play main role in life expectancy. 

<p style="font-family: Arial; font-size:2.75em;color:black; font-style:bold"><br>
Part 4: Other Possible Reasons for Abnormalities 
</p><br>

#### In this part we will investigate additional possibilities to understand whether immunization has a major factor for life expectancy or not. We will introduce additional daily life styles that may be another factor for the abnormalities. 


```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
```


```python
# Importing dataset in csv to pandas data frame.
data = pd.read_csv("C:/Users/Albert/Desktop/Data Science course/Life Expectancy WHO/Life Expectancy.csv")
```


```python
# Droping rows with Null values.
data = data.dropna(axis=0)
```


```python
# To get more clear and easy to read data, we removing spaces from the column names.
data.columns = data.columns.str.replace(' ','')
```


```python
#countries with the lowest Adult Mortalities
#data[data['AdultMortality'] == data['AdultMortality'].min()]
```


```python
# GDP, Alcohol, and Schools accross each country between 2000 - 2014
plt.figure(figsize=(30,20))
sns.set(style='whitegrid')

plt.subplot(3,2,1)
aids = sns.barplot(data=data,x=data.Country[1:500],y=data['HIV/AIDS'],hue='Status')
aids.set_title("HIV/AIDS",fontsize=20)
plt.show()
```


![png](output_45_0.png)


This chart shows that people who are sick from HIV are mainly coming from developing countries. Since there is no vaccine against HIV, then maybe this can be a possible reason that life expectancy there is lower. We need to keep exploring to see if there are additional factors to solidify our theory. 


```python
# GDP, Alcohol, and Schools accross each country between 2000 - 2014
plt.figure(figsize=(30,20))
sns.set(style='whitegrid')

plt.subplot(3,2,1)
gdp = sns.barplot(data=data,x=data.Country[1:150],y=data['GDP'],hue='Status')
gdp.set_title("GDP",fontsize=20)

plt.subplot(3,2,2)
alcohol = sns.barplot(data=data,x=data.Country[1:150],y=data['Alcohol'],hue='Status')
alcohol.set_title("Alcohol Usage",fontsize=20)

plt.subplot(3,2,3)
school = sns.barplot(data=data,x=data.Country[1:150],y=data['Schooling'],hue='Status')
school.set_title("Schooling and Awareness",fontsize=20)

plt.show()
```


![png](output_47_0.png)


* From GDP plot, we can assume that since developing countries have lower average of GDP per capita, that may cause lower life expectancy. But, since the economy from each country is different we can't possibly factor this assumption and we need to explore the actual economical incentives.
* From Alcohol plot, it's showing that developed countries are higher consumers. Therefore, it's doesn't support the reason for them to have higher life expectancy.  
* From Schooling plot, it's hard to define the rows since there is no big difference between developed and developing countries. At the beginning we could assume that if developing countries have low schooling, then maybe they are lacking of health related factors which could result low life expectancy.

<p style="font-family: Arial; font-size:2.75em;color:black; font-style:bold"><br>
Conclusion 
</p><br>

* In this project, we analyzed dataset from WHO to find out why life expectancy between developed and developing countries are different. To sum up, based of the information which was collected by WHO we couldn't be able to find convincing factors to conclude what couses the difference in their life expectancy. The reason is because the dataset doesn't provide additional information to investigate deeply and into specific categories.

* Some of the specific categories should be:
    * Men vs. women
    * Smoking 
    * Stress
    * Main diseases like: Cancer, heart attack, stroke, diabetes...etc  
    * suicide rates 
* These categories might give us better understanding and reasons of how life expectancy between developing and developed countries are different.

<p style="font-family: Arial; font-size:2.75em;color:black; font-style:bold"><br>
Additional Visualization (A/B Testing)
</p><br>


```python
# 10 countries with lowest immunization coverage for Hepatitis-B
data = data.groupby('Country').mean().nsmallest(10,'HepatitisB').reset_index()
data
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
      <th></th>
      <th>Country</th>
      <th>Year</th>
      <th>Lifeexpectancy</th>
      <th>AdultMortality</th>
      <th>infantdeaths</th>
      <th>Alcohol</th>
      <th>HepatitisB</th>
      <th>Measles</th>
      <th>under-fivedeaths</th>
      <th>Polio</th>
      <th>Diphtheria</th>
      <th>HIV/AIDS</th>
      <th>GDP</th>
      <th>Population</th>
      <th>Schooling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Equatorial Guinea</td>
      <td>2014.0</td>
      <td>57.900000</td>
      <td>32.000000</td>
      <td>3.000000</td>
      <td>0.010000</td>
      <td>2.000000</td>
      <td>13.000000</td>
      <td>4.000000</td>
      <td>24.000000</td>
      <td>2.000000</td>
      <td>4.400000</td>
      <td>192.597330</td>
      <td>1.129424e+06</td>
      <td>9.200000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chad</td>
      <td>2011.0</td>
      <td>52.285714</td>
      <td>322.142857</td>
      <td>46.000000</td>
      <td>0.464286</td>
      <td>27.571429</td>
      <td>1527.571429</td>
      <td>79.142857</td>
      <td>31.857143</td>
      <td>27.857143</td>
      <td>3.814286</td>
      <td>712.233898</td>
      <td>7.677455e+06</td>
      <td>6.800000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>India</td>
      <td>2009.0</td>
      <td>66.000000</td>
      <td>116.636364</td>
      <td>1268.818182</td>
      <td>2.264545</td>
      <td>30.272727</td>
      <td>43188.545455</td>
      <td>1681.818182</td>
      <td>72.545455</td>
      <td>68.090909</td>
      <td>0.245455</td>
      <td>900.009910</td>
      <td>5.943872e+08</td>
      <td>10.409091</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Central African Republic</td>
      <td>2011.5</td>
      <td>51.416667</td>
      <td>444.833333</td>
      <td>16.166667</td>
      <td>0.820000</td>
      <td>41.833333</td>
      <td>273.166667</td>
      <td>23.500000</td>
      <td>42.500000</td>
      <td>41.833333</td>
      <td>5.733333</td>
      <td>431.961740</td>
      <td>3.072260e+06</td>
      <td>6.850000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nigeria</td>
      <td>2009.5</td>
      <td>52.840000</td>
      <td>340.200000</td>
      <td>523.700000</td>
      <td>8.177000</td>
      <td>42.300000</td>
      <td>21896.400000</td>
      <td>825.400000</td>
      <td>45.600000</td>
      <td>43.700000</td>
      <td>4.750000</td>
      <td>1675.523082</td>
      <td>5.459305e+07</td>
      <td>9.410000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Montenegro</td>
      <td>2010.0</td>
      <td>75.066667</td>
      <td>72.333333</td>
      <td>0.000000</td>
      <td>4.014444</td>
      <td>44.777778</td>
      <td>1.111111</td>
      <td>0.000000</td>
      <td>83.777778</td>
      <td>84.000000</td>
      <td>0.100000</td>
      <td>6613.087641</td>
      <td>2.965488e+05</td>
      <td>14.555556</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Liberia</td>
      <td>2011.0</td>
      <td>60.814286</td>
      <td>277.714286</td>
      <td>9.142857</td>
      <td>2.145714</td>
      <td>48.714286</td>
      <td>366.142857</td>
      <td>12.285714</td>
      <td>63.285714</td>
      <td>47.000000</td>
      <td>1.571429</td>
      <td>328.764246</td>
      <td>2.911058e+06</td>
      <td>9.700000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>France</td>
      <td>2007.0</td>
      <td>82.206667</td>
      <td>72.800000</td>
      <td>3.000000</td>
      <td>12.404667</td>
      <td>48.933333</td>
      <td>2828.600000</td>
      <td>3.466667</td>
      <td>98.266667</td>
      <td>98.066667</td>
      <td>0.100000</td>
      <td>25794.803399</td>
      <td>2.897635e+07</td>
      <td>15.873333</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Samoa</td>
      <td>2007.0</td>
      <td>73.593333</td>
      <td>134.133333</td>
      <td>0.000000</td>
      <td>2.704000</td>
      <td>54.666667</td>
      <td>0.600000</td>
      <td>0.000000</td>
      <td>57.000000</td>
      <td>62.733333</td>
      <td>0.100000</td>
      <td>1986.437795</td>
      <td>1.272855e+05</td>
      <td>12.626667</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Belize</td>
      <td>2007.0</td>
      <td>69.153333</td>
      <td>154.200000</td>
      <td>0.000000</td>
      <td>6.252667</td>
      <td>54.866667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>58.333333</td>
      <td>60.266667</td>
      <td>0.446667</td>
      <td>3871.879820</td>
      <td>1.577999e+05</td>
      <td>12.433333</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 10 countries with lowest coverage for Hepatitis-B and with the high Infant death 
plt.figure(figsize=(14,5))
sns.set(style='dark')

plt.subplot(1,2,1)
hepatitis_b = sns.barplot(data=data,x=data.Country,y=data['HepatitisB'])
hepatitis_b.set_title("10 countries with low immunization for Hepatitis-B",fontsize=13)
hepatitis_b.set_xticklabels(hepatitis_b.get_xticklabels(), rotation=90, ha="right", fontsize=12)

plt.subplot(1,2,2)
infant_death = sns.barplot(data=data,x=data.Country,y=data['infantdeaths'])
infant_death.set_title("infant deaths across each country",fontsize=13)
infant_death.set_xticklabels(infant_death.get_xticklabels(), rotation=90, ha="right", fontsize=12)


#ax = plt.scatter(x='HepatitisB',y='infantdeaths',data=data)
plt.show()
```


![png](output_53_0.png)


#### Analysis: 
* The data we obtained showing that, 'India' is the country with high infant deaths and second low immunization coverage for Hepatitis-B. Though 'Equatorial Guinea' is the country with least Hepatitis-B immunization coverage, the infant deaths count for it is low (close to 0). Among the top 10 countries with low immunization coverage for Hepatitis-B, India has the highest infant death.

## Statistical way of the experiment: A/B Testing

#### Hypothesis: 
* Increasing Hepatitis-B immunization coverage among infants should decrease infant deaths.

#### Experiment: 
* 5000 infants will be randomly selected from the total population with low income. Low income families should agree to anonymous data collection and ashould follow the guidelines in exchange for a nominal financial incentive.

#### Treatment: 
* The organization will be offering immunization coverage for Hepatitis-B and infants will be under close observation for one year.

#### The Success of mentioned experiment:
* If exclusively immunized infant deaths is at least 10% less than infants without immunization coverage, the null hypothesis that Hepatitis-B vaccination has no impact on infant deaths can be rejected.

<br><br><center><h1 style="font-size:2em;color:black">2018</h1></center>
<br>
<table>
<col width="650">
<col width="50">
<tr>
<td><img src="https://i.redd.it/k25bd3e1muh01.png
" align="center" style="width:3950px;height:360px;"/></td>
<td>
</td>
</tr>
</table>
