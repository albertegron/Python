#!/usr/bin/env python
# coding: utf-8

# <p style="font-family: Arial; font-size:2.75em;color:black; font-style:bold; " align="center""><br>
# World Life Expectancy (WHO)
# </p><br>

# <br><br><center><h1 style="font-size:2em;color:black">World's Population - 7.7 Billion and Growing...</h1></center>
# <br>
# <table>
# <col width="750">
# <col width="50">
# <tr>
# <td><img src="http://www.mapsnworld.com/blog/wp-content/uploads/2012/01/world-map.jpg
# " align="center" style="width:1550px;height:460px;"/></td>
# <td>
# </td>
# </tr>
# </table>

# <p style="font-family: Arial; font-size:2.75em;color:black; font-style:bold"><br>
# Introduction
# </p><br>
# 
# * It was found that the effect of immunization and human development index between developing and developed counties was not taken into account in the past. Most of the life expectancy studies were considered by demographic variables, income composition and mortality rates. For this srudy we will be considering dataset aspects from a period of 2000 to 2014 for all the countries (Developed and Developing). Important immunization like Hepatitis B, Polio, Measles and Diphtheria will be considered in addition to other health factors that can possibly affect life expectancy rates. 

# * In addition to immunization factors, we will also focus on mortality, economic, social and a health related factors. Since the dataset is based on different countries, it will be easier for a country to determine the predicting factor which is contributing to lower value of life expectancy. The main idea of this project is to  suggest a country which area should be given importance in order to efficiently improve its life expectancy.

# * Based on the results that were gathered from WHO in the past 15 years, there has been a huge development and awareness in health sector resulting in improvement of human mortality rates especially in the developing nations in comparison to the past 30 years. Therefore, in this project we have obtained data from year 2000-2014 for 134 countries for further analysis to analyse these developments. All predicting variables from the dataset were divided into several broad categories: Immunization related factors, Mortality factors, Economical factors and Social factors.

# <p style="font-family: Arial; font-size:2.75em;color:black; font-style:bold"><br>
# Part 1: Initial Exploration of the Dataset
# </p><br>

# In[1]:


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# Importing dataset in csv to pandas data frame.
data = pd.read_csv("C:/Users/Albert/Desktop/Data Science course/Life Expectancy WHO/Life Expectancy.csv")

# Testing the shape of the data
data.shape


# In[5]:


# Droping rows with Null values.
data = data.dropna(axis=0)


# In[6]:


# To get more clear and easy to read data, we removing spaces from the column names.
data.columns = data.columns.str.replace(' ','')


# In[7]:


data.head(16)


# In[8]:


# Number of countries the data was collected from 
country = data['Country'].unique().tolist()
len(country)


# In[9]:


# Getting the total number of years for each counry
years = data['Year'].unique().tolist()
len(years)


# In[10]:


# Range period of the years the data was collected
print(min(years)," to ",max(years))


# * For the first part, we carefully prepared and studied all the set of information our dataset has. Next part will be to extract the information from the data to find the causes for life expectancy and its abnormalities.

# <p style="font-family: Arial; font-size:2.75em;color:black; font-style:bold"><br>
# Part 2: World's Immunized Population Rates 
# </p><br>

# In[11]:


# Calculating total Immunized population between years 2000 - 2014
data.groupby('Year').Population.sum()


# * From this data we can clearly observe constant increase of population awareness and importance of immunization from developing and developed countries. There is a slight decrease at year 2008, but then gradual increase where the immunization level was doubled in 2014.  

# In[12]:


# Immunized life Expectancy from Developed and Developing Countries between 2000 - 2014
plt.figure(figsize=(6,6))

plt.bar(data.groupby('Status')['Status'].count().index,data.groupby('Status')['Lifeexpectancy'].mean())
plt.xlabel("Development Status",fontsize=15)
plt.ylabel("Average Life Expectancy",fontsize=15)
plt.title("Life Expectancy Comparison")

plt.show()


# * From this plot we observe that there is still slight difference between developing and developed countries in terms of life expectancy. We will try to investigate what are the reasons for this difference.

# In[13]:


# Life expectancy Through Years
plt.figure(figsize=(7,5))
sns.set(style='whitegrid')

life_expectancy = sns.barplot(data.groupby('Year')['Year'].count().index,data.groupby('Year')['Lifeexpectancy'].sum(),color='black')
plt.xlabel("Year",fontsize=15), plt.ylabel("Life Expectancy",fontsize=15), plt.title("Life Expectancy Through Years")

plt.show()


# * Life expectancy is constantly going up for the past 15 years. This indication can relate into implemented actions the countries are making for healthier and longer life.  

# In[14]:


# Contry/ies with the highest life expectancy level
data[data['Lifeexpectancy'] == data['Lifeexpectancy'].max()]


# * This chart shows that even though the life expectancy is constantly going up, the countries which are at the top of the chart are from developed regions. We need to understand whether developed countries are more aware of the immunization benefits or maybe there are additional factors. 

# In[15]:


# Contry/ies with the lowest life expectancy level
data[data['Lifeexpectancy'] == data['Lifeexpectancy'].min()]


# * Not surprisingly in this chart we see that the countries who have the lowest life expectancy are from developing regions. 

# <p style="font-family: Arial; font-size:2.75em;color:black; font-style:bold"><br>
# Part 3: Mortality Rates and Immunization
# </p><br>

# #### In this part we will investigate groups of mortality between developed and developing countries. We will extract and plot from our dataset the immunization factors that are being used to explore whether they made any impact of world's life expectancy. 

# In[17]:


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


# These group of graphs inicate the following:
#    * From these set of plots we can observe that hight percentage of mortality are coming from developing countries. However, since life expectancy between developed and developing countries are very close it is hard to predict whether immunization is the main factor for the higher developing countries mortality. There might be additional reasons why developing countries are not correlated with the developed countries mortality rates

# In[18]:


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


# These group of graphs inicate the following:
#    * It's clearly seen that developed and developing countries are fully aware of being vaccinated. So, from these set of plots we can estimate that there might be correlation between immunization and life expectancy. However,  it's not clear for Measles if we can rule it out, since there is no record that developed countries are immuned and very low percentage of developing countries are being vaccinated. Perhaps developing countries are not fully aware of Measles and therefore not many of them considering the importance to immune themselves.   

# In[19]:


# 3 countries with lowest percentange of immunization coverage for HepatitisB
data = data.groupby('Country').mean().nsmallest(3,'HepatitisB').reset_index()
data


# In[20]:


# 3 countries with lowest percentange of immunization coverage for Polio
data = data.groupby('Country').mean().nsmallest(3,'Polio').reset_index()
data


# In[21]:


# 3 countries with lowest percentange of immunization coverage for Diphtheria
data = data.groupby('Country').mean().nsmallest(3,'Diphtheria').reset_index()
data


# * Even though both developed and developing countries are immunizing themselves against HepatitisB, Polio and Diphtheria. We can still see that the countries who have the lowest percentage of immunizaion coverage are coming from developing regions. 

# In[22]:


#countries with the lowest Adult Mortalities
data[data['AdultMortality'] == data['AdultMortality'].min()]


# * For this part we wanted to focuse on adult mortality since by looking on the chart we can see that most of the countries who have the lowest adult mortallty are from developing regions. So, perhaps there are other reasons why life expectancy in developing countries are lower and maybe being immunized don't play main role in life expectancy. 

# <p style="font-family: Arial; font-size:2.75em;color:black; font-style:bold"><br>
# Part 4: Other Possible Reasons for Abnormalities 
# </p><br>

# #### In this part we will investigate additional possibilities to understand whether immunization has a major factor for life expectancy or not. We will introduce additional daily life styles that may be another factor for the abnormalities. 

# In[23]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[24]:


# Importing dataset in csv to pandas data frame.
data = pd.read_csv("C:/Users/Albert/Desktop/Data Science course/Life Expectancy WHO/Life Expectancy.csv")


# In[25]:


# Droping rows with Null values.
data = data.dropna(axis=0)


# In[26]:


# To get more clear and easy to read data, we removing spaces from the column names.
data.columns = data.columns.str.replace(' ','')


# In[30]:


#countries with the lowest Adult Mortalities
#data[data['AdultMortality'] == data['AdultMortality'].min()]


# In[27]:


# GDP, Alcohol, and Schools accross each country between 2000 - 2014
plt.figure(figsize=(30,20))
sns.set(style='whitegrid')

plt.subplot(3,2,1)
aids = sns.barplot(data=data,x=data.Country[1:500],y=data['HIV/AIDS'],hue='Status')
aids.set_title("HIV/AIDS",fontsize=20)
plt.show()


# This chart shows that people who are sick from HIV are mainly coming from developing countries. Since there is no vaccine against HIV, then maybe this can be a possible reason that life expectancy there is lower. We need to keep exploring to see if there are additional factors to solidify our theory. 

# In[28]:


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


# * From GDP plot, we can assume that since developing countries have lower average of GDP per capita, that may cause lower life expectancy. But, since the economy from each country is different we can't possibly factor this assumption and we need to explore the actual economical incentives.
# * From Alcohol plot, it's showing that developed countries are higher consumers. Therefore, it's doesn't support the reason for them to have higher life expectancy.  
# * From Schooling plot, it's hard to define the rows since there is no big difference between developed and developing countries. At the beginning we could assume that if developing countries have low schooling, then maybe they are lacking of health related factors which could result low life expectancy.

# <p style="font-family: Arial; font-size:2.75em;color:black; font-style:bold"><br>
# Conclusion 
# </p><br>

# * In this project, we analyzed dataset from WHO to find out why life expectancy between developed and developing countries are different. To sum up, based of the information which was collected by WHO we couldn't be able to find convincing factors to conclude what couses the difference in their life expectancy. The reason is because the dataset doesn't provide additional information to investigate deeply and into specific categories.
# 
# * Some of the specific categories should be:
#     * Men vs. women
#     * Smoking 
#     * Stress
#     * Main diseases like: Cancer, heart attack, stroke, diabetes...etc  
#     * suicide rates 
# * These categories might give us better understanding and reasons of how life expectancy between developing and developed countries are different.

# <p style="font-family: Arial; font-size:2.75em;color:black; font-style:bold"><br>
# Additional Visualization (A/B Testing)
# </p><br>

# In[29]:


# 10 countries with lowest immunization coverage for Hepatitis-B
data = data.groupby('Country').mean().nsmallest(10,'HepatitisB').reset_index()
data


# In[30]:


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


# #### Analysis: 
# * The data we obtained showing that, 'India' is the country with high infant deaths and second low immunization coverage for Hepatitis-B. Though 'Equatorial Guinea' is the country with least Hepatitis-B immunization coverage, the infant deaths count for it is low (close to 0). Among the top 10 countries with low immunization coverage for Hepatitis-B, India has the highest infant death.

# ## Statistical way of the experiment: A/B Testing

# #### Hypothesis: 
# * Increasing Hepatitis-B immunization coverage among infants should decrease infant deaths.
# 
# #### Experiment: 
# * 5000 infants will be randomly selected from the total population with low income. Low income families should agree to anonymous data collection and ashould follow the guidelines in exchange for a nominal financial incentive.
# 
# #### Treatment: 
# * The organization will be offering immunization coverage for Hepatitis-B and infants will be under close observation for one year.

# #### The Success of mentioned experiment:
# * If exclusively immunized infant deaths is at least 10% less than infants without immunization coverage, the null hypothesis that Hepatitis-B vaccination has no impact on infant deaths can be rejected.

# <br><br><center><h1 style="font-size:2em;color:black">2018</h1></center>
# <br>
# <table>
# <col width="650">
# <col width="50">
# <tr>
# <td><img src="https://i.redd.it/k25bd3e1muh01.png
# " align="center" style="width:3950px;height:360px;"/></td>
# <td>
# </td>
# </tr>
# </table>
