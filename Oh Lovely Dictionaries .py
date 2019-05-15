#!/usr/bin/env python
# coding: utf-8

# In[1]:


# creating a mapping of state to abbreviation
states = [
	'Oregon: OR',
	'Florida: FL',
	'California: CA',
	'New York: NY',
	'Michigan: MI'
]

# creating a basic set of states and some cities in them
cities = [
	'CA: San Francisco',
	'MI: Detroit',
	'FL: Jacksonville'
]

# adding some more cities
cities['NY'] = 'New York'
cities['OR'] = 'Portland'

# printing out some cities 
print ('-' * 10)
print ("NY State has: ", cities['NY'])
print ("OR State has: ", cities['OR'])

#printing some states 
print ('-' * 10)
print ("Michigan abbreviation is: ", cities['Michigan'])
print ("Florida abbreviation is: ", cities['Florida'])

# doing it by using the state then cities dict\
print ('-' * 10)
print ("Michigan has: ", cities[states['Michigan']])
print ("Florida has: ", cities[states['Florida']])

# printing every state abbreviation
print ('-' * 10)
for state, abbrev in states.item():
	print ("%s is abbreviated %s" % (state, abbrev))

#printing every city in state 
print ('-' * 10)
for abbrev, city in cities.items():
	print ("%s has the city %s" % (abbrev, city))

# Doing both at the same time
print ('-' * 10)
for state, abbrev in states.items():
	print ("%s state is abbreviated %s and has city %s" % (state, abbrev, cities[abbrev]))
	
print ('-' * 10)
# getting an abbreviation by state that might not be there 
state = states.get('Texas', None)

if not state:
	print ("Sorry, no Texas.")

# getting a city with a default value
city = cities.get('TX', 'Does Not Exist')
print ("The city for the state TX i s: %s" % city) 


# In[ ]:




