#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Using Class to define the behaviour of an object
# And the kind of info an object can store
# The info in class is stored in attributes 
# The functions that belong to a class are called methods
# A child class inherits the attributes and methods from its parent class

class Dog():
    """Represent a dog."""
    
    def __init__(self, name):
        """Initialize dog object."""
        self.name = name
        
    def sit(self):
        """Simulate sitting."""
        print(self.name + " is sitting.")
        
my_dog = Dog('Rex')

print(my_dog.name + " is a great dog!")
my_dog.sit()


# In[ ]:




