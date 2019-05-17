#!/usr/bin/env python
# coding: utf-8

# In[3]:


class Party(object):

	def __init__(self, lyrics):
		self.lyrics = lyrics
		
	def sing_me_a_song(self):
		for line in self.lyrics:
			print (line)
			
candy_cando = Party(["Candy can do it",
				     "I don't know how far I can sing",
				     "So I guess I should stop here"])
				  
fireworks_beautiful = Party(["The entire family waching them",
						     "With pockets full of shells"])

candy_cando.sing_me_a_song()
fireworks_beautiful.sing_me_a_song() 


# In[ ]:




