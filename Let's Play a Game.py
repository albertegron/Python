#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sys import exit

def money_room():
	print ("This room is full of money. How much do you take?")
	
	forward = raw_input("> ")
	if "0" in forward or "1" in forward:
		how_much = int(forward)
	else:
		dead("Man, learn to type a number.")
		
	if how_much < 1000:
		print ("Nice, you're not greedy, you win!")
		exit(0)
	else:
		dead("You greedy bastard!")
	

def bear_room():
	print ("There is a bear here.")
	print ("The bear has a bunch of honey and is standing in front of another door.")
	print ("What's going to be your next step to move the bear?")
	print ("Press 1 to steal the honey or 2 to taunt the bear")
	bear_moved = False
	
	while True:
		forward = raw_input("> ")
		
		if forward == "1":
			dead("The bear gets pissed off and chews your leg off.")
		elif forward == "2" and not bear_moved:
			print ("The bear has mooved from the door. You can go through it now.")
			print ("In front of you there are two doors")
			print ("Press 1 to open first door or 2 to open the second door")
			bear_moved = True
		elif forward == "1" and bear_moved:
			dead("You fell inside the big black hole, you are dead.")
		elif forward == "2" and bear_moved:
			money_room()
		else:
			print ("I got no idea what that means.")

def ghost_room():
	print ("Here you see the great Ghost of Dookie.")
	print ("He, it, whatever stares at you will make you go crazy.")
	print ("Do you flee for your life (press 1) or eat your head (press 2)?")
	
	while True:
		forward = raw_input("> ")
	
		if "1" in forward:
			start()
		elif "2" in forward:
			dead("Well that was tasty!")
		else:
			ghost_room()
		
		
def candy_room():
	print ("Here you have entered into candy room and you see two black barrels")
	print ("Be careful, one barell has poisonous candy and the other moves you into next room")
	print ("Which one will you choose? (press 1 or 2)")
	
	forward = raw_input("> ")
	
	if "2" in forward:
		money_room()
	elif "1" in forward:
		dead("You choose the poisonous candy, you're dead!")
	else:
		ghost_room()
		
		
def dead(why):
	print why, ("Good job, game is over!")
	exit(0)
	
def start():
	print ("You are in dark room.")
	print ("There are 3 doors, to your Right (1), Middle (2) and to your Left (3).")
	print ("Which one do you choose?")
	
	while True:
		forward = raw_input("> ")
	
		if forward == "3":
			bear_room()
		elif forward == "1":
			ghost_room()
		elif forward == "2":
			candy_room()
		else:
			print ("I have no idea what that means.")
		
		
start()


# In[ ]:




