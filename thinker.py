import numpy as np
import random
import pickle
from flask import Flask, render_template
from flask_cors import CORS
from itertools import combinations
import pymongo
import json

'''
A tictactoe bot that using qlearning approach for it's decision. all state
is encoded as a string from already taken move.

'''

#setting
qtable = {}
lrate = 0.2
wincondition_ori = [[1,2,3], [4,5,6], [7,8,9],
[1,4,7], [2,5,8], [3,6,9], [1,5,9], [3,5,7]]

wincondition = wincondition_ori.copy()

#4 combi
combi_4_tuple = combinations([1,2,3,4,5,6,7,8,9], 4)
combi_4_list = [list(ele) for ele in combi_4_tuple]

for i in range(len(wincondition_ori)):
	for j in range(len(combi_4_list)):
		if set(wincondition_ori[i]).issubset(combi_4_list[j]):
			wincondition.append(combi_4_list[j])

#5 combi
combi_5_tuple = combinations([1,2,3,4,5,6,7,8,9], 5)
combi_5_list = [list(ele) for ele in combi_5_tuple]

for i in range(len(wincondition_ori)):
	for j in range(len(combi_4_list)):
		if set(wincondition_ori[i]).issubset(combi_5_list[j]):
			wincondition.append(combi_5_list[j])


#[1,2,3,4], [1,2,3,5], [1,2,3,6], [1,2,3,7], [1,2,3,8], [1,2,3,9],
#[4,5,6,7], [4,5,6,8], [4,5,6,9], [1,4,5,6], [2,4,5,6], [3,4,5,6],
#[6,7,8,9], [5,7,8,9], [4,7,8,9], [3,7,8,9], [2,7,8,9], [1,7,8,9],
#[1,2,4,7], [1,3,4,7], [1,4,5,7], [1,4,6,7], [1,4,7,8], [1,4,7,9],
#[1,2,5,8], [2,3,5,8], [2,4,5,8], [2,5,6,8], [2,5,7,8], [2,5,8,9],
#[1,3,6,9], [2,3,6,9], [3,4,6,9], [3,5,6,9], [3,6,7,9], [3,6,8,9],
#[1,2,5,9], [1,3,5,9], [1,4,5,9], [1,5,6,9], [1,5,7,9], [1,5,8,9],
#[1,3,5,7], [2,3,5,7], [3,4,5,7], [3,5,6,7], [3,5,7,8], [3,5,7,9],

#[1,2,3,4,5], [1,2,3,4,6], [1,2,3,4,7], [1,2,3,4,8], [1,2,3,4,9], 
#[1,2,3,5,6], [1,2,3,5,7], [1,2,3,5,8], [1,2,3,5,9],
#[1,2,3,6,7], [1,2,3,6,8], [1,2,3,6,9],
#[1,2,3,7,8], [1,2,3,7,9],
#[1,2,3,8,9],

#]



winreward = 10.
drawreward = 5.
losereward = -10.
#epsilon = 1.0
#decay_epsilon = 0.005
explore_chance = 0.05
train_mode = False

#save all state and action as dict here
knowledge1_file = 'wisdom1.pkl'
knowledge2_file = 'wisdom2.pkl'

client = pymongo.MongoClient()
db_wisdom = client['rl_tictactoe']['wisdom']

def load_wisdom(wisdom_name):
	wisdom = {}
	all_knowledge = list(db_wisdom.find({'wisdom_name': wisdom_name}))
	for i in range(len(all_knowledge)):
		ext_state = all_knowledge[i]['state']
		ext_act = all_knowledge[i]['action']
		ext_q = all_knowledge[i]['q_value']
		wisdom[(ext_state, ext_act)] = ext_q
	return wisdom

def save_wisdom(list_state, list_action, list_q_value, wisdom_name):
	for iii in range(len(list_state)):
		db_wisdom.update({'wisdom_name': wisdom_name, 'state': list_state[iii], 
		'action': list_action[iii]}, {'$set' : {'q_value': list_q_value[iii]}}, upsert=True)	

def getQ(state, action, qtable): #get Q states
	if(qtable.get((state,action))) is None:
		return 0
	else:
		return qtable[(state,action)]

def monte_carlo_calc(state, action, reward, qtable):
	q_old = getQ(state, action, qtable)
	q_new = q_old + lrate * (reward - q_old)
	return q_new

def random_move(all_move):
	avail_move = list(np.arange(1,10))
	real_avail_move = []
	for ii in range(len(avail_move)):
		if avail_move[ii] not in all_move:
			real_avail_move.append(avail_move[ii])
	move = random.choice(real_avail_move)
	return str(move)

#assuming player move 1st
def first_personality(state_str, wisdom):
	#sanitize param
	if type(state_str) is not str:
		return 'invalid param', 'invalid param', wisdom
	
	try:
		int(state_str)
	except ValueError:
		return 'invalid param', 'invalid param', wisdom
	
	if '.' in state_str:
		return 'invalid param', 'invalid param', wisdom
	
	if len(state_str) > 9:
		return 'invalid param', 'invalid param', wisdom
	
	list_move = []
	for i in range(len(state_str)):
		list_move.append(int(state_str[i]))
	
	if len(list_move) != len(set(list_move)):
		return 'invalid param', 'invalid param', wisdom
	
	#end of sanitize
	
	condition = 'ongame'
	
	#dividing move to com and player
	#since player on the first move so
	#all even move is player the rest is com
	player_move = []
	com_move = []
	for i in range(len(list_move)):
		if i % 2:
			com_move.append(list_move[i])
		else:
			player_move.append(list_move[i])
	
	player_move_sorted = list(np.sort(player_move))
	com_move_sorted = list(np.sort(com_move))
	
	#first check if there is any winner in this state
	#if player win, we punish but if com win
	#we give bot reward
	print(player_move_sorted)
	if player_move_sorted in wincondition:
		next_move = ''
		condition = 'player win'
		#punish
		#extract all state and action of com
		all_state_action = []
		for i in range(len(com_move)):
			com_move_str = str(com_move[i])
			idx = state_str.index(com_move_str)
			com_state_str = state_str[:idx]
			all_state_action.append((com_state_str, com_move_str))
			print(com_state_str, com_move_str)
		
		#I believe the first move was not to blame for the losing
		all_q_value_saving = []
		if len(all_state_action) == 2:
			punishment = np.array([losereward*0, losereward*1,])
			for ii in range(len(all_state_action)):
				q_new = monte_carlo_calc(all_state_action[ii][0], all_state_action[ii][1], punishment[ii], wisdom)
				wisdom1[all_state_action[ii]] = q_new
				all_q_value_saving.append(q_new)
		elif len(all_state_action) == 3:
			punishment = np.array([losereward*0, losereward*0.3, losereward*0.7])
			for ii in range(len(all_state_action)):
				q_new = monte_carlo_calc(all_state_action[ii][0], all_state_action[ii][1], punishment[ii], wisdom)
				wisdom1[all_state_action[ii]] = q_new
				all_q_value_saving.append(q_new)
		elif len(all_state_action) == 4:
			punishment = np.array([losereward*0, losereward*0.1, losereward*0.7, losereward*0.2])
			for ii in range(len(all_state_action)):
				q_new = monte_carlo_calc(all_state_action[ii][0], all_state_action[ii][1], punishment[ii], wisdom)
				wisdom[all_state_action[ii]] = q_new
				all_q_value_saving.append(q_new)

		#save wisdom
		# Store data (serialize)
		all_state_saving = []
		all_action_saving = []
		for j in range(len(all_state_action)):
			all_state_saving.append(all_state_action[j][0])
			all_action_saving.append(all_state_action[j][1])
		save_wisdom(all_state_saving, all_action_saving, all_q_value_saving, 'wisdom1')
		


	elif com_move_sorted in wincondition:
		next_move = ''
		condition = 'player lose'
		#rewarding for win
		all_state_action = []
		for i in range(len(com_move)):
			com_move_str = str(com_move[i])
			idx = state_str.index(com_move_str)
			com_state_str = state_str[:idx]
			all_state_action.append((com_state_str, com_move_str))
			print(com_state_str, com_move_str)
		
		#I believe the first move was not to praise for the winning
		all_q_value_saving = []
		if len(all_state_action) == 2:
			reward = np.array([winreward*0, winreward*1,])
			for ii in range(len(all_state_action)):
				q_new = monte_carlo_calc(all_state_action[ii][0], all_state_action[ii][1], reward[ii], wisdom)
				wisdom1[all_state_action[ii]] = q_new
				all_q_value_saving.append(q_new)
		elif len(all_state_action) == 3:
			reward = np.array([winreward*0, winreward*0.3, winreward*0.7])
			for ii in range(len(all_state_action)):
				q_new = monte_carlo_calc(all_state_action[ii][0], all_state_action[ii][1], reward[ii], wisdom)
				wisdom1[all_state_action[ii]] = q_new
				all_q_value_saving.append(q_new)
		elif len(all_state_action) == 4:
			reward = np.array([winreward*0, winreward*0.1, winreward*0.7, winreward*0.2])
			for ii in range(len(all_state_action)):
				q_new = monte_carlo_calc(all_state_action[ii][0], all_state_action[ii][1], reward[ii], wisdom)
				wisdom[all_state_action[ii]] = q_new
				all_q_value_saving.append(q_new)
		
		#save wisdom
		# Store data (serialize)
		all_state_saving = []
		all_action_saving = []
		for j in range(len(all_state_action)):
			all_state_saving.append(all_state_action[j][0])
			all_action_saving.append(all_state_action[j][1])
		save_wisdom(all_state_saving, all_action_saving, all_q_value_saving, 'wisdom1')
	

	elif com_move_sorted not in wincondition and player_move_sorted not in wincondition and len(state_str) == 9:
		next_move = ''
		condition = 'draw'
		#rewarding for draw
		all_state_action = []
		all_q_value_saving = []
		for i in range(len(com_move)):
			com_move_str = str(com_move[i])
			idx = state_str.index(com_move_str)
			com_state_str = state_str[:idx]
			all_state_action.append((com_state_str, com_move_str))
			print(com_state_str, com_move_str)
		reward = np.array([drawreward*0, drawreward*0.2, drawreward*0.6, drawreward*0.2])
		for ii in range(len(all_state_action)):
				q_new = monte_carlo_calc(all_state_action[ii][0], all_state_action[ii][1], reward[ii], wisdom)
				wisdom[all_state_action[ii]] = q_new
				all_q_value_saving.append(q_new)
		
		#save wisdom
		# Store data (serialize)
		all_state_saving = []
		all_action_saving = []
		for j in range(len(all_state_action)):
			all_state_saving.append(all_state_action[j][0])
			all_action_saving.append(all_state_action[j][1])
		save_wisdom(all_state_saving, all_action_saving, all_q_value_saving, 'wisdom1')
		
		
	else:
		#still there is no winner, so continue the game and give move
		if train_mode:
			next_move = str(random_move(list_move))
		else:
			token = random.uniform(0,1)
			if token < explore_chance:
				next_move = random_move(list_move)
			else:
				avail_move = list(np.arange(1,10))
				real_avail_move = []
				for ii in range(len(avail_move)):
					if avail_move[ii] not in list_move:
						real_avail_move.append(avail_move[ii])
				avail_move = real_avail_move
				q_avail_move = []
				for ii in range(len(avail_move)):
					try:
						q_score = wisdom[(state_str, str(avail_move[ii]))]
					except KeyError:
						q_score = 0
					q_avail_move.append(q_score)
				idx_move = np.argmax(q_avail_move)
				next_move = str(avail_move[idx_move])
	
	#there is possibility that com win when he move in this moment
	future_state_str = state_str+next_move
	future_list_move = []
	for i in range(len(future_state_str)):
		future_list_move.append(int(future_state_str[i]))

	future_com_move = []
	for i in range(len(future_list_move)):
		if i % 2:
			future_com_move.append(future_list_move[i])
	
	future_com_move_sorted = list(np.sort(future_com_move))
	if future_com_move_sorted in wincondition:
		condition = 'player lose'
		#rewarding for win
		all_state_action = []
		all_q_value_saving = []
		for i in range(len(future_com_move)):
			com_move_str = str(future_com_move[i])
			idx = future_state_str.index(com_move_str)
			com_state_str = future_state_str[:idx]
			all_state_action.append((com_state_str, com_move_str))
			print(com_state_str, com_move_str)
		
		#I believe the first move was not to praise for the winning
		if len(all_state_action) == 2:
			reward = np.array([winreward*0, winreward*1,])
			for ii in range(len(all_state_action)):
				q_new = monte_carlo_calc(all_state_action[ii][0], all_state_action[ii][1], reward[ii], wisdom)
				wisdom1[all_state_action[ii]] = q_new
				all_q_value_saving.append(q_new)
		elif len(all_state_action) == 3:
			reward = np.array([winreward*0, winreward*0.3, winreward*0.7])
			for ii in range(len(all_state_action)):
				q_new = monte_carlo_calc(all_state_action[ii][0], all_state_action[ii][1], reward[ii], wisdom)
				wisdom1[all_state_action[ii]] = q_new
				all_q_value_saving.append(q_new)
		elif len(all_state_action) == 4:
			reward = np.array([winreward*0, winreward*0.1, winreward*0.7, winreward*0.2])
			for ii in range(len(all_state_action)):
				q_new = monte_carlo_calc(all_state_action[ii][0], all_state_action[ii][1], reward[ii], wisdom)
				wisdom[all_state_action[ii]] = q_new
				all_q_value_saving.append(q_new)
		
		#save wisdom
		# Store data (serialize)
		all_state_saving = []
		all_action_saving = []
		for j in range(len(all_state_action)):
			all_state_saving.append(all_state_action[j][0])
			all_action_saving.append(all_state_action[j][1])
		save_wisdom(all_state_saving, all_action_saving, all_q_value_saving, 'wisdom1')
	
	
	return next_move, condition, wisdom

			
#	elif com_move_sorted in wincondition:
		#reward

#assuming player move 2nd
#def second_personality():


#gain all wisdom1
#try:
#	wisdom1_open = open(knowledge1_file)
#	wisdom1 = pickle.load(wisdom1_open)
#if not exist, start from the fool
#except FileNotFoundError:
#	wisdom1 = {}

#gain all wisdom2
#try:
#	wisdom2_open = open(knowledge2_file)
#	wisdom2 = pickle.load(wisdom2_open)
#if not exist, start from the fool
#except FileNotFoundError:
#	wisdom2 = {}

wisdom1 = load_wisdom('wisdom1')
wisdom2 = load_wisdom('wisdom2')

app = Flask(__name__)
CORS(app)


@app.route('/think1/<position>')
def process(position):
	global wisdom1
	next_move, condition, wisdom1 = first_personality(position, wisdom1)
	answer = {}
	answer['condition'] = condition
	answer['next_move'] = next_move
	if next_move != 'invalid param':
		return json.dumps(answer)
	else:
		return 'invalid param'

#next_move, condition, wisdom1 = first_personality('15274', wisdom1)
#print(wisdom1, condition, next_move)
