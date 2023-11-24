import os
import pandas as pd
from Demo import demo_S_plus_TSNE,demo_TSNE,get_info_as_numpy,appendToCSV
from time import time

class S_TSNE_Time(object):
	''' Runs the S+TSNE algorithm and accumulates the inference time for a given an iteration'''
	def __init__(self,
		batch,
		params_s_tsne,
		slice_size,
		random_state_slice,
		use_tqdm=True,
		):

		#parameters variables to execute the algo
		self.batch= batch
		self.params_s_tsne = params_s_tsne
		self.slice_size = slice_size
		self.random_state_slice = random_state_slice
		self.use_tqdm = use_tqdm

		#operacional variables, to accumulate results from iterations
		self.iteration = 0
		self.timers = {}


	def register_initial_points_number(self,x_initial,y_initial):
		self.initial_shift = len(list(y_initial))
		self.start_timer()
		pass

	def start_timer(self,*args,**kwargs):
		self.start = time()
		pass


	def register_time(self,*args,**kwargs):
		self.end = time()
		self.timers[self.iteration+self.initial_shift] = self.end-self.start
		pass


	def batch_time(self,*args,**kwargs):
		self.iteration += 1
		if (self.iteration % self.batch) == 0:
			self.register_time()
		pass


	
	def run(self,X, y, basePath):
		#Makes sure the directory for storing csv files exists
		os.makedirs(basePath,exist_ok=True)

		#run S+TSNE with the defined functions
		demo_S_plus_TSNE(X, y, self.params_s_tsne, self.slice_size, self.random_state_slice,
			collect_points=False,
			use_tqdm=self.use_tqdm,
			pre_init=self.register_initial_points_number, 
			pre_cycle=self.register_time,
			pre_iteration=self.start_timer,
			post_iteration=self.batch_time,
			pre_flush=self.start_timer,
			post_flush=self.register_time,
			post_cycle=None
		)

		#write csv with values
		path = os.path.join(basePath,"timer.csv")
		appendToCSV(path,self.timers)
		pass



class TSNE_Time(object):
	''' Runs the TSNE algorithm and accumulates inference time given an iteration'''

	def __init__(self, batch, params_tsne, slice_size, random_state_slice, use_tqdm):
		#parameters variables to execute the algo
		self.batch = batch
		self.params_tsne = params_tsne
		self.slice_size = slice_size
		self.random_state_slice = random_state_slice
		self.use_tqdm = use_tqdm
		
		#operacional variables, to accumulate results from iterations
		self.timers = {}

	
	def start_timer(self,*args,**kwargs):
		self.start = time()
		pass

	def register_time(self,X_acc,y_acc,embedding,iteration):
		self.end = time()
		self.timers[iteration] = self.end-self.start
		pass


	def run(self,X, y, basePath):
		#Makes sure the directory for storing csv files exists
		os.makedirs(basePath,exist_ok=True)

		#Runs the TSNE demo per batch
		demo_TSNE(X,y, self.batch, self.params_tsne, self.slice_size, self.random_state_slice, 
			use_tqdm=self.use_tqdm,
			pre_cycle=None,
			pre_iteration=self.start_timer,
			post_iteration=self.register_time,
			pre_flush=self.start_timer,
			post_flush=self.register_time,
			post_cycle=None
		)

		#write csv with values
		path = os.path.join(basePath,"timer.csv")
		appendToCSV(path,self.timers)
		pass