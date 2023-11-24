import os
import pandas as pd
from Demo import demo_S_plus_TSNE,demo_TSNE,get_info_as_numpy,appendToCSV
import tracemalloc


class S_TSNE_Memory(object):
	''' Runs the S+TSNE algorithm and accumulates both memory usage and peak given an iteration'''
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
		self.usage = {}
		self.peak = {}
		pass

	
	def register_initial_points_number(self,x_initial,y_initial,stsne=None):
		self.initial_shift = len(list(y_initial))
		self.start_collection()
		pass


	def start_collection(self,*args,**kwargs):
		tracemalloc.start()
		self.usage[0],self.peak[0] = tracemalloc.get_traced_memory()
		tracemalloc.reset_peak()
		pass


	def collect_memory(self,*args,**kwargs):
		self.usage[self.iteration+self.initial_shift],self.peak[self.iteration+self.initial_shift] = tracemalloc.get_traced_memory()
		tracemalloc.reset_peak()
		pass

	def batch_memory(self,*args,**kwargs):
		self.iteration += 1
		if (self.iteration % self.batch) == 0:
			self.collect_memory()
		pass


	
	def run(self,X, y, basePath):
		#Makes sure the directory for storing csv files exists
		os.makedirs(basePath,exist_ok=True)

		#run S+TSNE with the defined functions
		demo_S_plus_TSNE(X, y, self.params_s_tsne, self.slice_size, self.random_state_slice,
			collect_points=False,
			use_tqdm=self.use_tqdm,
			pre_init=self.register_initial_points_number,
			pre_cycle=self.collect_memory,
			pre_iteration=None,
			post_iteration=self.batch_memory,
			pre_flush=None,
			post_flush=self.collect_memory,
			post_cycle=None
		)

		#write csv with values
		path = os.path.join(basePath,"memoryUsage.csv")
		appendToCSV(path,self.usage)
		
		path = os.path.join(basePath,"memoryPeak.csv")
		appendToCSV(path,self.peak)
		pass





class TSNE_Memory(object):
	''' Runs the TSNE algorithm and accumulates both memory usage and peak given an iteration'''

	def __init__(self, batch, params_tsne, slice_size, random_state_slice, use_tqdm):
		#parameters variables to execute the algo
		self.batch = batch
		self.params_tsne = params_tsne
		self.slice_size = slice_size
		self.random_state_slice = random_state_slice
		self.use_tqdm = use_tqdm
		
		#operacional variables, to accumulate results from iterations
		self.usage = {}
		self.peak = {}

	
	def start_collection(self,*args,**kwargs):
		tracemalloc.start()
		self.usage[0],self.peak[0] = tracemalloc.get_traced_memory()
		tracemalloc.reset_peak()
		pass


	def collect_memory(self,X_acc,y_acc,embedding,iteration):
		self.usage[iteration],self.peak[iteration] = tracemalloc.get_traced_memory()
		tracemalloc.reset_peak()
		pass


	def run(self,X, y, basePath):
		#Makes sure the directory for storing csv files exists
		os.makedirs(basePath,exist_ok=True)

		#Runs the TSNE demo per batch
		demo_TSNE(X,y, self.batch, self.params_tsne, self.slice_size, self.random_state_slice, 
			use_tqdm=self.use_tqdm,
			pre_cycle=self.start_collection,
			pre_iteration=None,
			post_iteration=self.collect_memory,
			pre_flush=None,
			post_flush=self.collect_memory,
			post_cycle=None
		)

		#write csv with values
		path = os.path.join(basePath,"memoryUsage.csv")
		appendToCSV(path,self.usage)
		
		path = os.path.join(basePath,"memoryPeak.csv")
		appendToCSV(path,self.peak)
		pass



