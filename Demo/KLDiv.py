import os
import pandas as pd
from Demo import demo_S_plus_TSNE,demo_TSNE,get_info_as_numpy,appendToCSV
from openTSNE import kl_divergence
from openTSNE import affinity


class S_TSNE_KLDiv(object):
	''' Runs the S+TSNE algorithm and accumulates the KL divergence for a given an iteration'''
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
		self.divergences = {}


	def register_initial_points_number(self,x_initial,y_initial,stsne=None):
		self.initial_shift = len(list(y_initial))
		pass


	def get_kl_s_tsne(self,stsne):
		affinities = affinity.PerplexityBasedNN(
		    stsne.all_points,
		    **self.params_s_tsne["affinity_params"]
		)
		self.divergences[self.iteration + self.initial_shift] = kl_divergence.kl_divergence_exact(affinities.P.toarray(),stsne.all_embeds)
		pass


	def batch_kl(self,x=None,y=None,stsne=None):
		if (self.iteration % self.batch) == 0:
			self.get_kl_s_tsne(stsne)
		pass


	def update(self,*args,**kwargs):
		self.iteration += 1
		pass

	
	def run(self,X, y, basePath):
		#Makes sure the directory for storing csv files exists
		os.makedirs(basePath,exist_ok=True)

		#run S+TSNE with the defined functions
		demo_S_plus_TSNE(X, y, self.params_s_tsne, self.slice_size, self.random_state_slice,
			collect_points=True,
			use_tqdm=self.use_tqdm,
			pre_init=None, 
			pre_cycle=self.register_initial_points_number,
			pre_iteration=self.batch_kl,
			post_iteration=self.update,
			pre_flush=None,
			post_flush=self.get_kl_s_tsne,
			post_cycle=None
		)

		#write csv with values
		path = os.path.join(basePath,"KL_div.csv")
		appendToCSV(path,self.divergences)
		pass




class TSNE_KLDiv(object):
	''' Runs the TSNE algorithm and accumulates the KL divergence for a given an iteration'''

	def __init__(self, batch, params_tsne, slice_size, random_state_slice, use_tqdm):
		#parameters variables to execute the algo
		self.batch = batch
		self.params_tsne = params_tsne
		self.slice_size = slice_size
		self.random_state_slice = random_state_slice
		self.use_tqdm = use_tqdm
		
		#operacional variables, to accumulate results from iterations
		self.divergences = {}

	
	def get_kl_tsne(self,X_acc,y_acc,embedding,iteration):
		self.divergences[iteration] = kl_divergence.kl_divergence_exact(embedding.affinities.P.toarray(),embedding)
		pass 


	def run(self,X, y, basePath):
		#Makes sure the directory for storing csv files exists
		os.makedirs(basePath,exist_ok=True)

		#Runs the TSNE demo per batch
		demo_TSNE(X,y, self.batch, self.params_tsne, self.slice_size, self.random_state_slice, 
			use_tqdm=self.use_tqdm,
			pre_cycle=None,
			pre_iteration=None,
			post_iteration=self.get_kl_tsne,
			pre_flush=None,
			post_flush=self.get_kl_tsne,
			post_cycle=None
		)

		#write csv with values
		path = os.path.join(basePath,"KL_div.csv")
		appendToCSV(path,self.divergences)
		pass