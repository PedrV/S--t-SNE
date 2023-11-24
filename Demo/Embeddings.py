import os
import numpy as np
import pandas as pd
from Demo import demo_S_plus_TSNE,demo_TSNE,get_info_as_numpy




class S_TSNE_Embedding(object):
	''' Runs the S+TSNE algorithm and accumulates the visual components of it for further analysis'''

	def __init__(self,
		batch,
		params_s_tsne,
		slice_size,
		random_state_slice,
		use_tqdm=True,
		collect_density_classes=False,
		collect_embeds=True
		):
		#parameters variables to execute the algo
		self.batch= batch
		self.params_s_tsne = params_s_tsne
		self.slice_size = slice_size
		self.random_state_slice = random_state_slice
		self.use_tqdm = use_tqdm
		
		#operacional variables, to accumulate results from iterations
		self.iteration = 0
		self.collect_density_classes = collect_density_classes
		self.collect_embeds = collect_embeds
		self.collect_classes = collect_density_classes or collect_embeds


	def initial_collection(self,x_initial=None,y_initial=None,stsne=None):
		#collect classes and iterations for future dataframe and plots
		self.classes = list(y_initial) if self.collect_classes else None
		self.initial_shift = len(self.classes)
		self.iteration_reference = [self.initial_shift for _ in self.classes]
	
		# initailize the component collecton
		self.allHulls = None
		self.allDense = None
		self.embeddings = None
		self.iteration = 0
		self.batch_reference = 0
		self.counter = 0
		pass


	def full_collection(self,stsne):
		#function imported from Demo.py for extracting hulls and dense points from the stsne class
		allHulls,allDense = get_info_as_numpy(
			stsne,
			self.classes,
			self.iteration + self.initial_shift,
			self.allHulls,
			self.allDense,
			self.collect_density_classes
		)

		#update collection dictionary
		self.allHulls = allHulls
		self.allDense = allDense
		self.iteration_reference += [(self.iteration + self.initial_shift) for _ in range(self.counter)]
		self.counter = 0
		pass

	def batch_collection(self,x=None,y=None,stsne=None):
		# If the iterations reach the batch number registered in the collection dictionary,
		# the dictionary is updated with new hulls and density points
		if (self.iteration % self.batch) == 0:
			# Extract new hulls and density points
			self.full_collection(stsne)	
		pass

	def collection_update(self,x=None,y=None,stsne=None):
		# append classes and iteration references to the collection dictionary
		# update the number of iteractions
		if self.collect_classes:
			self.classes.append(y)
		self.iteration += 1
		self.counter += 1
		pass

	def collect_embeddings(self,stsne):
		# Collects the embeddings from the S+TSNE class if requested initially		
		if self.collect_embeds:
			self.embeddings = np.hstack([
				np.expand_dims(self.iteration_reference, axis=1),
				stsne.all_embeds,
				np.expand_dims(self.classes, axis=1)
			])
		pass

	def run(self,X, y, basePath):
		#Makes sure the directory for storing csv files exists
		os.makedirs(basePath,exist_ok=True)

		#run S+TSNE with the defined functions
		demo_S_plus_TSNE(X, y, self.params_s_tsne, self.slice_size, self.random_state_slice,
			collect_points=self.collect_classes,
			use_tqdm=self.use_tqdm,
			pre_init=None,
			pre_cycle=self.initial_collection,
			pre_iteration=self.batch_collection,
			post_iteration=self.collection_update,
			pre_flush=None,
			post_flush=self.full_collection,
			post_cycle=self.collect_embeddings
		)

		# Converts numpy arrays to Panda DataFrames and writes to disk on "basePath"
		if self.embeddings is not None:
			pd.DataFrame(self.embeddings,columns=["Iteration","X","Y","Class"]).to_csv(os.path.join(basePath,"embeddings.csv"),index=False)
		
		pd.DataFrame(self.allDense,columns=["Iteration","X","Y","Class"]).to_csv(os.path.join(basePath,"density_points.csv"),index=False)
		pd.DataFrame(self.allHulls,columns=["Iteration","X","Y","Hull"]).to_csv(os.path.join(basePath,"hullVertices.csv"),index=False)
		pass






class TSNE_embedding(object):
	''' Runs the TSNE algorithm and accumulates the visual components of it for further analysis'''

	def __init__(self, batch, params_tsne, slice_size, random_state_slice, use_tqdm):
		#parameters variables to execute the algo
		self.batch = batch
		self.params_tsne = params_tsne
		self.slice_size = slice_size
		self.random_state_slice = random_state_slice
		self.use_tqdm = use_tqdm
		
		#operacional variables, to accumulate results from iterations
		self.all_embeds = None


	def tsne_embeddings(self,X_acc,y_acc,embedding,iteration):
		#Collects TSNE embeddings of all points after the execution of the algo
		#each iteration this function stacks embeddings or initialize the stacking
		if self.all_embeds is not None:
			self.all_embeds = np.vstack([
				self.all_embeds,
				np.hstack([
					np.expand_dims([iteration for _ in y_acc],axis=1),
					embedding,
					np.expand_dims(y_acc,axis=1),
				])
			])
		
		else:
			self.all_embeds = np.hstack([
				np.expand_dims([iteration for _ in y_acc],axis=1),
				embedding,
				np.expand_dims(y_acc,axis=1)
			])
		pass


	def run(self,X, y, basePath):
		#Makes sure the directory for storing csv files exists
		os.makedirs(basePath,exist_ok=True)

		#Runs the TSNE demo per batch
		demo_TSNE(X,y, self.batch, self.params_tsne, self.slice_size, self.random_state_slice, 
			use_tqdm=self.use_tqdm,
			pre_cycle=None,
			pre_iteration=None,
			post_iteration=self.tsne_embeddings,
			pre_flush=None,
			post_flush=self.tsne_embeddings,
			post_cycle=None
		)

		#writes result to csv file
		pd.DataFrame(self.all_embeds,columns=["Iteration","X","Y","Class"]).to_csv(os.path.join(basePath,"embeddings.csv"),index=False)
		pass




