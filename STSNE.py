from river import base
import openTSNE as oTSNE
from openTSNE import affinity
import scipy.sparse as sparse
import scipy.spatial as spatial
from sklearn.cluster import AgglomerativeClustering
from ecs.coordinator import Coordinator
import numpy as np
import random


__all__ = ["STSNE"]



class STSNE(base.Clusterer):
	'''
	S+TSNE class created based on river package API (https://riverml.xyz/)
	for more detail at this class check the full functionality of S+t-SNE at https://github.com/PedrV/S--t-SNE
	
	parameters:
		initial_points: np.ndarray -> initial points used in the first step of embedding;

	Key parameters:
		max_batch_size: int -> Defined size of batch before a new embedding is incremented to the existing one;

		Clustering Parameters:
			This Algorithm implements Agglomerative Clustering for convex hulls, the following parameters are exposed;
			
			hulls_threshold: int -> maximum distance between hulls before a new cluster is considered a hull;

			n_hulls: int -> specify apriori the number of hulls to be described at the projection (None for any number);

		
		Density Search parameters:
			This set of parameters define the search of the density points (PEDRUL) at the original N-D representation,
			this search is made with a KDTree.

			density_search_radius: int -> Distance radius considered in the search of the neighbourhood of a given point
			to define its "density";

			density_min_points: int -> minimum of denisty points to be kept from each iteration;

			density_max_points: int -> maximum of denisty points to be kept from each iteration;


		openTSNE Embedding parameters:
			This class bases its embedding funcionalities from the package openTSNE (https://github.com/pavlin-policar/openTSNE)
			for more details access (https://opentsne.readthedocs.io/en/stable/).

			affinity_params: dictionary -> key parameters defined for PerplexityBasedNN from openTSNE;

			initialization_params: dictionary -> key parameters defined for initialization.pca from openTSNE;

			embedding_params: dictionary -> key parameters defined for TSNEEmbedding from openTSNE;

			exageration_params: dictionary -> key parameters defined for TSNEEmbedding.optimize, 
			this parameters will be used in a inicial optmize step for exageration, for more details check 
			(https://opentsne.readthedocs.io/en/stable/examples/02_advanced_usage/02_advanced_usage.html);

			optimize_params: dictionary -> key parameters defined for TSNEEmbedding.optimize used as a commom optmization step;

			initial_exageration_params: dictionary -> key parameters defined for TSNEEmbedding.optimize for exageration, 
			these parameters will only be used in the initial embedding of the data
			
			initial_optimize_params: dictionary -> key parameters defined for TSNEEmbedding.optimize for optimization, 
			these parameters will only be used in the initial embedding of the data


		DEMO parameters:
			This set of parameters is only recomendded for deomnstrations of the working algorith.

			retain_points_demo: Boolean -> stores in memory every 2D projection made by the object in its lifetime;

			prob_removal_demo: float -> probability of random removal of density points at each batch flush;

		
		Important instance variables:
			self.coordinatorECS: ecs.coordinator.Coordinator -> Responsible for the Exponential Cobweb Slicing (ECS) and its respective parameters

	'''
	def __init__(self,initial_points,
				hulls_threshold=100,
				n_hulls=None,
				max_batch_size=200,
				density_search_radius=20,
				density_min_points=30,
				density_max_points=50,
				affinity_params={"perplexity":20,
								 "metric":"euclidean",
								 "n_jobs":-1,
								 "random_state":42,
								 "verbose":True,},
				initialization_params={"random_state":42},
				embedding_params={"negative_gradient_method":"bh",
								  "n_jobs":8,
								  "random_state":42,
								  "verbose":False},

				exageration_params={
					"n_iter":12,"exaggeration":2,"momentum":0.5,
					"learning_rate":0.1,"max_grad_norm":0.25,"max_step_norm":None
				},
				optimize_params={
					"n_iter":250,"momentum":0.9, "exaggeration":1,
					"learning_rate":0.1,"max_grad_norm":0.25,"max_step_norm":None
				},				
				initial_exageration_params={
					"n_iter":250, "exaggeration":15, "momentum":0.5
				},
				initial_optimize_params={
					"n_iter":500, "momentum":0.8
				},
				retain_points_demo=False,
				prob_removal_demo=0,
				):

		super().__init__()
		self.affinity_params = affinity_params
		self.optimize_params = optimize_params
		self.exageration_params = exageration_params
		
		self.hulls_threshold = hulls_threshold
		self.n_hulls = n_hulls

		self.embedding_params = embedding_params
		self.batch = []
		self.max_batch_size = max_batch_size

		self.density_points = None
		self.density_points_emb = None
		self.density_search_radius = density_search_radius
		self.density_min_points = density_min_points
		self.density_max_points = density_max_points

		self.retain_points_demo = retain_points_demo
		self.prob_removal_demo = prob_removal_demo

		
		#initializes the coordinator for ECS
		self.coordinatorECS = Coordinator()

		if self.density_min_points > self.max_batch_size:
			raise Exception("Minimum Density Points should not be higher than the maximum batch size")

		if ("k_neighbors" in self.affinity_params 
			and self.affinity_params["k_neighbors"] != "auto" 
			and self.density_min_points < self.affinity_params["k_neighbors"]):
			raise Exception("Minimum Density Points should be higher than the affinities' KNN")

		
		#Embedding of the initial points using the original t-SNE algorithm and the initial_* parameters
		self.affinity = affinity.PerplexityBasedNN(initial_points,**affinity_params)
		initialization = oTSNE.initialization.pca(initial_points,**initialization_params)
		embedding = oTSNE.TSNEEmbedding(initialization,self.affinity,**self.embedding_params)
		embedding.optimize(**initial_exageration_params, inplace=True)
		embedding.optimize(**initial_optimize_params, inplace=True)
		

		if self.retain_points_demo:
			self.all_embeds = embedding
			self.all_points = initial_points

		self.get_next_embed(embedding,initial_points)


	def get_next_embed(self,embedding,points):
		'''
		Define the next embedding object for further expantion based on the selected density points
		Recalculates the Hulls for the projection using the coordinator
		'''	
		points_to_hull = embedding
		self.coordinatorECS.new_batch(self.get_polys(points_to_hull))
		self.polygons = []
		for idV in self.coordinatorECS.convex_polygons:
			try:
				# Check for the hull update at the Coordinator
				self.polygons.append(spatial.ConvexHull(self.coordinatorECS.convex_polygons[idV].convex_region))
			except:
				pass

		#Updates Density Points based on the new hulls
		new_density_points = self.get_N_density_points(points,embedding,N=self.density_min_points)
		self.remove_density_points_random(self.prob_removal_demo)
		
		if self.density_points is not None:
			self.density_points = np.vstack([self.density_points,new_density_points[0]])
			self.density_points_emb = np.vstack([self.density_points_emb,new_density_points[1]])
		
		else:
			#Update internal variable of Density Points
			self.density_points = new_density_points[0]
			self.density_points_emb = new_density_points[1]

		#Check if all Density Points are projected at any Hull
		self.check_density_points()
		#Creates next embed
		self.affinity = affinity.PerplexityBasedNN(self.density_points,**self.affinity_params)
		self.embedding = oTSNE.TSNEEmbedding(self.density_points_emb,self.affinity,**self.embedding_params)
		pass


	def get_polys(self,points):
		'''
		Define the new Hulls at the projection for upadte at the ECS Coordinator
		'''
		#Performs Agglomerative Clustering for Hull definition
		clustering = AgglomerativeClustering(distance_threshold=self.hulls_threshold ,n_clusters=self.n_hulls)
		clustering.fit(points)
		
		indexes = []
		hulls = []
		for i in np.unique(clustering.labels_):
			indexes.append(np.where(clustering.labels_ == i)[0])

		for iis in indexes:
			pointsHull = np.take(points, iis, axis=0).reshape((-1,2))
			try:
				#Calculate Hulls using scipy.spatial.ConvexHull 
				#(https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html)
				hull = spatial.ConvexHull(pointsHull)
				hulls.append(hull)
			except Exception as e:
				pass
		return hulls


	def flush_batch(self):
		'''
		Flushes Batch, embedding the new points and updating the Hulls
		'''
		newBatch = np.array(self.batch)

		affinity_prs = {}
		if "perplexity" in self.affinity_params:
			affinity_prs["perplexity"] = self.affinity_params["perplexity"]

		if "k_neighbors" in self.affinity_params:
			affinity_prs["k_neighbors"] = self.affinity_params["k_neighbors"]

		
		#embed new points using simply the old embedding of the last iteration's Density Points
		partialEmbed = self.embedding.prepare_partial(newBatch, initialization="median", k=self.affinity.n_samples, **affinity_prs)
		self.exageration_params["inplace"] = True
		self.optimize_params["inplace"] = True
		partialEmbed.optimize(**self.exageration_params)
		partialEmbed.optimize(**self.optimize_params)
		
		if self.retain_points_demo:
			self.all_points = np.vstack([self.all_points,newBatch])
			self.all_embeds = np.vstack([self.all_embeds,partialEmbed])
		
		#updates the embedding and resets the batch
		self.get_next_embed(partialEmbed,newBatch)
		self.batch = []
		pass


	def remove_density_points_random(self,prob_removal):
		'''
		Removes Density Points at random based on a probability
		'''
		if self.density_points is not None:
			mask = np.random.choice([True, False], size=self.density_points.shape[0], p=[1-prob_removal, prob_removal])
			self.density_points = self.density_points[mask]
			self.density_points_emb = self.density_points_emb[mask]
			if not np.any(self.density_points):
				self.density_points = None
				self.density_points_emb = None
		pass


	def get_N_density_points(self,points,embeddings,N):
		'''
		Searches in the K-D space for the denser points to be used as anchors in the next interactions,
		this method specifically searches for the N most dense points considering non overlapping neighborhoods
		'''
		
		#check for the denser points at each Hull
		groups = {i:[] for i in range(len(self.polygons))}
		inds = {i:[] for i in range(len(self.polygons))}
		for ind,(point,embedding) in enumerate(zip(points,embeddings)):
			proj = self.polygon_projection(embedding)
			if proj is not None:
				proj = proj[0]
				groups[proj].append(point)
				inds[proj].append(ind)

		
		#Defines how many denisty points should be available per hull
		n_per_hull = N//len(self.polygons)
		result = set()
		allPs = []

		groups = {i:groups[i] for i in groups if len(groups[i])>0}
		for groupI in groups:
			#Querries the Neighborhood of each point, anotates their density and their neighbors
			group = np.array(groups[groupI])
			accs = set()
			provisory = set()
			tree = spatial.KDTree(group)
			density = tree.query_ball_point(group,self.density_search_radius)
			pts = {i:set(x) for i,x in enumerate(density)}
			sort_vals = sorted([i for i in pts], key=lambda x: len(pts[x]),reverse=True)
			for ind in sort_vals:
				if len(provisory) >= n_per_hull:
					break
				
				if ind not in accs and not pts[ind].issubset(accs):
					#Verifies transitivity discounting Density points with closed neighbourhoods
					provisory.add(ind)
					result.add(inds[groupI][ind])
					accs.update(pts[ind])

			if len(provisory) < n_per_hull:
				#Completes the density points of the hulls with remaining denser points till the minimum threshold is stablished
				dif = [inds[groupI][x] for x in sort_vals if x not in provisory][:(n_per_hull - len(provisory))]
				result.update(dif)

		if len(result) < N:
			#Completes the density points with remaining denser points till the total threshold is stablished
			tree = spatial.KDTree(points)
			density = tree.query_ball_point(points,self.density_search_radius)
			sort_vals = sorted(list(range(len(density))), key=lambda x: len(density[x]))
			dif = [x for x in sort_vals if x not in result][:(N - len(result))]
			result.update(dif)
		
		return points[list(result)],embeddings[list(result)]


	def check_density_points(self):
		'''
		Updates the Density points from the embedding, 
		checking if all the selected density points are contained in a ConvexHull
		'''
		selected = []
		selected_emb = []
		for emb,point in zip(self.density_points_emb,self.density_points):
			if len(selected) > self.density_max_points:
				break

			if self.polygon_projection(emb) is not None:
				selected.append(point)
				selected_emb.append(emb)

			random_number = random.random()

		self.density_points = np.array(selected)
		self.density_points_emb = np.array(selected_emb)
		return self.density_points,self.density_points_emb


	def polygon_projection(self,projection):
		'''
		Given a 2D projection this method returns which Convex Hull contains the projection, 
		if no Hull contains it None is given
		'''
		for ind,polygon in enumerate(self.polygons):
			hull = spatial.Delaunay(polygon.points[polygon.vertices])
			if np.any(hull.find_simplex(projection)>=0):
				return ind,polygon
		return None


	def components(self):
		'''
		Returns both the convexHull objects (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html)
		and the density points (np.ndarray) embeddings for plotting
		'''
		return self.polygons, self.density_points_emb


	def learn_one(self, x, sample_weight=None):
		#Follows River API where x is a dictionary of values to be learned
		#if the batchSize is matched, the batch is flushed
		self.batch.append(list(x.values()))
		if len(self.batch) >= self.max_batch_size:
			self.flush_batch()
		return self


	def predict_one(self, x, sample_weight=None):
		#Follows River API where x is a dictionary of values to be predicted
		#this method forces the batch flush independently of its size
		#this method returns the components of the embedding
		self.batch.append(list(x.values()))
		self.flush_batch()
		return self.components()