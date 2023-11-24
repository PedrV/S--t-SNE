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

		self.coordinatorECS = Coordinator()

		if self.density_min_points > self.max_batch_size:
			#print(self.density_min_points)
			#print(self.max_batch_size)
			raise Exception("Minimum Density Points should not be higher than the maximum batch size")


		if ("k_neighbors" in self.affinity_params 
			and self.affinity_params["k_neighbors"] != "auto" 
			and self.density_min_points < self.affinity_params["k_neighbors"]):
			raise Exception("Minimum Density Points should be higher than the affinities' KNN")

		self.affinity = affinity.PerplexityBasedNN(initial_points,**affinity_params)
		initialization = oTSNE.initialization.pca(initial_points,**initialization_params)
		embedding = oTSNE.TSNEEmbedding(initialization,self.affinity,**self.embedding_params)
		embedding.optimize(**initial_exageration_params, inplace=True)
		embedding.optimize(**initial_optimize_params, inplace=True)
		
		self.retain_points_demo = retain_points_demo
		self.prob_removal_demo = prob_removal_demo


		if self.retain_points_demo:
			self.all_embeds = embedding
			self.all_points = initial_points

		self.get_next_embed(embedding,initial_points)


	def get_next_embed(self,embedding,points):
		#if self.retain_points_demo:
		#	points_to_hull = self.all_embeds
		#else:
		#	points_to_hull = embedding
		
		points_to_hull = embedding
		self.coordinatorECS.new_batch(self.get_polys(points_to_hull))
		self.polygons = []
		for idV in self.coordinatorECS.convex_polygons:
			try:
				self.polygons.append(spatial.ConvexHull(self.coordinatorECS.convex_polygons[idV].convex_region))
			except:
				pass

		new_density_points = self.get_N_density_points(points,embedding,N=self.density_min_points)

		self.remove_density_points_random(self.prob_removal_demo)

		if self.density_points is not None:
			self.density_points = np.vstack([self.density_points,new_density_points[0]])
			self.density_points_emb = np.vstack([self.density_points_emb,new_density_points[1]])
		
		else:
			self.density_points = new_density_points[0]
			self.density_points_emb = new_density_points[1]

		self.check_density_points()
		self.affinity = affinity.PerplexityBasedNN(self.density_points,**self.affinity_params)
		self.embedding = oTSNE.TSNEEmbedding(self.density_points_emb,self.affinity,**self.embedding_params)
		pass


	def get_polys(self,points):
		clustering = AgglomerativeClustering(distance_threshold=self.hulls_threshold ,n_clusters=self.n_hulls)
		clustering.fit(points)
		indexes = []
		hulls = []
		for i in np.unique(clustering.labels_):
			indexes.append(np.where(clustering.labels_ == i)[0])

		for iis in indexes:
			pointsHull = np.take(points, iis, axis=0).reshape((-1,2))
			try:
				hull = spatial.ConvexHull(pointsHull)
				hulls.append(hull)
			except Exception as e:
				pass
		return hulls


	def flush_batch(self):
		newBatch = np.array(self.batch)

		affinity_prs = {}
		if "perplexity" in self.affinity_params:
			affinity_prs["perplexity"] = self.affinity_params["perplexity"]

		if "k_neighbors" in self.affinity_params:
			affinity_prs["k_neighbors"] = self.affinity_params["k_neighbors"]

		partialEmbed = self.embedding.prepare_partial(newBatch, initialization="median", k=self.affinity.n_samples, **affinity_prs)
		self.exageration_params["inplace"] = True
		self.optimize_params["inplace"] = True
		partialEmbed.optimize(**self.exageration_params)
		partialEmbed.optimize(**self.optimize_params)
		
		if self.retain_points_demo:
			self.all_points = np.vstack([self.all_points,newBatch])
			self.all_embeds = np.vstack([self.all_embeds,partialEmbed])
		self.get_next_embed(partialEmbed,newBatch)
		self.batch = []
		pass


	def remove_density_points_random(self,prob_removal):
		if self.density_points is not None:
			mask = np.random.choice([True, False], size=self.density_points.shape[0], p=[1-prob_removal, prob_removal])
			self.density_points = self.density_points[mask]
			self.density_points_emb = self.density_points_emb[mask]
			if not np.any(self.density_points):
				self.density_points = None
				self.density_points_emb = None
		pass


	def get_N_density_points(self,points,embeddings,N):
		groups = {i:[] for i in range(len(self.polygons))}
		inds = {i:[] for i in range(len(self.polygons))}
		for ind,(point,embedding) in enumerate(zip(points,embeddings)):
			proj = self.polygon_projection(embedding)
			if proj is not None:
				proj = proj[0]
				groups[proj].append(point)
				inds[proj].append(ind)

		n_per_hull = N//len(self.polygons)
		result = set()
		allPs = []

		groups = {i:groups[i] for i in groups if len(groups[i])>0}
		for groupI in groups:
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
					provisory.add(ind)
					result.add(inds[groupI][ind])
					accs.update(pts[ind])

			if len(provisory) < n_per_hull:
				dif = [inds[groupI][x] for x in sort_vals if x not in provisory][:(n_per_hull - len(provisory))]
				result.update(dif)

		if len(result) < N:
			tree = spatial.KDTree(points)
			density = tree.query_ball_point(points,self.density_search_radius)
			sort_vals = sorted(list(range(len(density))), key=lambda x: len(density[x]))
			dif = [x for x in sort_vals if x not in result][:(N - len(result))]
			result.update(dif)
		
		return points[list(result)],embeddings[list(result)]


	def check_density_points(self):
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
		for ind,polygon in enumerate(self.polygons):
			hull = spatial.Delaunay(polygon.points[polygon.vertices])
			if np.any(hull.find_simplex(projection)>=0):
				return ind,polygon
		return None


	def components(self):
		return self.polygons, self.density_points_emb


	def learn_one(self, x, sample_weight=None):
		self.batch.append(list(x.values()))
		if len(self.batch) >= self.max_batch_size:
			self.flush_batch()
		#x: dict
		#return self
		return self


	def predict_one(self, x, sample_weight=None):
		self.batch.append(list(x.values()))
		self.flush_batch()
		#x: dict
		#return assigned_cluster
		return self.components()