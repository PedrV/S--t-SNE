import os
import sys
from river import stream
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from openTSNE import TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization
from openTSNE import kl_divergence

import sys
sys.path.append('../')
from STSNE import STSNE

###% S+TSNE %###



def demo_S_plus_TSNE(X, y, params_s_tsne, slice_size, random_state_slice,
		collect_points=False,
		use_tqdm=True,
		pre_init=None,
		pre_cycle=None,
		pre_iteration=None,
		post_iteration=None,
		pre_flush=None,
		post_flush=None,
		post_cycle=None
	):

	#split DataSet for initial split and iteration split
	x_iter, x_initial, y_iter, y_initial = train_test_split(X, y, test_size=slice_size, random_state=random_state_slice)

	#run pre_init function if defined
	if pre_init is not None:
		pre_init(x_initial,y_initial)
	
	#initialize S+Tsne
	stsne = STSNE(x_initial,retain_points_demo=collect_points,**params_s_tsne)


	#run pre_cycle function if defined
	if pre_cycle is not None:
		pre_cycle(x_initial,y_initial,stsne=stsne)

	#prepare iterator
	iterator = stream.iter_array(x_iter,y_iter) 
	iterator = tqdm(iterator) if use_tqdm else iterator

	#iterate over dataSet running pre and post iteration function if defined
	for x, y in iterator:
		if pre_iteration is not None:
			pre_iteration(x,y,stsne=stsne)
		
		#run S+Tsne
		stsne.learn_one(x)
		
		if post_iteration is not None:
			post_iteration(x,y,stsne=stsne)

	
	#flush remaining points, as an iteration
	if pre_flush is not None:
		pre_flush(stsne=stsne)
	
	stsne.flush_batch()

	if post_flush is not None:
		post_flush(stsne=stsne)

	#run post_cycle function if defined	
	if post_cycle is not None:
		post_cycle(stsne=stsne)



###% Vanilla TSNE %###



def demo_TSNE(X,y, batch, params_tsne, slice_size, random_state_slice, 
		use_tqdm=True,
		pre_cycle=None,
		pre_iteration=None,
		post_iteration=None,
		pre_flush=None,
		post_flush=None,
		post_cycle=None
	):
	
	#split DataSet for initial split and iteration split
	x_iter, x_initial, y_iter, y_initial = train_test_split(X, y, test_size=slice_size, random_state=random_state_slice)

	#accumulates points
	X_acc = x_initial
	y_acc = y_initial
	
	#run pre_cycle function if defined
	if pre_cycle is not None:
		pre_cycle(x_initial,y_initial)

	#prepare iterator
	iterator = stream.iter_array(x_iter,y_iter) 
	iterator = tqdm(iterator) if use_tqdm else iterator
	iteration = 0

	#iterate over dataSet running pre and post iteration function if defined
	for x,y in iterator:

		#register iteration value for a given batch
		if iteration % batch == 0:
			if pre_iteration is not None:
				pre_iteration(X_acc,y_acc,iteration + len(y_initial))

			embedding = apply_TSNE(X_acc,params_tsne)
			
			if post_iteration is not None:
				post_iteration(X_acc,y_acc,embedding,iteration + len(list(y_initial)))
		
		#stack values for the next iteration
		new_point = np.array(list(x.values()))
		X_acc = np.vstack([X_acc,new_point])
		y_acc = np.hstack([y_acc,y])
		iteration += 1

	#Flush remaining points on last iteration
	if pre_flush is not None:
		pre_flush(X_acc,y_acc,iteration+len(list(y_initial)))

	embedding = apply_TSNE(X_acc,params_tsne)
	
	if post_flush is not None:
		post_flush(X_acc,y_acc,embedding,iteration+ len(list(y_initial)))

	#run post_cycle function if defined	
	if post_cycle is not None:
		post_cycle(X_acc,y_acc)



def apply_TSNE(X,params_tsne):
	#initialize affinities and embeddings using parameters available on TestParams.py dictionary
	random_state = params_tsne["random_state"]
	affinities_tsne = affinity.PerplexityBasedNN(X,random_state=random_state,**params_tsne["affinities_params"])
	init_tsne = initialization.pca(X, random_state=random_state)
	embedding = TSNEEmbedding(init_tsne,affinities_tsne,random_state=random_state,**params_tsne["embedding_params"])

	#run optimization process with available parameters
	embedding_ = embedding.optimize(**params_tsne["optmizations"][0])
	for optmization_parameter in params_tsne["optmizations"][1:]:
		embedding_ = embedding.optimize(**optmization_parameter)

	return embedding_



###########% Auxiliary Function %#################

#Component Extraction


def get_hulls_as_numpy(hulls, iteration):
	#extract hulls as numpy array of vertices given the hull component of the S+TSNE
	hullsNP = None
	for i,hull in enumerate(hulls):
		#each hull vertex is anotated with its hull and the iteration at it belongs
		aux = np.hstack([
			np.expand_dims([iteration for _ in hull.vertices],axis=1),
			hull.points[hull.vertices],
			np.expand_dims([i for _ in hull.vertices],axis=1)
		])
		if hullsNP is not None:
			hullsNP = np.vstack([hullsNP,aux])
		else:
			hullsNP = aux
	return hullsNP



def get_density_as_numpy(density_embs, allPoints_embs, classes, iteration,collect_dense_class):
	#extract density points as numpy array, with respective classes if provided so
	if collect_dense_class:
		#gets the index of the density point based on its embed
		#given that tsne may not be injective, we will take care of duplicates by opting for the last appearence of a given point
		tuple_points = np.flip(allPoints_embs.view(",".join(["float64" for _ in range(2)])).reshape(-1)) 
		tuple_dense = np.flip(density_embs.view(",".join(["float64" for _ in range(2)])).reshape(-1))
		test,indexes_unique = np.unique(tuple_points,return_index=True)
		tuple_points = tuple_points[np.sort(indexes_unique)]

		#gets the class of embeded density points based on its index extracted
		array_classes = np.array(classes)[np.sort(indexes_unique)]
		dense_inds = np.nonzero(np.in1d(tuple_points,tuple_dense))[0][-len(tuple_dense):]
		clasDens = array_classes[dense_inds]
	
	else:
		#if the test should not collect the density classes, the deafult class will be 0
		clasDens = [0 for _ in density_embs]
	return np.hstack([np.expand_dims([iteration for _ in density_embs],axis=1),density_embs,np.expand_dims(clasDens,axis=1)])



def get_info_as_numpy(stsne, classes, iteration, arrayHulls, arrayDense, collect_dense_class):
	#extract both density and hulls in numpy format given an S+TSNE object
	aux_hulls = get_hulls_as_numpy(stsne.components()[0],iteration)
	aux_dense = get_density_as_numpy(stsne.density_points_emb, stsne.all_embeds,classes, iteration, collect_dense_class)
	
	if arrayHulls is not None:
		aux_hulls = np.vstack([arrayHulls,aux_hulls])
		aux_dense = np.vstack([arrayDense,aux_dense])
	return aux_hulls,aux_dense




#write dictionary to csv
def appendToCSV(path, dictVal):
	d = {str(x):[dictVal[x]] for x in dictVal}
	df = pd.DataFrame.from_dict(d)	

	if os.path.exists(path):
		dfO = pd.read_csv(path, header=0)
		df= pd.concat([dfO, df],axis=0)
	
	df.to_csv(path, index=False)
	pass
