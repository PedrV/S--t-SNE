# PARAMETERS and shared functions across Tests

random_state_slice = 30
parameters_s_tsne = {
	"MNIST":{
		"hulls_threshold" : 50,
		"max_batch_size" :50,
		"density_search_radius" : 45,
		"density_min_points" : 100,
		"density_max_points" : 500,
		"prob_removal_demo" : 0,
		"affinity_params" : {"perplexity":30,
							 "metric":"euclidean",
							 "n_jobs":-1,
							 "random_state":42,
							 "verbose":False,
							},
		"initialization_params" : {"random_state":42},
		"embedding_params" : {"negative_gradient_method":"bh",
							  "n_jobs":8,
							  "random_state":42,
							  "verbose":False
							 },
		"exageration_params" : {
			"n_iter":400,"exaggeration":6,"momentum":0.8,
			"learning_rate":0.1,"max_grad_norm":0.25,"max_step_norm":None
		},
		"optimize_params" : {
			"n_iter":650,"momentum":0.9, "exaggeration":1,
			"learning_rate":0.1,"max_grad_norm":0.25,"max_step_norm":None
		},				
		"initial_exageration_params" : {
			"n_iter":450, "exaggeration":35, "momentum":0.8
		},
		"initial_optimize_params" : {
			"n_iter":700, "momentum":0.8
		}
	},

	"DRIFTED_500K":{
		"hulls_threshold" : 450,
		"max_batch_size" : 200,
		"density_search_radius" : 70,
		"density_min_points" : 100,
		"density_max_points" : 500,
		"prob_removal_demo" : 0,
		"affinity_params" : {"perplexity":30,
							 "metric":"euclidean",
							 "n_jobs":-1,
							 "random_state":42,
							 "verbose":False,
							},
		"initialization_params" : {"random_state":42},
		"embedding_params" : {"negative_gradient_method":"bh",
							  "n_jobs":8,
							  "random_state":42,
							  "verbose":False
							 },
		"exageration_params" : {
			"n_iter":400,"exaggeration":5,"momentum":0.8,
			"learning_rate":0.1,"max_grad_norm":0.25,"max_step_norm":None
		},
		"optimize_params" : {
			"n_iter":650,"momentum":0.9, "exaggeration":1,
			"learning_rate":0.1,"max_grad_norm":0.25,"max_step_norm":None
		},				
		"initial_exageration_params" : {
			"n_iter":450, "exaggeration":35, "momentum":0.8
		},
		"initial_optimize_params" : {
			"n_iter":700, "momentum":0.8
		}
	}
}


BASESETUP = {
	"MNIST":[
		(200,.2,30,(200,250),(450,700)),
		(200,.3,30,(200,250),(450,700)),
		(200,.5,30,(200,250),(450,700)),
		(200,.6,30,(200,250),(450,700)),
		################################
		(300,.2,30,(200,250),(450,700)),
		(300,.3,30,(200,250),(450,700)),
		(300,.5,30,(200,250),(450,700)),
		(300,.6,30,(200,250),(450,700)),
		################################
		(400,.2,30,(200,250),(450,700)),
		(400,.3,30,(200,250),(450,700)),
		(400,.5,30,(200,250),(450,700)),
		(400,.6,30,(200,250),(450,700)),
		################################
		(500,.2,30,(200,250),(450,700)),
		(500,.3,30,(200,250),(450,700)),
		(500,.5,30,(200,250),(450,700)),
		(500,.6,30,(200,250),(450,700)),
		#############DENSITY############
		(400,.2,30,(100,150),(450,700)),
		(400,.3,30,(100,150),(450,700)),
		(400,.5,30,(100,150),(450,700)),
		(400,.6,30,(100,150),(450,700)),
		################################
		(400,.2,30,(200,250),(450,700)),
		(400,.3,30,(200,250),(450,700)),
		(400,.5,30,(200,250),(450,700)),
		(400,.6,30,(200,250),(450,700)),
		################################
		(400,.2,30,(300,350),(450,700)),
		(400,.3,30,(300,350),(450,700)),
		(400,.5,30,(300,350),(450,700)),
		(400,.6,30,(300,350),(450,700)),
		################################
		(400,.2,30,(400,450),(450,700)),
		(400,.3,30,(400,450),(450,700)),
		(400,.5,30,(400,450),(450,700)),
		(400,.6,30,(400,450),(450,700)),
		#############ITERS##############
		(400,.2,30,(300,350),(250,400)),
		(400,.3,30,(300,350),(250,400)),
		(400,.5,30,(300,350),(250,400)),
		(400,.6,30,(300,350),(250,400)),
		################################
		(400,.2,30,(300,350),(350,600)),
		(400,.3,30,(300,350),(350,600)),
		(400,.5,30,(300,350),(350,600)),
		(400,.6,30,(300,350),(350,600)),
		################################
		(400,.2,30,(300,350),(450,800)),
		(400,.3,30,(300,350),(450,800)),
		(400,.5,30,(300,350),(450,800)),
		(400,.6,30,(300,350),(450,800)),
		################################
		(400,.2,30,(300,350),(650,1000)),
		(400,.3,30,(300,350),(650,1000)),
		(400,.5,30,(300,350),(650,1000)),
		(400,.6,30,(300,350),(650,1000)),
	],
	
	"DRIFTED_500K":[
		(200,.005,30,(200,250),(450,700)),
		(400,.005,30,(200,250),(450,700)),
		(400,.005,30,(400,450),(450,700)),
		(400,.005,30,(400,450),(650,1000))
	]
}



params_tsne = {
	"random_state":42,
	"affinities_params":{
		"perplexity":30,
		"metric":"euclidean",
		"n_jobs":8,
		"verbose":False,		
	},
	"embedding_params":{
		"negative_gradient_method":"bh",
		"n_jobs":8,
		"verbose":False,	
	},
	"optmizations":[
		{
			"n_iter":250, 
			"exaggeration":12,
			"momentum":0.5
		},
		{
			"n_iter":500, 
			"momentum":0.8
		},
	]
}




def setup(x,setupType):
	import os
	global BASESETUP, parameters_s_tsne, params_s_tsne

	params_s_tsne = parameters_s_tsne[setupType]

	batch,slice_size,random_state_slice,density_params,iters, = BASESETUP[setupType][x]
	
	params_s_tsne["density_min_points"],params_s_tsne["density_max_points"] = density_params
	params_s_tsne["exageration_params"]["n_iter"],params_s_tsne["optimize_params"]["n_iter"]  = iters
	params_s_tsne["max_batch_size"] = batch

	basedir_tsne = os.path.join("Results",setupType,f"B{batch}_SL{slice_size}_TSNE")
	basedir_s_tsne = os.path.join("Results",setupType, f"B{batch}_SL{slice_size}_IT{params_s_tsne['optimize_params']['n_iter']}_D{params_s_tsne['density_min_points']}")
	
	return batch, slice_size, params_tsne, params_s_tsne, basedir_tsne, basedir_s_tsne, random_state_slice



