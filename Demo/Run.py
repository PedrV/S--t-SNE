#!D:\Mestrado\.VENVS\DSMTestes\Scripts\python.exe
from subprocess import Popen
import os
import logging
from sklearn import datasets
import pandas as pd

from Embeddings import S_TSNE_Embedding,TSNE_embedding
from Timer import S_TSNE_Time,TSNE_Time
from KLDiv import S_TSNE_KLDiv,TSNE_KLDiv
from Memory import S_TSNE_Memory,TSNE_Memory

from Demo_Params import setup,BASESETUP

os.makedirs("Results",exist_ok=True)
logging.basicConfig(filename=os.path.join("Results","ResultsDemoLogs.log"), level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("RunDemoLogger")

logger.info("========== Started ==========")


TRIALS = {
	"MNIST":5,
	"DRIFTED_500K":3,
}

ALGOS={
	"MNIST":["STSNE","TSNE"],
	"DRIFTED_500K":["STSNE"],	
}

METHODS={
	"MNIST":["embedding","KL","memory","timer"],
	"DRIFTED_500K":["embedding","memory","timer"],
}



TOTAL_PARAMS = {x:len(BASESETUP[x]) for x in BASESETUP}

for setupType in TRIALS:
	trials = TRIALS[setupType]
	combinations = TOTAL_PARAMS[setupType]
	logger.info(f"========== {setupType} ==========")
		
	for combination in range(combinations):
		batch, slice_size, params_tsne, params_s_tsne, basedir_tsne, basedir_s_tsne, random_state_slice = setup(combination,setupType)
		
		logger.info(f"Running B{batch} SL{slice_size} IT{params_s_tsne['optimize_params']['n_iter']} D{params_s_tsne['density_min_points']}")
		
		#Load Datasets
		if setupType == "MNIST":
			digits = datasets.load_digits()
			X, y = digits['data'], digits['target']

		else:
			vals = pd.read_csv("driftingPoints_shuffle.csv", header=0)
			X, y = vals.drop(["Time","Distribution"],axis=1).to_numpy(), vals['Distribution'].to_numpy()
			y[y=="A"] = 0
			y[y=="B"] = 1
			y[y=="C"] = 2

		
		if "embedding" in METHODS[setupType] and "STSNE" in ALGOS[setupType]:
			S_TSNE_Embedding(batch,params_s_tsne,slice_size,random_state_slice,False).run(X,y,basedir_s_tsne)
		
		if "embedding" in METHODS[setupType] and "TSNE" in ALGOS[setupType]:
			TSNE_embedding(batch,params_tsne,slice_size,random_state_slice,False).run(X,y,basedir_tsne)

		
		for trial in range(trials):
			logger.info(f"Trial {trial}")
			
			if "memory" in METHODS[setupType] and "STSNE" in ALGOS[setupType]:
				memory_process_stsne = Popen(f"py Run_Memory.py {combination} {setupType} 0")
			
			if "memory" in METHODS[setupType] and "TSNE" in ALGOS[setupType]:
				memory_process_tsne = Popen(f"py Run_Memory.py {combination} {setupType} 1")


			if "KL" in METHODS[setupType] and "STSNE" in ALGOS[setupType]:
				S_TSNE_KLDiv(batch,params_s_tsne,slice_size,random_state_slice,False).run(X,y,basedir_s_tsne)

			if "KL" in METHODS[setupType] and "STSNE" in ALGOS[setupType]:
				TSNE_KLDiv(batch,params_tsne,slice_size,random_state_slice,False).run(X,y,basedir_tsne)

			
			
			if "memory" in METHODS[setupType] and "STSNE" in ALGOS[setupType]:
				memory_process_stsne.wait()
			
			if "memory" in METHODS[setupType] and "TSNE" in ALGOS[setupType]:
				memory_process_tsne.wait()


			
			if "timer" in METHODS[setupType] and "STSNE" in ALGOS[setupType]:
				S_TSNE_Time(batch,params_s_tsne,slice_size,random_state_slice,False).run(X,y,basedir_s_tsne)
			
			if "timer" in METHODS[setupType] and "TSNE" in ALGOS[setupType]:
				TSNE_Time(batch,params_tsne,slice_size,random_state_slice,False).run(X,y,basedir_tsne)

logger.info("========== Finished ==========")