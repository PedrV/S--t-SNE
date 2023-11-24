#!D:\Mestrado\.VENVS\DSMTestes\Scripts\python.exe
import sys
from Memory import S_TSNE_Memory,TSNE_Memory
from Demo_Params import setup
from sklearn import datasets
import pandas as pd

batch, slice_size, params_tsne, params_s_tsne, basedir_tsne, basedir_s_tsne, random_state_slice = setup(int(sys.argv[1]),sys.argv[2])


if sys.argv[2] == "MNIST":
	digits = datasets.load_digits()
	X, y = digits['data'], digits['target']

else:
	vals = pd.read_csv("driftingPoints_shuffle.csv", header=0)
	X, y = vals.drop(["Time","Distribution"],axis=1).to_numpy(), vals['Distribution'].to_numpy()
	y[y=="A"] = 0
	y[y=="B"] = 1
	y[y=="C"] = 2


if int(sys.argv[3]) > 0:
	TSNE_Memory(batch,params_tsne,slice_size,random_state_slice,False).run(X,y,basedir_tsne)

else:
	S_TSNE_Memory(batch,params_s_tsne,slice_size,random_state_slice,False).run(X,y,basedir_s_tsne)