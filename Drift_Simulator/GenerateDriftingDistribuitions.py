from clusterSim import ClusterSim
import pandas as pd
import numpy as np

'''
Script used to generate three 3-D Distributions A,B and C using the ClusterSim class
the script register the points in a CSV file and plot the distributions in 3-D
'''

covA1 = [
	[ 1, 0, 0],
	[ 0, 1, 0],
	[ 0, 0, 1],
]
covA2 = [
	[ 30, 0, 0],
	[ 0, 42, 0],
	[ 0, 0, 81],
]


covB1 = [
	[ 30, 0, 0],
	[ 0, 42, 0],
	[ 0, 0, 81],
]
covB2 = [
	[ 12, 21, 5],
	[ 21, 9, 13],
	[ 5, 13, 7],
]

covC1 = [
	[ 340, 81, 27],
	[ 81, 472, 37],
	[ 27, 37, 381],
]
covC2 = [
	[ 2, 0, 0],
	[ 0, 2, 0],
	[ 0, 0, 2],
]




points = {}


clSims = [ClusterSim("A",[120,130,150],covA1,[0,0,0],covA2,100,10), 
		  ClusterSim("B",[-10,16,-250],covB1,[90,130,200],covB2,100,10), 
		  ClusterSim("C",[40,3,-25],covC1,[40,3,-25],covC2,100,10)]


timeVariation = []
timeVal = 0
pointsPerTick = 70
for x in range(2500):
	for clSim in clSims:
		if clSim.name not in points:
			points[clSim.name] = clSim.get_n_points(pointsPerTick,numpy=True)

		else:
			points[clSim.name] = np.vstack([points[clSim.name],clSim.get_n_points(pointsPerTick,numpy=True)])

	timeVariation += [timeVal for x in range(pointsPerTick)]
	timeVal += 1

	[clSim.tick() for clSim in clSims]


dfs = {x:pd.DataFrame(points[x]) for x in points}

for x in dfs:
	dfs[x]["Dist"] = x
	dfs[x]["timeVar"] = timeVariation
	dfs[x].columns = ["x","y","z","Distribution","Time"]


toConcat = [df.copy() for df in dfs.values()]

conc = pd.concat(toConcat)
conc.to_csv("driftingPoints_class.csv", index=False)
toShuffle = [(conc[conc["Time"] == x]).sample(frac = 1) for x in conc["Time"].unique()]

shuffled = pd.concat(toShuffle)
shuffled.to_csv("driftingPoints_shuffle.csv", index=False)

scales = ['Algae','Amp',"purp"]
import plotly.graph_objects as go


fig = go.Figure(data=[
	go.Scatter3d(
    x=df["x"],
    y=df["y"],
    z=df["z"],
    mode='markers',
    marker=dict(
        size=12,
        color=df["Time"],
        colorscale=scale,
        opacity=0.8
    )) for df,scale in zip(dfs.values(),scales)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()

