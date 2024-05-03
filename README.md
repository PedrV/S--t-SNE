# S+t-SNE

#### [S+t-SNE - Bringing dimensionality reduction to data streams](https://arxiv.org/abs/2403.17643)

We present S+t-SNE, an adaptation of the t-SNE algorithm designed to handle infinite data streams.
The core idea behind S+t-SNE is to update the t-SNE embedding incrementally as new data arrives, ensuring scalability and adaptability to handle streaming scenarios. By selecting the most important points at each step, the algorithm ensures scalability while keeping informative visualisations. Employing a blind method for drift management adjusts the embedding space, facilitating continuous visualisation of evolving data dynamics.
Our experimental evaluations demonstrate the effectiveness and efficiency of S+t-SNE. The results highlight its ability to capture patterns in a streaming scenario. We hope our approach offers researchers and practitioners a real-time tool for understanding and interpreting high-dimensional data.

**Contributors**: Pedro C. Vieira[^1] <pedrocvieira@fc.up.pt>, João P. Montrezol <joao.antunes@fc.up.pt>, João T. Vieira <up201905419@fc.up.pt>, João Gama <jgama@fep.up.pt>


## Organization

- [images/paper_plus_extra/](images/paper_plus_extra/): images used in the paper
- [images/paper_plus_extra/extra/](images/paper_plus_extra/extra/): images not used in the paper
- [Drift_Simulator/](Drift_Simulator/): scripts to create the dataset with drifted used in the paper
- [Drift_Simulator/Images/](Drift_Simulator/Images/): 3d images of dataset with drift used in the paper
- [Demo/](Demo/): scripts used for demonstration of results from S+t-SNE, the scripts include classes for testing time and memory complexity as well as KL divergence. The Run.py file is a compilation of parameters and methodologies of testing for measuring the algorithm and saving CSVs results.
- [ecs/](ecs/): scripts used to run Exponential Cobweb Slicing (ECS)
- [STSNE.py](STSNE.py): Main class of the S+t-SNE addapted for the [River API](https://github.com/online-ml/river) structure.


## Running S+t-SNE

#### Using S+t-SNE
The provided S+t-SNE implementation is based on River API, to run the python class one need to initialize it with a sample of the data on its constructor alongside the parameters for the t-SNE operations. Afterwards one can use one of two options to embed examples:
- `learn_one(x)`, to passively learn an example, adds an example to the current batch, waiting till the points threshold is met and the batch is flushed;
- `predict_one(x)`, to actively learn an example, adds an example to the current batch and forces the batch to be flushed;
- `flush_batch()`, to force a batch to be flushed;

As for ECS, the hyperparameters used correspond to how the exponential decay behaves. Said parameters can be readily changed in [subregion.py](ecs/subregion.py).
The number of concentric regions can also be changed by modifying the `layers_distance_ratio` in [convex_polygon.py](ecs/convex_polygon.py). Each value in the list
will be translated into a distance from the centroid and a concetric region will be constructed based on said distance. 
Any other change like how points are counted involve, *for now*, changing multiple functions.

#### Tests
Firstly, go to this drive [folder](https://drive.google.com/drive/folders/1saFrxafJgHsPeKMOJpvVKYQALpUnumGr?usp=drive_link) and download the folder `Demo/`. Copy all folders and files to the `Demo/` folder obtained from the repository. In the case of `ProcessedAnalysisCSV/` directory move the files inside it to the corresponding directory in the repository. The same applies to the files inside `Demo/` in the drive. As for the directory `Results/` in the drive, its enough to just copy it to the `Demo/` in the repository.

After completing that, one can simply explore the functions on the avilable jupyter notebooks to generate the plots with the available CSVs on the `Results/` folder. To generate the CSVs by yourself, check if the file `driftingPoints_shuffle.csv` is available on the `Demo/` folder and SKLearn library is available, afterwards one can personalize the test parameters on the file `Demo_Params.py`, and test routines on the file `Run.py`. Finally run the `Run.py` scripts to get new versions of the CSVs at the `Results/` folder. 

## Notes
- In the version of the paper freely available in ArXiv, the fourth paragraph of section 3.3 - Handling Drift is poorly worded. A better description would be:

    > Our solution involves using the convex hulls obtained from clustering and dividing them into parts. Each partition will employ blind drift detection by exponential decay based on the number of iterations in S+t-SNE. This allows parts without new points during a period (given by exponential decay) to disappear, ensuring consistency. We parameterise the exponential decay with three parameters, $\alpha = 0.88$, $\beta = 1.6$ and $\eta = 0.01$, yielding $N(t) = \alpha e^{−t\eta+\beta}$ where $t$ is the number of iterations. This expression encapsulates our deﬁnition of drift. In this conﬁguration, a polygon in iteration 200 will have section x cut if said section does not receive points for more than $N(200)$ iterations.

The version published in the conference should have everything corrected. We will probably update the file in ArXiv soon. 

- Better documentation for ECS will be added soon

## Contributions

- Feel free to improve the code!
  
- **If you have any questions do not hesitate to contact us (preferably the corresponding author) either by email or using GitHub functionalities!** :)

[^1]: Corresponding Author
