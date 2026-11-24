# MachineLearning

My solutions to the data, probability and machine learning labs from BYU's Applied and Computational Mathematics Emphasis (ACME), drawn from *Volumes 1–3* of the lab manuals. Each converted lab is a notebook that pairs the spec's problem statements with my solutions and their output, plus condensed notes on the concepts the lab covers.

| Lab | Volume | Notebook |
|-----|--------|----------|
| The PageRank Algorithm | Vol. 1, Lab 13 | [PageRank/pagerank.ipynb](PageRank/pagerank.ipynb) |
| Nearest Neighbor Search | Vol. 2, Lab 3 | [NearestNeighbor/nearest_neighbor.ipynb](NearestNeighbor/nearest_neighbor.ipynb) |
| Markov Chains | Vol. 2, Lab 5 | [MarkovChains/markov_chains.ipynb](MarkovChains/markov_chains.ipynb) |
| Value Function Iteration | Vol. 2, Lab 18 | [ValueFunctionIteration/value_iteration.ipynb](ValueFunctionIteration/value_iteration.ipynb) |
| Policy Function Iteration | Vol. 2, Lab 19 | [PolicyIteration/policy_iteration.ipynb](PolicyIteration/policy_iteration.ipynb) |

### Volume 3 labs (original course notebooks, not yet converted)

| Lab | Files |
|-----|-------|
| ARMA Models | [ARMA/](ARMA/) |
| K-Means Clustering | [KMeans/](KMeans/) |
| Kalman Filter | [KalmanFilter/](KalmanFilter/) |
| Gibbs Sampling and LDA | [LDA/](LDA/) |
| Metropolis | [Metropolis/](Metropolis/) |
| PCA and LSI | [PCA_LSI/](PCA_LSI/) |
| Scikit-Learn Introduction | [SciKitIntro/](SciKitIntro/) |
| Speech Recognition | [SpeechRecognition/](SpeechRecognition/) |

Each folder also keeps the grader feedback from my original submissions (`*_feedback.txt`).

## Running the notebooks

Each notebook runs from its own folder. The data each lab uses sits next to its notebook. The notebooks need NumPy, SciPy, Matplotlib and scikit-learn. The Nearest Neighbor lab imports `trees.py`, my binary search tree module from the DataStructures labs, which is kept next to its notebook.

## Credits

The lab specifications come from the [Foundations of Applied Mathematics](https://foundations-of-applied-mathematics.github.io/) lab manuals, © Dr. J. Humpherys, licensed under [CC BY 3.0 US](http://creativecommons.org/licenses/by/3.0/us/). The data files come from the original lab materials and the project's official [data repository](https://github.com/Foundations-of-Applied-Mathematics/Data); `mnist_subset.npz` stands in for the `PostalData.npz` file the Nearest Neighbor spec used, which is no longer published. Problem statements are quoted from the specs; the solutions are my own.
