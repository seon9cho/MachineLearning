# MachineLearning

My solutions to the data, probability and machine learning labs from BYU's Applied and Computational Mathematics Emphasis (ACME), drawn from *Volumes 1–3* of the lab manuals. Each lab is a notebook that pairs the spec's problem statements with my solutions and their output, plus condensed notes on the concepts the lab covers.

| Lab | Volume | Notebook |
|-----|--------|----------|
| The PageRank Algorithm | Vol. 1, Lab 13 | [PageRank/pagerank.ipynb](PageRank/pagerank.ipynb) |
| Nearest Neighbor Search | Vol. 2, Lab 3 | [NearestNeighbor/nearest_neighbor.ipynb](NearestNeighbor/nearest_neighbor.ipynb) |
| Markov Chains | Vol. 2, Lab 5 | [MarkovChains/markov_chains.ipynb](MarkovChains/markov_chains.ipynb) |
| Value Function Iteration | Vol. 2, Lab 18 | [ValueFunctionIteration/value_iteration.ipynb](ValueFunctionIteration/value_iteration.ipynb) |
| Policy Function Iteration | Vol. 2, Lab 19 | [PolicyIteration/policy_iteration.ipynb](PolicyIteration/policy_iteration.ipynb) |
| ARMA Models | Vol. 3 | [ARMA/arma.ipynb](ARMA/arma.ipynb) |
| K-Means Clustering | Vol. 3 | [KMeans/kmeans_clustering.ipynb](KMeans/kmeans_clustering.ipynb) |
| Kalman Filter | Vol. 3 | [KalmanFilter/kalman_filter.ipynb](KalmanFilter/kalman_filter.ipynb) |
| Gibbs Sampling and LDA | Vol. 3 | [LDA/gibbs_lda.ipynb](LDA/gibbs_lda.ipynb) |
| Metropolis | Vol. 3 | [Metropolis/metropolis.ipynb](Metropolis/metropolis.ipynb) |
| PCA and LSI | Vol. 3 | [PCA_LSI/pca_lsi.ipynb](PCA_LSI/pca_lsi.ipynb) |
| Introduction to scikit-learn | Vol. 3 | [SciKitIntro/scikit_intro.ipynb](SciKitIntro/scikit_intro.ipynb) |
| Speech Recognition | Vol. 3 | [SpeechRecognition/speech_recognition.ipynb](SpeechRecognition/speech_recognition.ipynb) |

The Volume 3 labs quote their problem statements from the course's lab notebooks (and, for ARMA, from the spec source), since I don't have the Volume 3 manual from that time. The Kalman Filter statements weren't kept, so that notebook shows only my solutions, and PCA and LSI quotes the lab template's descriptions.

Each folder also keeps the grader feedback from my original submissions (`*_feedback.txt`).

## Running the notebooks

Each notebook runs from its own folder. The data each lab uses sits next to its notebook. The notebooks need NumPy, SciPy, Matplotlib, pandas and scikit-learn; the ARMA notebook also uses statsmodels for a cross-check.

- The Nearest Neighbor lab imports `trees.py`, my binary search tree module from the DataStructures labs, which is kept next to its notebook.
- The Speech Recognition lab uses `MFCC.py`, `gmm.py` and `gmmhmm.py`, helper modules provided with the lab (the last two come from an old version of scikit-learn). They carry small `# [compat]` fixes so they run on current NumPy and scikit-learn.

## Credits

The lab specifications come from the [Foundations of Applied Mathematics](https://foundations-of-applied-mathematics.github.io/) lab materials, © Dr. J. Humpherys, licensed under [CC BY 3.0 US](http://creativecommons.org/licenses/by/3.0/us/). The data files come from the original lab materials and the project's official [data repository](https://github.com/Foundations-of-Applied-Mathematics/Data); `mnist_subset.npz` stands in for the `PostalData.npz` file the Nearest Neighbor spec used, which is no longer published. Problem statements are quoted from the specs and course notebooks; the solutions are my own.
