This folder contains scripts used to compute homophily scores and link prediction experiments in the paper

- Higher-order homophily on simplicial complexes. Arnab Sarker, Natalie Northrup, and Ali Jadbabaie. 2024.

<code>compute_scores.py</code> computes the simplicial homophily scores and hypergraph homophily scores for each dataset. It can also be modified to compute the scores on the training data in the link prediction experiments.

<code>link_prediction.py</code> runs the experiments on link prediction, with and without the use of node features to predict if higher-order links will form in the later part of the dataset.
