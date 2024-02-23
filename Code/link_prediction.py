import pandas as pd
import numpy as np
import networkx as nx
from math import comb
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import coo_array
import pickle
import sys

start_split = 25
mid_split = 50
end_split = 100

## Step 1: Build Feature Matrix for each dataset
max_size = 8

def process_aux(dataset, start, end):
    simplex_col_idxs = [] # each simplex corresponds to a column
    node_row_idxs = [] # each node corresponds to a row
    data = [] # will be filled with ones
    
    nverts = np.loadtxt(f"LinkPredictionData/{dataset}/nverts.txt", dtype=int)
    times = np.loadtxt(f"LinkPredictionData/{dataset}/times.txt")
    simplices = np.loadtxt(f"LinkPredictionData/{dataset}/simplices.txt", dtype=int)
    
    max_size = 8
    curr_idx = 0
    col_idx = 0
    for simplex_size, t in zip(nverts, times):
        if((simplex_size <= max_size) and (t >= start) and (t < end)):
            curr_simplex = list(simplices[curr_idx:curr_idx+simplex_size])
            simplex_col_idxs += [col_idx]*simplex_size
            node_row_idxs += curr_simplex
            data += [1]*simplex_size
            col_idx += 1
        curr_idx += simplex_size
    
    # bipartite incidence matrix mapping each node to a 
    A = coo_array((data, (node_row_idxs, simplex_col_idxs)))
    B = (A @ A.T )
    B.setdiag(0)
    G = nx.from_scipy_sparse_array(B)  
    S_deg = {idx: v for idx, v in enumerate(np.sum(B, axis=1)) if v > 0}
    return B, G, S_deg

# return all subsets of k nodes for which all simplices of size (k-1) exist, but none of size k
def get_hollow_size_k(k, dataset, start, end, G=None, store_files=False):
    if(G is None):
        _, G, _ = process_aux(dataset, start, end)
    
    nverts = np.loadtxt(f"LinkPredictionData/{dataset}/nverts.txt", dtype=int)
    times = np.loadtxt(f"LinkPredictionData/{dataset}/times.txt")
    simplices = np.loadtxt(f"LinkPredictionData/{dataset}/simplices.txt", dtype=int)
    
    # first, compute all interactions of size (k-1) and interactions of size k
    max_size = 8
    curr_idx = 0
    km1_arr = []
    k_arr = []
    for simplex_size, t in zip(nverts, times):
        if((simplex_size <= max_size) and (t >= start) and (t < end)):
            curr_simplex = list(simplices[curr_idx:curr_idx+simplex_size])
            for simplex in combinations(curr_simplex, k-1):
                km1_arr.append(np.sort(simplex))
            for simplex in combinations(curr_simplex, k):
                k_arr.append(np.sort(simplex))
        curr_idx += simplex_size
        
    if((k == 3) and store_files):
        km1_df = pd.DataFrame(km1_arr).drop_duplicates()
        km1_df.columns = ['node_1', 'node_2']
        km1_df.to_csv(f"LinkPredictionData/{dataset}/size-{k-1}-training.csv", index=False)
    km1_arr = pd.DataFrame(km1_arr).drop_duplicates().values
    
    
    col_names = [f'node_{i+1}' for i in range(k)]
    k_df = pd.DataFrame(k_arr).drop_duplicates()
    k_df.columns = col_names
    
    
    # then, get the subsets of size k from the k-1 interactions where all lower order interactions occur
    skeleton_simplices = []
    for row in km1_arr:
        ## for each simplex, find the common neighbors of all the nodes
        common_neighbors = set(G.neighbors(row[0]))
        for i in range(1, k-1):
            common_neighbors = common_neighbors.intersection(set(G.neighbors(row[i])))

        for t in common_neighbors:
            curr_simplex = sorted(list(row) + [t])
            skeleton_simplices.append(curr_simplex)

    skeleton_df = pd.DataFrame(skeleton_simplices)
    skeleton_df.columns = col_names
    
    skeleton_df['ct'] = 1
    skeleton_df = skeleton_df.groupby(col_names)['ct'].count().reset_index()
    
    
    # this dataframe contains all subsets of size k where associated (k-1) interactions occur
    skeleton_df = skeleton_df.loc[skeleton_df['ct'] == k, col_names]
    
    # storing the dataframes
    if(store_files):
        skeleton_df.to_csv(f"LinkPredictionData/{dataset}/skeleton-size-{k}-training.csv", index=False)
        k_df.to_csv(f"LinkPredictionData/{dataset}/size-{k}-training.csv", index=False)
    
    merged_df = pd.merge(skeleton_df, k_df, on=col_names, how='left', indicator=True)
    
    hollow_df = merged_df.loc[merged_df['_merge'] == 'left_only', col_names]
    return hollow_df


def create_feature_matrix(k, size_k_hollow, B, G, S_deg):
    num_feats = 2 * comb(k, 2) + 2 * k + 1
    X = np.zeros((len(size_k_hollow), 2*num_feats))
    for idx, (df_idx, row) in enumerate(size_k_hollow.iterrows()):
        node_labels = np.sort([row[f'node_{i+1}'] for i in range(k)])
        if(k == 3):
            pair_labels = [(1, 2), (2, 3), (1, 3)]
            pair_labels = [np.sort([row[f'node_{i}'], row[f'node_{j}']]) for (i,j) in pair_labels]
        else:
            pair_labels = [tuple(np.sort([i, j])) for (i,j) in combinations(node_labels, 2)]
            pair_labels.sort()
        col_idx = 0
        for (i, j) in pair_labels:
            X[idx, col_idx] = B[(i, j)]
            col_idx += 1
        for i in node_labels:
            X[idx, col_idx] = G.degree(i)
            col_idx += 1
        for i in node_labels:
            X[idx, col_idx] = S_deg[i]
            col_idx += 1
        for (i, j) in pair_labels:
            common_ij = set(G.neighbors(i)) & set(G.neighbors(j))
            X[idx, col_idx] = len(common_ij)
            col_idx += 1
        # all common neighbors
        all_neighbors = set(G.neighbors(row[f'node_1']))
        for l in range(2, k+1):
            all_neighbors = all_neighbors & set(G.neighbors(row[f'node_{l}']))
        X[idx, col_idx] = len(all_neighbors)
        
        X[idx, num_feats:(num_feats + comb(k, 2) + 2 * k)] = np.log(X[idx, 0:comb(k, 2) + 2 * k])
        
        X[idx, (num_feats + comb(k, 2) + 2 * k):] = np.log(X[idx, comb(k, 2) + 2 * k:num_feats] + 1)
    return X

def create_feature_matrix_homophily(k, size_k_hollow, B, G, S_deg, labels):
    num_feats = 2 * comb(k, 2) + 2 * k + 1
    X = np.zeros((len(size_k_hollow), 2*num_feats + 1))
    X[:, :-1] = create_feature_matrix(k, size_k_hollow, B, G, S_deg)
    for idx, (df_idx, row) in enumerate(size_k_hollow.iterrows()):
        is_homophilous = True
        try:
            for i in range(k-1):
                if(labels[row[f'node_{i+1}']] != labels[row[f'node_{i+2}']]):
                    is_homophilous = False
        except Exception as e:
            is_homophilous = False
        X[idx, -1] = 1 if is_homophilous else 0
    return X


## check if each open simplex closes later
def get_closures_k(k, size_k_hollow_pre, start_test, end_test):
    nverts = np.loadtxt(f"LinkPredictionData/{dataset}/nverts.txt", dtype=int)
    times = np.loadtxt(f"LinkPredictionData/{dataset}/times.txt")
    simplices = np.loadtxt(f"LinkPredictionData/{dataset}/simplices.txt", dtype=int)
    
    # first, compute all interactions of size (k-1) and interactions of size k
    curr_idx = 0
    k_arr = []
    for simplex_size, t in zip(nverts, times):
        if((simplex_size <= max_size) and (t >= start_test) and (t < end_test)):
            curr_simplex = list(simplices[curr_idx:curr_idx+simplex_size])
            for simplex in combinations(curr_simplex, k):
                k_arr.append(np.sort(simplex))
        curr_idx += simplex_size
        
    col_names = [f'node_{i+1}' for i in range(k)]
    k_df = pd.DataFrame(k_arr).drop_duplicates()
    k_df.columns = col_names
    
    merged_df = pd.merge(size_k_hollow_pre, k_df, on=col_names, how='left', indicator=True)
    return merged_df['_merge'].map({'left_only': 0, 'both': 1})


results_arr = []

datasets = ["cont-hospital", "cont-workplace-13", "hosp-DAWN", "coauth-dblp",
                "email-Enron", "cont-workplace-15",
                "cont-primary-school", "bills-senate", "cont-high-school",
                "bills-house"]


for dataset in datasets:
    for k in [3, 4, 5]:
        try:
            # set up auxilliary matrices
            all_ts = np.loadtxt(f'LinkPredictionData/{dataset}/times.txt')
            min_t = np.min(all_ts) - 1
            start_t = np.quantile(all_ts, start_split / 100)
            med_t = np.quantile(all_ts, mid_split / 100)
            end_t = np.max(all_ts) + 1

            B_train, G_train, S_deg_train = process_aux(dataset, min_t, start_t)
            B_test, G_test, S_deg_test = process_aux(dataset, min_t, med_t)

            # get labels
            label_df = pd.read_csv(f"LinkPredictionData/{dataset}/labels.csv")
            label_dict = {row['id']: row['group_code'] for idx, row in label_df.iterrows()}


            # create feature matrix
            size_k_hollow_train = get_hollow_size_k(k, dataset, min_t, start_t, G=G_train)
            
            
            ## Test performance
            size_k_hollow_test = get_hollow_size_k(k, dataset, min_t, med_t, G=G_test, store_files=True)
            
            
            for homophily in [True, False]:
                if(homophily):
                    X_train = create_feature_matrix_homophily(k, size_k_hollow_train, B_train, G_train, S_deg_train, label_dict)
                else:
                    X_train = create_feature_matrix(k, size_k_hollow_train, B_train, G_train, S_deg_train)

                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                y_train = get_closures_k(k, size_k_hollow_train, start_t, med_t)

                ## Train linear model -- unregularized logistic regression
                clf = LogisticRegression(fit_intercept=True, solver='liblinear', C=1e42)
                clf.fit(X_train, y_train)

                if(homophily):
                    X_test = create_feature_matrix_homophily(k, size_k_hollow_test, B_test, G_test, S_deg_test, label_dict)
                else:
                    X_test = create_feature_matrix(k, size_k_hollow_test, B_test, G_test, S_deg_test)
                    

                X_test = scaler.transform(X_test)
                y_test = get_closures_k(k, size_k_hollow_test, med_t, end_t)
                

                # classifier results
                probs = clf.predict_proba(X_test)[:, 1]
                auc_pc_score = average_precision_score(y_test, probs) / np.mean(y_test)
                
                ## confidence intervals
                n_samples = 100
                bootstrap_scores = []
                i = 0
                while i < n_samples:
                    try:
                        sample_index = np.random.choice(range(0, len(y_train)), int(0.8*len(y_train)))

                        X_samples = X_train[sample_index]
                        y_samples = y_train[sample_index]    
                        
                        if(np.sum(y_samples) == 0):
                            continue

                        lr = LogisticRegression(fit_intercept=True, solver='liblinear', C=1e42)
                        lr.fit(X_samples, y_samples)
                        curr_probs = lr.predict_proba(X_test)[:, 1]
                        bootstrap_scores.append(average_precision_score(y_test, curr_probs) / np.mean(y_test))
                        i += 1
                    except Exception as e:
                        print(i)
                    

                results_arr.append({
                    "k": k,
                    "dataset": dataset,
                    "homophily_feature": homophily,
                    "rel_auc_pr": auc_pc_score, 
                    "baseline": np.mean(y_test),
                    'std_err': np.std(bootstrap_scores) / n_samples
                })

                pd.DataFrame(results_arr).to_csv(f"Results/linkprediction_results_unreg_Cinf.csv", index=False)
        except Exception as e:
            print(dataset, k, homophily)
            print(e)

