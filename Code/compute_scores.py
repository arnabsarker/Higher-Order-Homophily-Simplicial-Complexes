import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import comb
from scipy.stats import pearsonr

datasets = ['cont-village', 'cont-hospital', 'cont-workplace-13', 'cont-workplace-15', 
            'email-Enron', 'cont-primary-school', 'bills-senate', 'cont-high-school', 
            'bills-house', 'hosp-DAWN', 'soc-youtube', 'coauth-dblp', 
            'clicks-trivago', 'soc-livejournal', 'soc-orkut', 'soc-flickr']

# datasets for link prediction
# datasets = ["cont-hospital", "cont-workplace-13", "cont-workplace-15", "hosp-DAWN", "bills-senate", "email-Enron",
#                     "bills-house", "coauth-dblp", "cont-primary-school", "cont-high-school"]
            
max_simplex_size = 8

results_arr = []
for dataset in datasets:
    print(dataset)
    curr_info = {'dataset': dataset}
    label_df = pd.read_csv(f"ScoreData/{dataset}/labels.csv")
    label_dict = {row['id']: row['group_code'] for idx, row in label_df.iterrows()}
    
    try:
        k = 2
        simplex_df = pd.read_csv(f"ScoreData/{dataset}/size-{k}-unique-sorted.csv")
        filter_idxs = simplex_df['node_1'].isin(label_dict.keys())
        for e in range(2, k + 1):
            filter_idxs = filter_idxs & simplex_df[f'node_{e}'].isin(label_dict.keys())
        simplex_df = simplex_df[filter_idxs]
        
        affinity = np.mean(
                        simplex_df.apply(lambda row: len(set([label_dict[x] for x in row])) == 1, axis=1)
        )
        
        nodes_in_size_k_interactions = pd.unique(simplex_df.values.flatten())
        labels_on_nodes = [label_dict[x] for x in nodes_in_size_k_interactions]
        labels, label_counts = np.unique(labels_on_nodes, return_counts=True)
        hypergraph_baseline = sum([comb(nc, k) for nc in label_counts]) / comb(sum(label_counts), k)
        curr_info[f'affinity_{k}'] = affinity
        curr_info[f'edge_baseline'] = hypergraph_baseline
    except Exception as e:
        print(dataset, k)
        print(e)
    
    for k in range(3, max_simplex_size + 1):
        try:
            simplex_df = pd.read_csv(f"ScoreData/{dataset}/size-{k}-unique-sorted.csv")
            filter_idxs = simplex_df['node_1'].isin(label_dict.keys())
            for e in range(2, k + 1):
                filter_idxs = filter_idxs & simplex_df[f'node_{e}'].isin(label_dict.keys())
            simplex_df = simplex_df[filter_idxs]
            
            skeleton_df = pd.read_csv(f"ScoreData/{dataset}/skeleton-size-{k}.csv")
            filter_idxs = skeleton_df['node_1'].isin(label_dict.keys())
            for e in range(2, k + 1):
                filter_idxs = filter_idxs & skeleton_df[f'node_{e}'].isin(label_dict.keys())
            skeleton_df = skeleton_df[filter_idxs]

            ## affinity -- for each row, check if all labels are a unique value
            affinity = np.mean(
                            simplex_df.apply(lambda row: len(set([label_dict[x] for x in row])) == 1, axis=1)
            )

            # simplicial baseline is same but computed on the skeleton
            simplicial_baseline = np.mean(
                            skeleton_df.apply(lambda row: len(set([label_dict[x] for x in row])) == 1, axis=1)
            )

            # hypergraph baseline involves computing frequency of each class and then using formula from paper
            nodes_in_size_k_interactions = pd.unique(simplex_df.values.flatten())
            labels_on_nodes = [label_dict[x] for x in nodes_in_size_k_interactions]
            labels, label_counts = np.unique(labels_on_nodes, return_counts=True)

            hypergraph_baseline = sum([comb(nc, k) for nc in label_counts]) / comb(sum(label_counts), k)

            curr_info[f'affinity_{k}'] = affinity
            curr_info[f'simplicial_baseline_{k}'] = simplicial_baseline
            curr_info[f'hypergraph_baseline_{k}'] = hypergraph_baseline
        except Exception as e:
            print(dataset, k)
            print(e)
        
    results_arr.append(curr_info)
        

    results_df = pd.DataFrame(results_arr)
    results_df['dyadic_score'] = results_df['affinity_2'] / results_df['edge_baseline']
    results_df[f'log_dyadic_score'] = np.log(results_df[f'dyadic_score'])

    for k in range(3, max_simplex_size + 1):
        results_df[f'simplicial_score_{k}'] = results_df[f'affinity_{k}'] / results_df[f'simplicial_baseline_{k}']
        results_df[f'hypergraph_score_{k}'] = results_df[f'affinity_{k}'] / results_df[f'hypergraph_baseline_{k}']
        results_df[f'log_simplicial_score_{k}'] = np.log(results_df[f'simplicial_score_{k}'])
        results_df[f'log_hypergraph_score_{k}'] = np.log(results_df[f'hypergraph_score_{k}'])

    results_df.to_csv("Results/homophily_results.csv", index=False)
