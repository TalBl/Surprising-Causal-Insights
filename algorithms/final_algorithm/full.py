import pandas as pd
import numpy as np
from time import time
#from Tal.clean_meps import build_mini_df, SUBPOPULATIONS, create_value_dict, OUTCOME_COLUMN, TREATMENTS_COLUMNS
from Tal.clean_so import build_mini_df, SUBPOPULATIONS, create_value_dict, OUTCOME_COLUMN, TREATMENTS_COLUMNS
from algorithms.divexplorer.divexplorer import DivergenceExplorer
from algorithms.final_algorithm.find_best_treatment import find_best_treatment
from algorithms.final_algorithm.new_greedy import find_group, calc_facts_metrics
from tqdm import tqdm
import warnings
from dowhy import CausalModel
from Utils import getTreatmentCATE
import networkx as nx

warnings.filterwarnings("ignore")

PRECENT_CLEANING_THRESHOLD = 0.5
DIVEXPLORER_THRESHOLD_SUPPORT = 0.1
PROJECT_DIRECTORY = "so"
K = 5
LAMDA_VALUES = [0.0001, 0.00005, 0.00009]
CALCED_INTERSECTIONS = {}
CALCED_GRAPHS = {}

GROUP_1 = [{"att": "group1", "value": 1}]
GROUP_2 = [{"att": "group2", "value": 1}]

def clean_treatments(df, GRAPH, OUTCOME_COLUMN, TREATMENTS, cols_dict):
    edges, cared_atts = [], []
    atts = list(set([x['att'] for x in TREATMENTS.values()]))
    for line in GRAPH:
        if '->' in line:
            edges.append([line.split(" ->")[0].split("'")[1], line.split("-> ")[1].split(";'")[0]])
    causal_graph = nx.DiGraph()
    causal_graph.add_edges_from(edges)
    for att in atts:
        model = CausalModel(
            data=df,
            graph=causal_graph,
            treatment=cols_dict[att],
            outcome=OUTCOME_COLUMN)
        estimands = model.identify_effect()
        if not estimands.no_directed_path:
            cared_atts.append(att)
    clean_treats = {}
    for k, t in TREATMENTS.items():
        if t['att'] in cared_atts:
            clean_treats[k] = t
    return clean_treats


def algorithm():
    df_clean, cols_dict = build_mini_df()
    df_clean.to_csv(f"outputs/{PROJECT_DIRECTORY}/clean_data.csv", index=False)
    # step 1 - get best subpopulation by divexplorer
    df_clean = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/clean_data.csv")
    fp_diver = DivergenceExplorer(df_clean)
    subgroups = fp_diver\
        .get_pattern_divergence(min_support=DIVEXPLORER_THRESHOLD_SUPPORT, quantitative_outcomes=[OUTCOME_COLUMN],
                                group_1_column="group1", group_2_column="group2", attributes=SUBPOPULATIONS, COLUMNS_TO_IGNORE=[])\
        .sort_values(by=f"{OUTCOME_COLUMN}_div", ascending=False, ignore_index=True)
    subgroups = subgroups.dropna()
    subgroups.to_csv(f"outputs/{PROJECT_DIRECTORY}/interesting_subpopulations.csv", index=False)
    # step 2 - find the best treatment for each subpopulation
    df_clean = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/clean_data.csv")
    subgroups = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/interesting_subpopulations.csv")
    intresting_subpopulations = list(subgroups['itemset'])
    r_dict = []
    with open(f"data/{PROJECT_DIRECTORY}/causal_dag.txt", "r") as f:
        GRAPH = f.readlines()
    TREATMENTS_COLUMNS = list(df_clean.columns)
    for att in SUBPOPULATIONS:
        TREATMENTS_COLUMNS.remove(att)
    TREATMENTS_COLUMNS.remove('group1')
    TREATMENTS_COLUMNS.remove('group2')
    TREATMENTS_COLUMNS.remove(OUTCOME_COLUMN)
    treatments = create_value_dict(df_clean, TREATMENTS_COLUMNS)
    c_treatments = clean_treatments(df_clean, GRAPH, OUTCOME_COLUMN, treatments, cols_dict)
    calced_graphs = {}
    for itemset in tqdm(intresting_subpopulations, total=len(intresting_subpopulations)):
        r, calced_graphs = find_best_treatment(df=df_clean, item_set=itemset, output_att=OUTCOME_COLUMN, treatments=c_treatments, graph=GRAPH, cols_dict=cols_dict, graph_dict=calced_graphs)
        if not r:
            continue
        r_dict.append({"itemset": itemset, "treatment": r})
    pd.DataFrame(data=r_dict).to_csv(f"outputs/{PROJECT_DIRECTORY}/subpopulations_and_treatments.csv", index=False)
    # step 3 - find the best group with greedy algorithm"""
    df_clean = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/clean_data.csv")
    df_metadata = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/subpopulations_and_treatments.csv")
    results = calc_facts_metrics(data=df_clean, meta_data=df_metadata, OUTCOME_COLUMN=OUTCOME_COLUMN, graph=GRAPH, cols_dict=cols_dict)
    pd.DataFrame(results).to_csv(f"outputs/{PROJECT_DIRECTORY}/all_facts.csv", index=False)
    df_facts = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/all_facts.csv")
    find_group(df_clean, df_facts, PROJECT_DIRECTORY, OUTCOME_COLUMN)


algorithm()
