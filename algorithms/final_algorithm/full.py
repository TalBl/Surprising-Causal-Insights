import pandas as pd
import numpy as np
from time import time
#from Tal.clean_meps import build_mini_df, SUBPOPULATIONS, create_value_dict, OUTCOME_COLUMN, TREATMENTS_COLUMNS
from Tal.clean_so import build_mini_df, SUBPOPULATIONS, create_value_dict, OUTCOME_COLUMN, TREATMENTS_COLUMNS
#from Tal.clean_meps_for_naive import build_mini_df, SUBPOPULATIONS, create_value_dict, OUTCOME_COLUMN, TREATMENTS_COLUMNS
from algorithms.divexplorer.divexplorer import DivergenceExplorer
from algorithms.final_algorithm.find_best_treatment import find_best_treatment
from algorithms.final_algorithm.new_greedy import find_group, calc_facts_metrics
from tqdm import tqdm
import concurrent.futures
import warnings
from dowhy import CausalModel
from Utils import getTreatmentCATE, changeDAG
import networkx as nx

warnings.filterwarnings("ignore")

PRECENT_CLEANING_THRESHOLD = 0.5
DIVEXPLORER_THRESHOLD_SUPPORT = 0.004
PROJECT_DIRECTORY = "so2"
K = 5
LAMDA_VALUES = [0.0001, 0.00005, 0.00009]
CALCED_INTERSECTIONS = {}
CALCED_GRAPHS = {}

GROUP_1 = [{"att": "group1", "value": 1}]
GROUP_2 = [{"att": "group2", "value": 1}]

def clean_treatments(df, GRAPH, OUTCOME_COLUMN, TREATMENTS, cols_dict):
    edges, cared_atts = [], []
    graphs_dict = {}
    atts = list(set([cols_dict[x['att']] for x in TREATMENTS.values()]))
    for att in atts:
        DAG_ = changeDAG(GRAPH, {"att": att}, cols_dict)
        edges = []
        check = False
        for line in DAG_:
            if '->' in line:
                if 'TempTreatment ->' in line:
                    check = True
                edges.append([line.split(" ->")[0].split("'")[1], line.split("-> ")[1].split(";'")[0]])
        if not check:
            continue
        causal_graph = nx.DiGraph()
        causal_graph.add_edges_from(edges)
        model = CausalModel(
            data=df,
            graph=causal_graph,
            treatment="TempTreatment",
            outcome=OUTCOME_COLUMN)
        estimands = model.identify_effect()
        if not estimands.no_directed_path:
            cared_atts.append(att)
            graphs_dict[att] = causal_graph
    clean_treats = {}
    for k, t in TREATMENTS.items():
        if t['att'] in cared_atts:
            clean_treats[k] = t
    return clean_treats, graphs_dict


def algorithm():
    df_clean, cols_dict = build_mini_df()
    df_clean.to_csv(f"outputs/{PROJECT_DIRECTORY}/clean_data.csv", index=False)
    # step 1 - get best subpopulation by divexplorer
    df_clean = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/clean_data.csv")
    fp_diver = DivergenceExplorer(df_clean)
    columns_to_ignore = ['RaceEthnicity_BlackorofAfricandescent=0', 'RaceEthnicity_EastAsian=0',
                         'RaceEthnicity_HispanicorLatino/Latina=0', 'RaceEthnicity_MiddleEastern=0',
                         'RaceEthnicity_NativeAmerican,PacificIslander,orIndigenousAustralian=0',
                         'RaceEthnicity_SouthAsian=0', 'RaceEthnicity_WhiteorofEuropeandescent=0']
    subgroups = fp_diver\
        .get_pattern_divergence(min_support=DIVEXPLORER_THRESHOLD_SUPPORT, quantitative_outcomes=[OUTCOME_COLUMN],
                                group_1_column="group1", group_2_column="group2", attributes=SUBPOPULATIONS, COLUMNS_TO_IGNORE=columns_to_ignore)\
        .sort_values(by=f"{OUTCOME_COLUMN}_div", ascending=False, ignore_index=True)
    subgroups = subgroups.dropna()
    subgroups.to_csv(f"outputs/{PROJECT_DIRECTORY}/interesting_subpopulations.csv", index=False)
    # step 2 - find the best treatment for each subpopulation
    df_clean = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/clean_data.csv")
    subgroups = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/interesting_subpopulations.csv")
    intresting_subpopulations = list(subgroups['itemset'])
    r_dict = []
    with open(f"data/so/causal_dag.txt", "r") as f:
        GRAPH = f.readlines()
    TREATMENTS_COLUMNS = list(set(df_clean.columns) - set(SUBPOPULATIONS + ['group1', 'group2', OUTCOME_COLUMN]))
    treatments = create_value_dict(df_clean, TREATMENTS_COLUMNS)
    c_treatments, calced_graphs = clean_treatments(df_clean, GRAPH, OUTCOME_COLUMN, treatments, cols_dict)
    """def run_in_parallel(itemsets):
        result_list = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(find_best_treatment, df_clean, x, OUTCOME_COLUMN, c_treatments, GRAPH, cols_dict, calced_graphs) for x in itemsets]
            for future in concurrent.futures.as_completed(futures):
                result_list.append(future.result())
        return result_list
        """

    results = []
    for itemset in tqdm(intresting_subpopulations, total=len(intresting_subpopulations)):
        r = find_best_treatment(df_clean, itemset, OUTCOME_COLUMN, c_treatments, GRAPH, cols_dict, calced_graphs)
        if r["treatment"]:
            results.append(r)
    pd.DataFrame(data=results).to_csv(f"outputs/{PROJECT_DIRECTORY}/subpopulations_and_treatments.csv", index=False)
    # step 3 - find the best group with greedy algorithm
    df_clean = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/clean_data.csv")
    df_metadata = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/subpopulations_and_treatments.csv")
    results = calc_facts_metrics(data=df_clean, meta_data=df_metadata, OUTCOME_COLUMN=OUTCOME_COLUMN, graph=GRAPH, cols_dict=cols_dict)
    pd.DataFrame(results).to_csv(f"outputs/{PROJECT_DIRECTORY}/all_facts.csv", index=False)
    df_facts = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/all_facts.csv")
    find_group(df_clean, df_facts, PROJECT_DIRECTORY, OUTCOME_COLUMN)


algorithm()
