import pandas as pd
import numpy as np
from time import time
#from Tal.clean_meps import build_mini_df, SUBPOPULATIONS, TREATMENTS, OUTCOME_COLUMN, GRAPH
from Tal.clean_so import build_mini_df, SUBPOPULATIONS, TREATMENTS, OUTCOME_COLUMN
from algorithms.divexplorer.divexplorer import DivergenceExplorer
from algorithms.final_algorithm.find_best_treatment import find_best_treatment
from algorithms.final_algorithm.new_greedy import find_group, calc_facts_metrics
from tqdm import tqdm
import warnings
from dowhy import CausalModel
from Utils import getTreatmentCATE

warnings.filterwarnings("ignore")

PRECENT_CLEANING_THRESHOLD = 0.5
DIVEXPLORER_THRESHOLD_SUPPORT = 0.1
PROJECT_DIRECTORY = "so"
K = 5
LAMDA_VALUES = [0.0001, 0.00005, 0.00009]
CALCED_INTERSECTIONS = {}

GROUP_1 = [{"att": "group1", "value": 1}]
GROUP_2 = [{"att": "group2", "value": 1}]

def clean_treatments(df, GRAPH, OUTCOME_COLUMN, TREATMENTS):
    r = []
    for t in TREATMENTS:
        ate_result = getTreatmentCATE(df, GRAPH, t, OUTCOME_COLUMN)
        if ate_result != 0:
            r.append(t)
    return r


def algorithm():
    df_clean = build_mini_df()
    df_clean.to_csv(f"outputs/{PROJECT_DIRECTORY}/clean_data.csv", index=False)
    # step 1 - get best subpopulation by divexplorer
    df_clean = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/clean_data.csv")
    df_clean = df_clean.loc[(df_clean["group1"] == 1) | (df_clean["group2"] == 1)]
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
    c_treatments = clean_treatments(df_clean, GRAPH, OUTCOME_COLUMN, TREATMENTS)
    print(c_treatments)
    for itemset in tqdm(intresting_subpopulations, total=len(intresting_subpopulations)):
        r = find_best_treatment(df=df_clean, group1=GROUP_1, group2=GROUP_2, item_set=itemset,
                                output_att=OUTCOME_COLUMN, treatments=c_treatments, graph=GRAPH)
        if not r:
            continue
        r_dict.append({"itemset": itemset, "treatment": r})
    pd.DataFrame(data=r_dict).to_csv(f"outputs/{PROJECT_DIRECTORY}/subpopulations_and_treatments.csv", index=False)

    # step 3 - find the best group with greedy algorithm
    df_clean = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/clean_data.csv")
    df_metadata = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/subpopulations_and_treatments.csv")
    results = calc_facts_metrics(data=df_clean, group1=GROUP_1, group2=GROUP_2, meta_data=df_metadata, OUTCOME_COLUMN=OUTCOME_COLUMN, graph=GRAPH)
    pd.DataFrame(results).to_csv(f"outputs/{PROJECT_DIRECTORY}/all_facts.csv", index=False)
    df_facts = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/all_facts.csv")
    find_group(df_clean, df_facts, PROJECT_DIRECTORY, OUTCOME_COLUMN)


algorithm()