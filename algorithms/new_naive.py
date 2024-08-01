import pandas as pd
import numpy as np
from dowhy import CausalModel
import networkx as nx
import itertools
import ast
from tqdm import tqdm
from time import time

PROJECT_DIRECTORY = "so"
K = 5
LAMDA_VALUES = [0.0001, 0.00005, 0.00009]
CLEAN_DF_PATH = f"../outputs/{PROJECT_DIRECTORY}/clean_data.csv"
PROTECTED_COLUMN = "Woman_Gender"
OUTCOME_COLUMN = "ConvertedCompYearly"
CLEAN_DF = pd.read_csv(CLEAN_DF_PATH)
CALCED_INTERSECTIONS = {}

def so_syn_dag():
    G = nx.DiGraph()
    G.add_edges_from([
        ("EdLevel", "ConvertedCompYearly"),
        ("YearsCodePro", "ConvertedCompYearly"),
        ("Country", "ConvertedCompYearly"),
        ("Age", "ConvertedCompYearly"),
        ("WorkExp", "ConvertedCompYearly"),
        ("Data scientist or machine learning specialist_DevType", "ConvertedCompYearly"),
        ("Developer, full-stack_DevType", "ConvertedCompYearly"),
        ("Engineer, data_DevType", "ConvertedCompYearly"),
        ("C#_LanguageHaveWorkedWith", "ConvertedCompYearly"),
        ("Go_LanguageHaveWorkedWith", "ConvertedCompYearly"),
        ("Java_LanguageHaveWorkedWith", "ConvertedCompYearly"),
        ("Julia_LanguageHaveWorkedWith", "ConvertedCompYearly"),
        ("Python_LanguageHaveWorkedWith", "ConvertedCompYearly"),
        ("Woman_Gender", "ConvertedCompYearly"),
        ("Man_Gender", "ConvertedCompYearly"),
        ("African_Ethnicity", "ConvertedCompYearly"),
        ("Black_Ethnicity", "ConvertedCompYearly"),
        ("Asian_Ethnicity", "ConvertedCompYearly"),
        ("White_Ethnicity", "ConvertedCompYearly"),
        ("new_treatment", "ConvertedCompYearly")
    ])
    return G


DAG_d = so_syn_dag()


def get_intersection(df_facts, att1, att2):
    if "_".join([att1['itemset'], att2['itemset']]) in CALCED_INTERSECTIONS:
        return CALCED_INTERSECTIONS["_".join([att1['itemset'], att2['itemset']])]
    if "_".join([att2['itemset'], att1['itemset']]) in CALCED_INTERSECTIONS:
        return CALCED_INTERSECTIONS["_".join([att2['itemset'], att1['itemset']])]
    item_set1 = ast.literal_eval(att1['itemset'])
    item_set2 = ast.literal_eval(att2['itemset'])
    population = CLEAN_DF.copy()
    for key, value in item_set1.items():
        population = population[population[key] == value]
        if population.shape[0] == 0:
            CALCED_INTERSECTIONS["_".join([att1['itemset'], att2['itemset']])] = 0
            return 0
    for key, value in item_set2.items():
        population = population[population[key] == value]
    if population.shape[0] == 0:
        CALCED_INTERSECTIONS["_".join([att1['itemset'], att2['itemset']])] = 0
        return 0
    r = population.shape[0]
    CALCED_INTERSECTIONS["_".join([att1['itemset'], att2['itemset']])] = r
    return r


def get_score(max_subpopulation, df_facts, group, alpha, K, lamda):
    intersection = 0
    checked_group = group.copy()
    utility_sum = sum(checked_group['utility'])
    utility = utility_sum / len(checked_group)
    for pair in itertools.combinations(checked_group.iterrows(), 2):
        intersection += get_intersection(df_facts, pair[0][1], pair[1][1])
    f_intersection = ((max_subpopulation*len(checked_group)*len(checked_group)) - intersection) / (max_subpopulation*len(checked_group)*len(checked_group))
    score = (alpha * utility) + ((1 - alpha) * f_intersection)
    return {"utility_sum": utility_sum, "utility": utility, "intersection_sum": intersection,
            "final_intersection": f_intersection, "score": score}


def greedy(df_facts, max_subpopulation, alpha, lamda):
    max_score = 0
    res_group = pd.DataFrame()
    lst = [x for x in range(df_facts.shape[0])]
    for combination in tqdm(itertools.combinations(lst, K)):
        wanted_group = df_facts.iloc[list(combination)]
        score_dict = get_score(max_subpopulation, df_facts, wanted_group, alpha, K, lamda)
        if score_dict["score"] > max_score:
            max_score = score_dict["score"]
            res_group = wanted_group
            d = score_dict
    return res_group


def find_group(clean_df):
    df_facts = pd.read_csv(f"../outputs/{PROJECT_DIRECTORY}/all_facts.csv")
    df_facts = df_facts.dropna()
    max_subpopulation = max(df_facts['size'])
    for lamda in [0.0001]:
        for alpha in [0.5]:
            group = greedy(df_facts, max_subpopulation, alpha, lamda)
            group.to_csv(f"../outputs/{PROJECT_DIRECTORY}/find_k/naive_{K}_{alpha}_{lamda}.csv", index=False)


if __name__ == '__main__':
    start = time()
    df = pd.read_csv(CLEAN_DF_PATH)
    #df = df[ATTS]
    end1 = time()
    find_group(df)