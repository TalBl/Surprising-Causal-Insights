from Tal.clean_acs import *
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import math
import time
from dowhy import CausalModel
import networkx as nx

#ATTS = ["Marital status", "Educational attainment", "Total person's income", "Sex"]
PROJECT_DIRECTORY = "acs"
K = 5
LAMDA_VALUES = [0.0001, 0.00005, 0.00009]
CLEAN_DF_PATH = "data/acs/acstest_mini.csv"
PROTECTED_COLUMN = "Sex"
OUTCOME_COLUMN = "Total person's income"
CLEAN_DF = pd.read_csv(CLEAN_DF_PATH)
#CLEAN_DF = CLEAN_DF[ATTS]
CALCED_INTERSECTIONS = {}


def get_intersection(df_facts, att1, att2):
    if att1[0] == att2[0] and att1[1] == att2[1]:
        row = df_facts.loc[(df_facts['subpopulation'] == att1[0])
                           & (df_facts['value_population'] == att1[1])]
        return row['size'].values[0]
    elif att1[0] == att2[0] and att1[1] != att2[1]:
        return 0
    else:
        row = df_facts.loc[(df_facts['subpopulation'] == att1[0])
                           & (df_facts['value_population'] == att1[1])
                           & (df_facts['treatment'] == att2[0])
                           & (df_facts['value'] == att2[1])]
        if row.shape[0] == 0:
            row = df_facts.loc[(df_facts['subpopulation'] == att2[0])
                               & (df_facts['value_population'] == att2[1])
                               & (df_facts['treatment'] == att1[0])
                               & (df_facts['value'] == att1[1])]
        if row.shape[0] == 0:
            if "_".join([att1[0], att1[1], att2[0], att2[1]]) in CALCED_INTERSECTIONS:
                return CALCED_INTERSECTIONS["_".join([att1[0], att1[1], att2[0], att2[1]])]
            if "_".join([att2[0], att2[1], att1[0], att1[1]]) in CALCED_INTERSECTIONS:
                return CALCED_INTERSECTIONS["_".join([att2[0], att2[1], att1[0], att1[1]])]
            r = CLEAN_DF.loc[(CLEAN_DF[att1[0]] == att1[1])
                             & (CLEAN_DF[att2[0]] == att2[1])].shape[0]
            CALCED_INTERSECTIONS["_".join([att1[0], att1[1], att2[0], att2[1]])] = r
            return r
    res = row['size_treated'].values[0]
    return res


def get_score(max_subpopulation, df_facts, group, alpha, K, lamda):
    intersection = 0
    checked_group = group.copy()
    utility_sum = 0
    for idx, i in checked_group.iterrows():
        utility_sum += (i['ni_score3'] * i['support'])
    utility = utility_sum / K
    for pair in itertools.combinations(checked_group.values.tolist(), 2):
        intersection += get_intersection(df_facts, pair[0], pair[1])
    f_intersection = ((max_subpopulation*len(checked_group)*len(checked_group)) - intersection) / (max_subpopulation*len(checked_group)*len(checked_group))
    score = (alpha * utility) + ((1 - alpha) * f_intersection)
    return pd.Series({"utility_sum": utility_sum, "utility": utility, "intersection_sum": intersection,
                      "final_intersection": f_intersection, "score": score})


def greedy(df_facts, max_subpopulation, alpha, lamda):
    max_score = 0
    res_group = pd.DataFrame()
    lst = [x for x in range(df_facts.shape[0])]
    for combination in tqdm(itertools.combinations(lst, K), total=1906884):
        wanted_group = df_facts.iloc[list(combination)]
        score_dict = get_score(max_subpopulation, df_facts, wanted_group, alpha, K, lamda)
        if score_dict["score"] > max_score:
            max_score = score_dict["score"]
            res_group = wanted_group
            d = score_dict
    return res_group


def find_group(clean_df):
    df_facts = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/all_facts_mini.csv")
    df_facts = df_facts.dropna()
    max_subpopulation = max(df_facts['size'])
    for lamda in [0.0001]:
        for alpha in [0.5]:
            group = greedy(df_facts, max_subpopulation, alpha, lamda)
            transposed_df1 = group.T
            transposed_df1.to_csv(f"outputs/{PROJECT_DIRECTORY}/find_k/naive_{K}_{alpha}_{lamda}.csv", index=False)


if __name__ == '__main__':
    start = time.time()
    df = pd.read_csv(CLEAN_DF_PATH)
    #df = df[ATTS]
    end1 = time.time()
    find_group(df)
