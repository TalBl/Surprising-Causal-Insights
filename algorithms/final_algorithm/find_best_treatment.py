import pandas as pd
import numpy as np
from itertools import combinations
from typing import Dict, List
from dowhy import CausalModel
import networkx as nx
import ast

OUTCOME_COLUMN = "HeartDisease"


def so_syn_dag():
    G = nx.DiGraph()
    G.add_edges_from([
        ("Developer, back-end", OUTCOME_COLUMN),
        ("Woman", OUTCOME_COLUMN),
        ("Country", OUTCOME_COLUMN),
        ("Age", OUTCOME_COLUMN),
        ("MentalHealth", OUTCOME_COLUMN),
        ("White", OUTCOME_COLUMN),
        ("Asian", OUTCOME_COLUMN),
        ("Black", OUTCOME_COLUMN),
        ("Master’s degree (M.A., M.S., M.Eng., MBA, etc.)", OUTCOME_COLUMN),
        ("Bachelor’s degree (B.A., B.S., B.Eng., etc.)", OUTCOME_COLUMN),
        ("Primary/elementary school", OUTCOME_COLUMN),
        ("new_treatment", OUTCOME_COLUMN)
    ])
    return G

DAG_d = so_syn_dag()


def calc_cate(group: list, df: pd.DataFrame, outcome_column, treatment: list):
    filtered_df = df.copy()
    for d in group:
        filtered_df = filtered_df[filtered_df[d["att"]] == d['value']]
    if filtered_df.shape[0] == 0:
        return None
    # Initialize the new_treatment column to 1
    filtered_df["new_treatment"] = 1
    if type(treatment) == dict:
        treatment = [treatment]
    for d in treatment:
        filtered_df["new_treatment"] = filtered_df["new_treatment"] & filtered_df[d["att"]].apply(lambda x: 1 if x == d["value"] else 0).astype(int)

    model = CausalModel(
        data=filtered_df,
        graph=DAG_d,
        treatment="new_treatment",
        outcome=outcome_column)
    values = list(set(filtered_df["new_treatment"]))
    if len(values) == 1:
        return None
    estimands = model.identify_effect()
    causal_estimate_reg = model.estimate_effect(estimands,
                                                method_name="backdoor.linear_regression",
                                                target_units="ate",
                                                effect_modifiers=[],
                                                test_significance=True)

    p_val = causal_estimate_reg.test_stat_significance()['p_value']
    if p_val < 0.5:
        return causal_estimate_reg.value
    return None

def calc_att_avg(group: List, df: pd.DataFrame, output_att: str) -> float:
    filtered_df = df.copy()
    for d in group:
        filtered_df = filtered_df[filtered_df[d["att"]] == d['value']]
    return filtered_df[output_att].mean()


def find_best_treatment(df: pd.DataFrame, group1: list, group2: list, item_set: str, output_att: str, treatments:list):
    elements = ast.literal_eval(f"{{{item_set[11:-2]}}}")  # Use ast.literal_eval to safely parse the set
    item_set = {}
    for element in elements:
        key, value = element.split('=')
        try:
            value = int(value)
        except:
            value = value
        item_set[key] = value
    for key, value in item_set.items():
        df = df[df[key] == value]
    item_set1_avg = calc_att_avg(group1, df, output_att)
    item_set2_avg = calc_att_avg(group2, df, output_att)
    is_avg_diff_positive = item_set1_avg - item_set2_avg > 0

    agreed_treatments = {}

    # First layer
    for t in treatments:
        f_res_group1 = calc_cate(group1, df, output_att, t)
        f_res_group2 = calc_cate(group2, df, output_att, t)
        if not f_res_group1 or not f_res_group2:
            continue
        if is_avg_diff_positive and f_res_group1 > f_res_group2:
            agreed_treatments[str(t)] = f_res_group1 - f_res_group2
        if not is_avg_diff_positive and f_res_group1 < f_res_group2:
            agreed_treatments[str(t)] = f_res_group1 - f_res_group2

    # Find the key with the maximum absolute value in agreed_item_set
    if not agreed_treatments:
        return None
    best_treatment = max(agreed_treatments, key=lambda k: abs(agreed_treatments[k]))

    # Second layer
    second_layer_agreed_treatments = {}
    for t1, t2 in combinations(agreed_treatments.keys(), 2):
        t = [ast.literal_eval(t1), ast.literal_eval(t2)]
        f_res_group1 = calc_cate(group1, df, output_att, t)
        f_res_group2 = calc_cate(group2, df, output_att, t)
        if not f_res_group1 or not f_res_group2:
            continue
        if is_avg_diff_positive and f_res_group1 > f_res_group2:
            second_layer_agreed_treatments[str(t)] = f_res_group1 - f_res_group2
        if not is_avg_diff_positive and f_res_group1 < f_res_group2:
            second_layer_agreed_treatments[str(t)] = f_res_group1 - f_res_group2

    if second_layer_agreed_treatments:
        best_treatment_second_layer = max(second_layer_agreed_treatments, key=lambda k: abs(second_layer_agreed_treatments[k]))
        return best_treatment if abs(agreed_treatments[best_treatment]) > abs(second_layer_agreed_treatments[best_treatment_second_layer]) else best_treatment_second_layer
    return ast.literal_eval(best_treatment)
