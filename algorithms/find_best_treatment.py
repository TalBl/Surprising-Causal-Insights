import pandas as pd
import numpy as np
from itertools import combinations
from typing import Dict, List
from dowhy import CausalModel
import networkx as nx
import ast

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


def calc_cate(df: pd.DataFrame, item_set: Dict, outcome_column, att: List, val: List):
    filtered_df = df.copy()
    for key, value in item_set.items():
        filtered_df = filtered_df[filtered_df[key] == value]
    if filtered_df.shape[0] == 0:
        return None
    # Initialize the new_treatment column to 1
    filtered_df["new_treatment"] = 1
    for attribute, desired_value in zip(att, val):
        filtered_df["new_treatment"] = filtered_df["new_treatment"] & filtered_df[attribute].apply(lambda x: 1 if x == desired_value else 0).astype(int)

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
    if p_val < 0.05:
        return causal_estimate_reg.value
    return None

def calc_att_avg(df: pd.DataFrame, item_set: Dict, output_att: str) -> float:
    """
    Calculate the average income of a given item set.
    """
    filtered_df = df.copy()
    for key, value in item_set.items():
        filtered_df = filtered_df[filtered_df[key] == value]
    return filtered_df[output_att].mean()


def find_best_treatment(df: pd.DataFrame, item_set: str, protected_att: str, protected_val, output_att: str, treatments:list):
    elements = ast.literal_eval(f"{{{item_set[11:-2]}}}")  # Use ast.literal_eval to safely parse the set
    item_set = {}
    for element in elements:
        key, value = element.split('=')
        try:
            value = int(value)
        except:
            value = value
        item_set[key] = value
    item_set_avg = calc_att_avg(df, item_set, output_att)
    item_set_with_protected = {**item_set, protected_att: protected_val}
    item_set_with_protected_avg = calc_att_avg(df, item_set_with_protected, output_att)
    is_avg_diff_positive = item_set_avg - item_set_with_protected_avg > 0

    relevant_treatments = treatments - item_set.keys()

    agreed_item_set = {}

    # First layer
    for att in relevant_treatments:
        att_values = df[att].unique() # Get all the unique values of the attribute
        for att_val in att_values:
            f_res = calc_cate(df, item_set, output_att, [att], [att_val])
            f_protected_res = calc_cate(df, item_set_with_protected, output_att, [att], [att_val])
            if not f_res or not f_protected_res:
                continue
            if is_avg_diff_positive and f_res > f_protected_res:
                agreed_item_set[(att, att_val)] = f_res - f_protected_res
            if not is_avg_diff_positive and f_res < f_protected_res:
                agreed_item_set[(att, att_val)] = f_res - f_protected_res

    # Find the key with the maximum absolute value in agreed_item_set
    best_treatment = max(agreed_item_set, key=lambda k: abs(agreed_item_set[k]))

    # Second layer
    second_layer_agreed_item_set = {}
    for couple in combinations(agreed_item_set.keys(), 2):
        tuple1, tuple2 = couple
        att1, att_val1 = tuple1
        att2, att_val2 = tuple2
        f_res = calc_cate(df, item_set, output_att, [att1, att2], [att_val1, att_val2])
        f_protected_res = calc_cate(df, item_set_with_protected, output_att, [att1, att2], [att_val1, att_val2])
        if not f_res or not f_protected_res:
            continue
        if is_avg_diff_positive and f_res > f_protected_res:
            second_layer_agreed_item_set[((att1, att2), (att_val1, att_val2))] = f_res - f_protected_res
        if not is_avg_diff_positive and f_res < f_protected_res:
            second_layer_agreed_item_set[((att1, att2), (att_val1, att_val2))] = f_res - f_protected_res

    if second_layer_agreed_item_set:
        best_treatment_second_layer = max(second_layer_agreed_item_set, key=lambda k: abs(second_layer_agreed_item_set[k]))
        return best_treatment if abs(agreed_item_set[best_treatment]) > abs(second_layer_agreed_item_set[best_treatment_second_layer]) else best_treatment_second_layer
    return best_treatment
