import pandas as pd
from itertools import combinations
import ast
from Utils import getTreatmentCATE

def find_best_treatment(df: pd.DataFrame, item_set: str, output_att: str, treatments:dict, graph, cols_dict: dict, graph_dict: dict):
    elements = ast.literal_eval(f"{{{item_set[11:-2]}}}")  # Use ast.literal_eval to safely parse the set
    item_set = {}
    for element in elements:
        key, value = element.split('=')
        try:
            value = float(value)
        except:
            value = value
        item_set[key] = value
    for key, value in item_set.items():
        df = df[df[key] == value]
    # build df for each group
    df_group1 = df.loc[df['group1'] == 1]
    df_group2 = df.loc[df['group2'] == 1]
    item_set1_avg = df_group1[output_att].mean()
    item_set2_avg = df_group2[output_att].mean()
    is_avg_diff_positive = item_set1_avg - item_set2_avg > 0

    agreed_treatments = {}

    # First layer
    for k, t in treatments.items():
        f_res_group1 = getTreatmentCATE(df_group1, graph, t, output_att, cols_dict, graph_dict)
        f_res_group2 = getTreatmentCATE(df_group2, graph, t, output_att, cols_dict, graph_dict)
        if not f_res_group1 or not f_res_group2:
            continue
        if is_avg_diff_positive and f_res_group1 > f_res_group2:
            agreed_treatments[k] = f_res_group1 - f_res_group2
        if not is_avg_diff_positive and f_res_group1 < f_res_group2:
            agreed_treatments[k] = f_res_group1 - f_res_group2

    # Find the key with the maximum absolute value in agreed_item_set
    if not agreed_treatments:
        return {"itemset": item_set, "treatment": None}
    best_treatment = max(agreed_treatments, key=lambda k: abs(agreed_treatments[k]))

    # Second layer
    second_layer_agreed_treatments = {}
    for t1, t2 in combinations(agreed_treatments.keys(), 2):
        t = [treatments[t1], treatments[t2]]
        f_res_group1 = getTreatmentCATE(df_group1, graph, t, output_att, cols_dict)
        f_res_group2 = getTreatmentCATE(df_group2, graph, t, output_att, cols_dict)
        if not f_res_group1 or not f_res_group2:
            continue
        if is_avg_diff_positive and f_res_group1 > f_res_group2:
            second_layer_agreed_treatments["+".join([t1,t2])] = f_res_group1 - f_res_group2
        if not is_avg_diff_positive and f_res_group1 < f_res_group2:
            second_layer_agreed_treatments["+".join([t1,t2])] = f_res_group1 - f_res_group2

    if second_layer_agreed_treatments:
        best_treatment_second_layer = max(second_layer_agreed_treatments, key=lambda k: abs(second_layer_agreed_treatments[k]))
        return {"itemset": item_set, "treatment": best_treatment} if abs(agreed_treatments[best_treatment]) > abs(second_layer_agreed_treatments[best_treatment_second_layer]) else {"itemset": item_set, "treatment": best_treatment_second_layer}
    return {"itemset": item_set, "treatment": best_treatment}
