import pandas as pd
import numpy as np
from dowhy import CausalModel
import networkx as nx
import itertools
import ast
from tqdm import tqdm
from Utils import getTreatmentCATE

K = 5
LAMDA_VALUES = [0.0001, 0.00005, 0.001]
CALCED_INTERSECTIONS = {}


def analyze_relation(data, group1, group2, itemset, treatment_d, size, size_group1, size_group2, support, std, diff_means, OUTCOME_COLUMN, graph, cols_dict):
    ate1, graph_dict = getTreatmentCATE(group1, graph, treatment_d, OUTCOME_COLUMN, cols_dict)
    ate2, graph_dict = getTreatmentCATE(group2, graph, treatment_d, OUTCOME_COLUMN, cols_dict, graph_dict)
    # getTreatmentCATE(df_group1, graph, t, output_att, cols_dict)
    if ate1 and ate2:
        iscore = abs(ate1 - ate2)
        return {'itemset': itemset, 'treatment': treatment_d, 'ate1': ate1, 'ate2':ate2,
                'iscore': iscore, 'size_itemset': size, 'size_group1': size_group1, "size_group2": size_group2, "support": support,
                "ni_score": ni_score(iscore, LAMDA_VALUES[2]),
                "utility": ni_score(iscore, LAMDA_VALUES[2]), "std": std, "diff_means": diff_means}
    return {'itemset': itemset, 'treatment': treatment_d, 'ate1': ate1, 'ate2':ate2,
            'iscore': None, 'size_itemset': size, 'size_group1': size_group1, "size_group2": size_group2, "support": support,
            "ni_score": None, "utility": None, "std": std, "diff_means": diff_means}


def ni_score(x, lamda):
    return 1 - (1 / (np.exp(lamda * x)))


# Convert values to appropriate types
def convert_value(val):
    try:
        return float(val)
    except ValueError:
        return val


def parse_treatment(input_str):
    # Remove the outer parentheses and strip extra quotes if present
    treats = input_str.split("+")
    res_list = []
    for t in treats:
        val = convert_value(t.split("_")[-1])
        value = lambda x, v=val: 1 if pd.notna(x) and x == v else 0
        att = t.replace("_"+t.split("_")[-1], "")
        res_list.append({"att": att, "value": value, "val_specified": val})
    return res_list


def parse_itemset(itemset):
    elements = ast.literal_eval(f"{{{itemset[11:-2]}}}")  # Use ast.literal_eval to safely parse the set
    item_set = {}
    for element in elements:
        key, value = element.split('=')
        item_set[key] = convert_value(value)
    return item_set


def calc_facts_metrics(data, meta_data, OUTCOME_COLUMN, graph, cols_dict, graph_dict=None):
    n = data.shape[0]
    results = []
    for idx, (itemset, treatment) in meta_data.iterrows():
        item_set = parse_itemset(itemset)
        treatments_d = parse_treatment(treatment)
        population = data.copy()
        for key, value in item_set.items():
            population = population[population[key] == value]
        if population.shape[0] == 0:
            continue
        size = population.shape[0]
        support = size / n
        std = np.std(population[OUTCOME_COLUMN])
        df_group1 = population.copy()
        df_group1 = df_group1.loc[df_group1['group1'] == 1]
        df_group2 = population.copy()
        df_group2 = df_group2.loc[df_group2['group2'] == 1]
        size_group1 = df_group1.shape[0]
        size_group2 = df_group2.shape[0]
        diff_means = np.mean(df_group1[OUTCOME_COLUMN]) - np.mean(df_group2[OUTCOME_COLUMN])
        treated = population.copy()
        for d in treatments_d:
            treated = treated[treated[d['att']] == d['val_specified']]
        if treated.shape[0] == 0:
            continue
        r = analyze_relation(population, df_group1, df_group2, item_set, treatments_d, size, size_group1, size_group2, support,
                             std, diff_means, OUTCOME_COLUMN, graph, cols_dict)
        if r:
            results.append(r)
    return results


def get_intersection(df_facts, att1, att2, df_clean):
    if "_".join([att1['itemset'], att2['itemset']]) in CALCED_INTERSECTIONS:
        return CALCED_INTERSECTIONS["_".join([att1['itemset'], att2['itemset']])]
    if "_".join([att2['itemset'], att1['itemset']]) in CALCED_INTERSECTIONS:
        return CALCED_INTERSECTIONS["_".join([att2['itemset'], att1['itemset']])]
    item_set1 = ast.literal_eval(att1['itemset'])
    item_set2 = ast.literal_eval(att2['itemset'])
    population = df_clean.copy()
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


def get_score(max_subpopulation, df_facts, group, attribute, alpha, K, lamda, df_clean, OUTCOME_COLUMN):
    intersection = 0
    checked_group = group.copy()
    checked_group.append(attribute)
    utility_sum = sum([i['utility'] for i in checked_group])
    utility = utility_sum / len(checked_group)
    if group:
        for pair in itertools.combinations(checked_group, 2):
            intersection += get_intersection(df_facts, pair[0], pair[1], df_clean)
    f_intersection = ((max_subpopulation*len(checked_group)*len(checked_group)) - intersection) / (max_subpopulation*len(checked_group)*len(checked_group))
    score = (alpha * utility) + ((1 - alpha) * f_intersection)
    return {"utility_sum": utility_sum, "utility": utility, "intersection_sum": intersection,
                      "final_intersection": f_intersection, "score": score}


def greedy(df_clean, df_facts, max_subpopulation, alpha, lamda, OUTCOME_COLUMN):
    print(f"LOOKING FOR GROUP alpha={alpha} and lamda={lamda}")
    i = 0
    group = []
    scores = []
    items = []
    while i < K:
        max_score = 0
        curr_row = None
        score_dictionary = None
        for indexes, group_rows in tqdm(df_facts.iterrows(), total=df_facts.shape[0]):
            if group_rows['itemset'] in items:
                continue
            score_dict = get_score(max_subpopulation, df_facts, group, group_rows, alpha, K, lamda, df_clean, OUTCOME_COLUMN)
            if score_dict and score_dict["score"] > max_score:
                max_score = score_dict["score"]
                curr_row = group_rows
                score_dictionary = score_dict
        group.append(curr_row)
        items.append(curr_row['itemset'])
        scores.append(score_dictionary)
        i += 1
    return group, scores


def find_group(df_clean, df_facts, PROJECT_DIRECTORY, OUTCOME_COLUMN):
    df_facts = df_facts.dropna()
    max_subpopulation = max(df_facts['size_itemset'])
    df_facts = df_facts[(df_facts['size_group1']>39) & (df_facts['size_group2']>39)]
    for lamda in [0.00009]:
        for alpha in [0.5]:
            group, scores = greedy(df_clean, df_facts, max_subpopulation, alpha, lamda, OUTCOME_COLUMN)
            df_calc = pd.concat(group, axis=1)
            transposed_df1 = df_calc.T
            transposed_df1.to_csv(f"outputs/{PROJECT_DIRECTORY}/find_k/{K}_{alpha}_{lamda}.csv", index=False)
            pd.DataFrame(scores).to_csv(f"outputs/{PROJECT_DIRECTORY}/scores/{K}_{alpha}_{lamda}.csv", index=False)

