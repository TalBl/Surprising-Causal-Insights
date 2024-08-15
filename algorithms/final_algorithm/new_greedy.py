import pandas as pd
import numpy as np
from dowhy import CausalModel
import networkx as nx
import itertools
import ast
from tqdm import tqdm

PROJECT_DIRECTORY = "heart"
K = 5
LAMDA_VALUES = [0.0001, 0.00005, 0.00009]
CLEAN_DF_PATH = f"outputs/{PROJECT_DIRECTORY}/clean_data.csv"
PROTECTED_COLUMN = "Woman_Gender"
OUTCOME_COLUMN = "HeartDisease"
CALCED_INTERSECTIONS = {}
CLEAN_DF = pd.read_csv(CLEAN_DF_PATH)

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


def calc_cate(group: list, df: pd.DataFrame, outcome_column, att: list, val: list):
    filtered_df = df.copy()
    for d in group:
        filtered_df = filtered_df[filtered_df[d["att"]] == d['value']]
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
        return None,None
    estimands = model.identify_effect()
    causal_estimate_reg = model.estimate_effect(estimands,
                                                method_name="backdoor.linear_regression",
                                                target_units="ate",
                                                effect_modifiers=[],
                                                test_significance=True)

    p_val = causal_estimate_reg.test_stat_significance()['p_value']
    if p_val < 0.5:
        return causal_estimate_reg.value, p_val
    return None,None


def analyze_relation(data, group1, group2, itemset, treatment_d, size, size_group1, size_group2, support, std, diff_means):
    ate1, p_v1 = calc_cate(group1, data, OUTCOME_COLUMN, list(treatment_d.keys()), list(treatment_d.values()))
    ate2, p_v2 = calc_cate(group2, data, OUTCOME_COLUMN, list(treatment_d.keys()), list(treatment_d.values()))
    if ate1 and ate2:
        iscore = abs(ate1 - ate2)
        return {'itemset': itemset, 'treatment': treatment_d, 'ate1': ate1, 'ate2':ate2,
                'iscore': iscore, 'size_itemset': size, 'size_group1': size_group1, "size_group2": size_group2, "support": support,
                "ni_score": ni_score(iscore, LAMDA_VALUES[2]),
                "p_v1": p_v1, "p_v2": p_v2, "utility": ni_score(iscore, LAMDA_VALUES[2])*support, "std": std, "diff_means": diff_means}
    return {'itemset': itemset, 'treatment': treatment_d, 'ate1': ate1, 'ate2':ate2,
            'iscore': None, 'size_itemset': size, 'size_group1': size_group1, "size_group2": size_group2, "support": support,
            "ni_score": None,
            "p_v1": p_v1, "p_v2": p_v2, "utility": None, "std": std, "diff_means": diff_means}


def ni_score(x, lamda):
    return 1 - (1 / (np.exp(lamda * x)))


# Convert values to appropriate types
def convert_value(val):
    try:
        return int(val)
    except ValueError:
        return val


def parse_treatment(input_str):
    # Remove the outer parentheses and strip extra quotes if present
    s = ast.literal_eval(f"{input_str}")
    res_dict = {}
    if type(s) == dict:
        s = [s]
    for d in s:
        res_dict[d["att"]] = d["value"]
    return res_dict


def parse_itemset(itemset):
    elements = ast.literal_eval(f"{{{itemset[11:-2]}}}")  # Use ast.literal_eval to safely parse the set
    item_set = {}
    for element in elements:
        key, value = element.split('=')
        item_set[key] = convert_value(value)
    return item_set


def calc_facts_metrics(data, group1, group2, meta_data):
    n = data.shape[0]
    results = []
    for idx, (itemset, treatment) in meta_data.iterrows():
        item_set = parse_itemset(itemset)
        treatments_d = parse_treatment(treatment)
        population = data.copy()
        for key, value in item_set.items():
            population = population[population[key] == value]
        if population.shape[0] == 0:
            return None
        size = population.shape[0]
        support = size / n
        std = np.std(population[OUTCOME_COLUMN])
        df_group1 = population.copy()
        for d in group1:
            df_group1 = df_group1[df_group1[d["att"]]==d["value"]]
        df_group2 = population.copy()
        for d in group2:
            df_group2 = df_group2[df_group2[d["att"]]==d["value"]]
        size_group1 = df_group1.shape[0]
        size_group2 = df_group2.shape[0]
        diff_means = np.mean(df_group1[OUTCOME_COLUMN]) - np.mean(df_group2[OUTCOME_COLUMN])
        treated = population.copy()
        for key, value in treatments_d.items():
            treated = treated[treated[key] == value]
        if treated.shape[0] == 0:
            return None
        r = analyze_relation(population, group1, group2, item_set, treatments_d, size, size_group1, size_group2, support,
                             std, diff_means)
        if r:
            results.append(r)
    pd.DataFrame(results).to_csv(f"outputs/{PROJECT_DIRECTORY}/all_facts.csv", index=False)


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


def get_score(max_subpopulation, df_facts, group, attribute, alpha, K, lamda):
    intersection = 0
    checked_group = group.copy()
    checked_group.append(attribute)
    utility_sum = sum([i['utility'] for i in checked_group])
    utility = utility_sum / len(checked_group)
    if group:
        for pair in itertools.combinations(checked_group, 2):
            intersection += get_intersection(df_facts, pair[0], pair[1])
    f_intersection = ((max_subpopulation*len(checked_group)*len(checked_group)) - intersection) / (max_subpopulation*len(checked_group)*len(checked_group))
    score = (alpha * utility) + ((1 - alpha) * f_intersection)
    return {"utility_sum": utility_sum, "utility": utility, "intersection_sum": intersection,
                      "final_intersection": f_intersection, "score": score}


def greedy(df_clean, df_facts, max_subpopulation, alpha, lamda):
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
            score_dict = get_score(max_subpopulation, df_facts, group, group_rows, alpha, K, lamda)
            if score_dict and score_dict["score"] > max_score:
                max_score = score_dict["score"]
                curr_row = group_rows
                score_dictionary = score_dict
        group.append(curr_row)
        items.append(curr_row['itemset'])
        scores.append(score_dictionary)
        i += 1
    return group, scores


def find_group(df_clean, df_facts):
    df_facts = df_facts.dropna()
    max_subpopulation = max(df_facts['size_itemset'])
    for lamda in [0.00009]:
        for alpha in [0.5]:
            group, scores = greedy(df_clean, df_facts, max_subpopulation, alpha, lamda)
            df_calc = pd.concat(group, axis=1)
            transposed_df1 = df_calc.T
            transposed_df1.to_csv(f"outputs/{PROJECT_DIRECTORY}/find_k/{K}_{alpha}_{lamda}.csv", index=False)
            pd.DataFrame(scores).to_csv(f"outputs/{PROJECT_DIRECTORY}/scores/{K}_{alpha}_{lamda}.csv", index=False)

