import pandas as pd
import numpy as np
from dowhy import CausalModel
import networkx as nx
import itertools
import ast
from tqdm import tqdm

PROJECT_DIRECTORY = "so"
K = 5
LAMDA_VALUES = [0.0001, 0.00005, 0.00009]
CLEAN_DF_PATH = f"../outputs/{PROJECT_DIRECTORY}/clean_data.csv"
PROTECTED_COLUMN = "Woman_Gender"
OUTCOME_COLUMN = "ConvertedCompYearly"
CALCED_INTERSECTIONS = {}
CLEAN_DF = pd.read_csv(CLEAN_DF_PATH)

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


def calc_cate(df, treatments, values, outcome):
    filtered_df = df.copy()
    filtered_df["new_treatment"] = 1
    for attribute, desired_value in zip(treatments, values):
        filtered_df["new_treatment"] = filtered_df["new_treatment"] & filtered_df[attribute].apply(lambda x: 1 if x == desired_value else 0).astype(int)

    model = CausalModel(
        data=filtered_df,
        graph=DAG_d,
        treatment="new_treatment",
        outcome=outcome)
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
    if p_val < 0.05:
        return causal_estimate_reg.value,p_val
    return None,None


def analyze_relation(data, itemset, treatment_d, size, size_protected, size_treated, support, std, diff_means):
    protected_population = data[data[PROTECTED_COLUMN] == 1]
    ate, p_v1 = calc_cate(data, list(treatment_d.keys()), list(treatment_d.values()), OUTCOME_COLUMN)
    ate_p, p_v2 = calc_cate(protected_population, list(treatment_d.keys()), list(treatment_d.values()), OUTCOME_COLUMN)
    if ate and ate_p:
        iscore = (max(ate, ate_p) - min(ate, ate_p))
        return {'itemset': itemset, 'treatment': treatment_d, 'ate': ate, 'ate_p':ate_p,
                'iscore': iscore, 'size': size, 'size_protected': size_protected, "size_treated": size_treated, "support": support,
                "ni_score1": ni_score(iscore, LAMDA_VALUES[0]), "ni_score2": ni_score(iscore, LAMDA_VALUES[1]), "ni_score3": ni_score(iscore, LAMDA_VALUES[2]),
                "p_v1": p_v1, "p_v2": p_v2, "utility": ni_score(iscore, LAMDA_VALUES[2])*support, "std": std, "diff_means": diff_means}
    return {'itemset': itemset, 'treatment': treatment_d, 'ate': ate, 'ate_p':ate_p,
            'iscore': None, 'size': size, 'size_protected': size_protected, "size_treated": size_treated, "support": support,
            "ni_score1": None, "ni_score2": None, "ni_score3": None, "p_v1": None, "p_v2": None, "utility": None, "std": std, "diff_means": diff_means}


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
    keys, values = s
    keys = list(keys) if type(keys)==tuple else [keys]
    values = list(values) if type(values) == tuple else [values]
    for idx in range(len(keys)):
        value, key = values[idx], keys[idx]
        res_dict[key] = value
    return res_dict


def parse_itemset(itemset):
    elements = ast.literal_eval(f"{{{itemset[11:-2]}}}")  # Use ast.literal_eval to safely parse the set
    item_set = {}
    for element in elements:
        key, value = element.split('=')
        item_set[key] = convert_value(value)
    return item_set


def calc_facts_metrics(data, meta_data):
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
        df_protected = population[population[PROTECTED_COLUMN] == 1]
        size_protected = df_protected.shape[0]
        diff_means = np.mean(population[OUTCOME_COLUMN]) - np.mean(df_protected[OUTCOME_COLUMN])
        treated = population.copy()
        for key, value in treatments_d.items():
            treated = treated[treated[key] == value]
        if treated.shape[0] == 0:
            return None
        size_treated = treated.shape[0]
        r = analyze_relation(population, item_set, treatments_d, size, size_protected, size_treated, support, std, diff_means)
        if r:
            results.append(r)
    pd.DataFrame(results).to_csv(f"../outputs/{PROJECT_DIRECTORY}/all_facts.csv", index=False)


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
        print(f"Add row itemset={curr_row['itemset']} treatment={curr_row['treatment']}")
        print(f" Now score is: {max_score}")
        i += 1
    return group, scores


def find_group(df_clean, df_facts):
    df_facts = df_facts.dropna()
    max_subpopulation = max(df_facts['size'])
    for lamda in [0.00009]:
        for alpha in [0.5]:
            group, scores = greedy(df_clean, df_facts, max_subpopulation, alpha, lamda)
            df_calc = pd.concat(group, axis=1)
            transposed_df1 = df_calc.T
            transposed_df1.to_csv(f"../outputs/{PROJECT_DIRECTORY}/find_k/{K}_{alpha}_{lamda}.csv", index=False)
            pd.DataFrame(scores).to_csv(f"../outputs/{PROJECT_DIRECTORY}/scores/{K}_{alpha}_{lamda}.csv", index=False)

