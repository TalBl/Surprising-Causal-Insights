from Tal.clean_acs import *
#from Tal.clean_so import *
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import math
import time
from dowhy import CausalModel
import networkx as nx
from scipy.stats import wasserstein_distance

PROJECT_DIRECTORY = "acs"
K = 5
LAMDA_VALUES = [0.0001, 0.00005, 0.00009]
CLEAN_DF_PATH = "../data/acs/acstest.csv"
PROTECTED_COLUMN = "Sex"
OUTCOME_COLUMN = "Total person's income"
CLEAN_DF = pd.read_csv(CLEAN_DF_PATH)
CALCED_INTERSECTIONS = {}

intersecction_list = []


def all_possible_relations(df_columns, data, outcome_column, protected_column):
    list_subpopulations = []
    tmp_lst = []
    for col in df_columns:
        if col == protected_column:
            continue
        values = set(data[col])
        if outcome_column != col:
            tmp_lst.append([col, values])
    for col, values in tmp_lst:
        for col2, values2 in tmp_lst:
            if col != col2:
                list_subpopulations.append([col, col2, outcome_column, values, values2])
    return list_subpopulations

def acs_syn_dag():
    G = nx.DiGraph()
    G.add_edges_from([
        ("Sex", "Total person's income"),
        ("Age", "Total person's income"),
        ("Citizenship status", "Total person's income"),
        ("Educational attainment", "Total person's income"),
        ("With a disability", "Total person's income"),
        ("Race/Ethnicity", "Total person's income"),
        ("When last worked", "Total person's income"),
        ("new_treatment", "Total person's income"),
        ("Marital status", "Total person's income"),
    ])
    return G


def so_syn_dag():
    G = nx.DiGraph()
    G.add_edges_from([
        ("Country", "ConvertedSalary"),
        ("Employment", "ConvertedSalary"),
        ("CompanySize", "ConvertedSalary"),
        ("YearsCoding", "ConvertedSalary"),
        ("Gender", "ConvertedSalary"),
        ("Age", "ConvertedSalary"),
        ("new_treatment", "ConvertedSalary")
    ])
    return G


DAG_d = acs_syn_dag()


def calc_cate(data, treatment, value, outcome):
    if data.shape[0] == 0:
        return None, None
    data_c = data.copy(deep=True)
    data_c["new_treatment"] = data_c[treatment].apply(lambda x: 1 if x == value else 0).astype(int)
    model = CausalModel(
        data=data_c,
        graph=DAG_d,
        treatment="new_treatment",
        outcome=OUTCOME_COLUMN)
    values = list(set(data_c["new_treatment"]))
    if len(values) == 1:
        return None, None
    estimands = model.identify_effect()
    causal_estimate_reg = model.estimate_effect(estimands,
                                                method_name="backdoor.linear_regression",
                                                target_units="ate",
                                                effect_modifiers=[],
                                                test_significance=True)

    p_val = causal_estimate_reg.test_stat_significance()['p_value']
    if p_val < 0.1:
        return causal_estimate_reg.value, p_val
    return None, p_val


def analyze_relation(data, subpopulation, value_popu, treatment, outcome, func_for_protected, value, size, size_protected, size_treated, support, std, kl, emd, diff_means):
    protected_population = func_for_protected(data)
    ate, p_v1 = calc_cate(data, treatment, value, outcome)
    ate_p, p_v2 = calc_cate(protected_population, treatment, value, outcome)
    if ate and ate_p:
        iscore = (max(ate, ate_p) - min(ate, ate_p))
        return {'subpopulation': subpopulation, 'value_population': value_popu, 'treatment': treatment, 'value': value, 'outcome': outcome, 'ate': ate, 'ate_p':ate_p,
                'iscore': iscore, 'size': size, 'size_protected': size_protected, "size_treated": size_treated, "support": support,
                "ni_score1": ni_score(iscore, LAMDA_VALUES[0]), "ni_score2": ni_score(iscore, LAMDA_VALUES[1]), "ni_score3": ni_score(iscore, LAMDA_VALUES[2]),
                "p_v1": p_v1, "p_v2": p_v2, "utility": ni_score(iscore, LAMDA_VALUES[2])*support, "std": std, "kl": kl, "emd": emd, "diff_means": diff_means}
    return {'subpopulation': subpopulation, 'value_population': value_popu, 'treatment': treatment, 'value': value, 'outcome': outcome, 'ate': ate, 'ate_p':ate_p,
            'iscore': None, 'size': size, 'size_protected': size_protected, "size_treated": size_treated, "support": support,
            "ni_score1": None, "ni_score2": None, "ni_score3": None, "p_v1": None, "p_v2": None, "utility": None, "std": std, "kl": kl, "emd": emd, "diff_means": diff_means}


def ni_score(x, lamda):
    return 1 - (1 / (np.exp(lamda * x)))


def differ_metrics(a, b):
    if len(a) > len(b):
        a = np.random.choice(a, len(b))
    elif len(b) > len(a):
        b = np.random.choice(b,len(a))
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    kl = np.ma.masked_invalid(np.where(a != 0, a * np.log(a / b), 0)).sum()
    emd = wasserstein_distance(a, b)
    diff_means = abs(np.mean(a) - np.mean(b))
    return kl, emd, diff_means


def calc_facts_metrics(data, func):
    relations = all_possible_relations(list(data.columns), data, OUTCOME_COLUMN, PROTECTED_COLUMN)
    n = data.shape[0]
    results = []
    small_treatments = []
    for subpopulation, treatment, outcome, values1, values2 in relations:
        for val_popu in values1:
            population = data.loc[data[subpopulation] == val_popu]
            size = population.shape[0]
            support = size / n
            std = np.std(population[OUTCOME_COLUMN])
            if size == 0:
                continue
            df_protected = func(population)
            size_protected = df_protected.shape[0]
            kl, emd, diff_means = differ_metrics(population[OUTCOME_COLUMN], df_protected[OUTCOME_COLUMN])
            for val_treat in values2:
                size_treated = population.loc[population[treatment] == val_treat].shape[0]
                results.append(analyze_relation(population, subpopulation, val_popu, treatment, outcome, func, val_treat, size, size_protected, size_treated, support, std, kl, emd, diff_means))
    pd.DataFrame(results).to_csv(f"../outputs/{PROJECT_DIRECTORY}/all_facts.csv", index=False)
    pd.DataFrame(small_treatments).to_csv(f"../outputs/{PROJECT_DIRECTORY}/small_treatments.csv", index=False)


def get_intersection(df_facts, att1, att2):
    if att1['subpopulation'] == att2['subpopulation'] and att1['value_population'] == att2['value_population']:
        row = df_facts.loc[(df_facts['subpopulation'] == att1['subpopulation'])
                           & (df_facts['value_population'] == att1['value_population'])]
        return row['size'].values[0]
    elif att1['subpopulation'] == att2['subpopulation'] and att1['value_population'] != att2['value_population']:
        return 0
    else:
        row = df_facts.loc[(df_facts['subpopulation'] == att1['subpopulation'])
                           & (df_facts['value_population'] == att1['value_population'])
                           & (df_facts['treatment'] == att2['subpopulation'])
                           & (df_facts['value'] == att2['value_population'])]
        if row.shape[0] == 0:
            row = df_facts.loc[(df_facts['subpopulation'] == att2['subpopulation'])
                               & (df_facts['value_population'] == att2['value_population'])
                               & (df_facts['treatment'] == att1['subpopulation'])
                               & (df_facts['value'] == att1['value_population'])]
        if row.shape[0] == 0:
            if "_".join([att1['subpopulation'], att1['value_population'], att2['subpopulation'], att2['value_population']]) in CALCED_INTERSECTIONS:
                return CALCED_INTERSECTIONS["_".join([att1['subpopulation'], att1['value_population'], att2['subpopulation'], att2['value_population']])]
            if "_".join([att2['subpopulation'], att2['value_population'], att1['subpopulation'], att1['value_population']]) in CALCED_INTERSECTIONS:
                return CALCED_INTERSECTIONS["_".join([att2['subpopulation'], att2['value_population'], att1['subpopulation'], att1['value_population']])]
            r = CLEAN_DF.loc[(CLEAN_DF[att1['subpopulation']] == att1['value_population'])
                             & (CLEAN_DF[att2['subpopulation']] == att2['value_population'])].shape[0]
            CALCED_INTERSECTIONS["_".join([att1['subpopulation'], att1['value_population'], att2['subpopulation'], att2['value_population']])] = r
            return r
    res = row['size_treated'].values[0]
    return res


def get_score(max_subpopulation, df_facts, group, attribute, alpha, K, lamda):
    intersection = 0
    checked_group = group.copy()
    checked_group.append(attribute)
    utility_sum = sum([i['utility'] for i in checked_group])
    utility = utility_sum / K
    if group:
        for pair in itertools.combinations(checked_group, 2):
            intersection += get_intersection(df_facts, pair[0], pair[1])
    f_intersection = ((max_subpopulation*len(checked_group)*len(checked_group)) - intersection) / (max_subpopulation*len(checked_group)*len(checked_group))
    intersecction_list.append(f_intersection)
    score = (alpha * utility) + ((1 - alpha) * f_intersection)
    return pd.Series({"utility_sum": utility_sum, "utility": utility, "intersection_sum": intersection,
                      "final_intersection": f_intersection, "score": score})


def greedy(df_facts, max_subpopulation, alpha, lamda):
    print(f"LOOKING FOR GROUP alpha={alpha} and lamda={lamda}")
    i = 0
    group = []
    scores = []
    while i < K:
        max_score = 0
        curr_row = None
        score_dictionary = None
        for index, row in df_facts.iterrows():
            if not any(row.equals(s) for s in group):
                score_dict = get_score(max_subpopulation, df_facts, group, row, alpha, K, lamda)
                if score_dict["score"] > max_score:
                    max_score = score_dict["score"]
                    curr_row = row
                    score_dictionary = score_dict
        group.append(curr_row)
        scores.append(score_dictionary)
        print(f"Add row subpopulation={curr_row['subpopulation']} valu_pol={curr_row['value_population']} treatment={curr_row['treatment']} value_treat={curr_row['value']}")
        print(f" Now score is: {max_score}")
        i += 1
    return group, scores

def find_group(clean_df):
    df_facts = pd.read_csv(f"../outputs/{PROJECT_DIRECTORY}/all_facts.csv")
    df_facts = df_facts.dropna()
    max_subpopulation = max(df_facts['size'])
    for lamda in [0.0001]:
        for alpha in [0.5]:
            group, scores = greedy(df_facts, max_subpopulation, alpha, lamda)
            df_calc = pd.concat(group, axis=1)
            transposed_df1 = df_calc.T
            transposed_df1.to_csv(f"../outputs/{PROJECT_DIRECTORY}/find_k/{K}_{alpha}_{lamda}.csv", index=False)
            df_scores = pd.concat(scores, axis=1)
            transposed_df2 = df_scores.T
            transposed_df2.to_csv(f"../outputs/{PROJECT_DIRECTORY}/scores/{K}_{alpha}_{lamda}.csv", index=False)


if __name__ == '__main__':
    start = time.time()
    df = pd.read_csv(CLEAN_DF_PATH)
    calc_facts_metrics(df, get_secured_population)
    end1 = time.time()
    find_group(df)
    end2 = time.time()
    print(f"calc facts took {end1-start}")
    print(f"find group took {end2 - end1}")
