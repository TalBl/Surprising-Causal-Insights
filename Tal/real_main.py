from Tal.clean_acs import *
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import math

ATTS = ["Age", "Citizenship status", "Marital status", "Educational attainment", "Total person's income", "With a disability", "Race/Ethnicity", "Sex"]
PROJECT_DIRECTORY = "acs"
K = 10
LAMDA_VALUES = [0.0005, 0.0003, 0.0001]
CLEAN_DF_PATH = "../data/acs/2018_all_data_clean_final.csv"
PROTECTED_COLUMN = "Sex"
OUTCOME_COLUMN = "Total person's income"


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


def calc_cate(data, treatment, value, outcome):
    popu_1 = np.mean(data.loc[data[treatment] == value][outcome])
    popu_0 = np.mean(data.loc[data[treatment] != value][outcome])
    return popu_1 - popu_0


def analyze_relation(data, subpopulation, value_popu, treatment, outcome, func_for_protected, value, size, size_protected, size_treated, support):
    protected_population = func_for_protected(data)
    ate = calc_cate(data, treatment, value, outcome)
    ate_p = calc_cate(protected_population, treatment, value, outcome)
    css = (max(ate, ate_p) - min(ate, ate_p))
    return {'subpopulation': subpopulation, 'value_population': value_popu, 'treatment': treatment, 'value': value, 'outcome': outcome, 'ate': ate, 'ate_p':ate_p,
            'css': css, 'size': size, 'size_protected': size_protected, "size_treated": size_treated, "support": support,
            "ni_score1": ni_score(css, LAMDA_VALUES[0]), "ni_score2": ni_score(css, LAMDA_VALUES[1]), "ni_score3": ni_score(css, LAMDA_VALUES[2])}


def ni_score(x, lamda):
    return 1 - (1 / (np.exp(lamda * x)))


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
            if size == 0:
                continue
            df_protected = func(population)
            size_protected = df_protected.shape[0]
            for val_treat in values2:
                size_treated = population.loc[population[treatment] == val_treat].shape[0]
                if size_treated < 100:
                    small_treatments.append({"subpopulation": subpopulation, "val_population": val_popu,
                                             "treatment": treatment, "val_treatment": val_treat, "size": size_treated})
                else:
                    results.append(analyze_relation(population, subpopulation, val_popu, treatment, outcome, func, val_treat, size, size_protected, size_treated, support))
    pd.DataFrame(results).to_csv(f"../outputs/{PROJECT_DIRECTORY}/all_facts.csv", index=False)
    pd.DataFrame(small_treatments).to_csv(f"../outputs/{PROJECT_DIRECTORY}/small_treatments.csv", index=False)


def calc_intersection(clean_df, att1, att2):
    size = clean_df.loc[(clean_df[att1['subpopulation']] == att1['value_population'])
                        & (clean_df[att2['subpopulation']] == att2['value_population'])]
    return size.shape[0]


def mini_calc_intersection(clean_df, att1):
    size = clean_df.loc[clean_df[att1['subpopulation']] == att1['value_population']]
    return size.shape[0]


def intersection(clean_df, protected_column):
    df_facts = pd.read_csv(f"../outputs/{PROJECT_DIRECTORY}/all_facts.csv")[['subpopulation', 'value_population']]
    df_n = df_facts.drop_duplicates().reset_index(drop=True)
    f_df = pd.DataFrame(columns=df_facts.columns)
    for pair in tqdm(itertools.combinations(range(df_n.shape[0]), 2), total=math.comb(df_n.shape[0], 2)):
        combination = [df_n.loc[pair[0]]['subpopulation'], df_n.loc[pair[1]]['subpopulation']]
        if protected_column in combination or df_n.loc[pair[0]]['subpopulation'] == df_n.loc[pair[1]]['subpopulation']:
            continue
        intersection_res = calc_intersection(clean_df, df_n.loc[pair[0]], df_n.loc[pair[1]])
        new_row_data = pd.Series({'subpopulation1': df_n.loc[pair[0]]['subpopulation'], 'popu_val1': df_n.loc[pair[0]]['value_population'],
                                  'subpopulation2': df_n.loc[pair[1]]['subpopulation'], 'popu_val2': df_n.loc[pair[1]]['value_population'],
                                  'intersection': intersection_res})
        f_df = pd.concat([f_df, pd.DataFrame([new_row_data], columns=new_row_data.index)]).reset_index(drop=True)
    f_df.to_csv(f"../outputs/{PROJECT_DIRECTORY}/intersection.csv", index=False)
    for index, row in tqdm(df_n.iterrows(), total=32):
        if protected_column in row['subpopulation']:
            continue
        intersection_res = mini_calc_intersection(clean_df, row)
        new_row_data = pd.Series({'subpopulation': row['subpopulation'], 'value_population': row['value_population'],
                                  'intersection': intersection_res})
        f_df = pd.concat([f_df, pd.DataFrame([new_row_data], columns=new_row_data.index)]).reset_index(drop=True)
    f_df.to_csv(f"../outputs/{PROJECT_DIRECTORY}/mini_intersection.csv", index=False)


def get_intersection(df_intersection, mini_intersection, att1, att2):
    try:
        if att1['subpopulation'] == att2['subpopulation'] and att1['value_population'] == att2['value_population']:
            row = mini_intersection.loc[(mini_intersection['subpopulation'] == att1['subpopulation'])
                                        & (mini_intersection['value_population'] == att1['value_population'])]
            return row['intersection'].values[0]
        elif att1['subpopulation'] == att2['subpopulation'] and att1['value_population'] != att2['value_population']:
            return 0
        else:
            row = df_intersection.loc[(df_intersection['subpopulation1'] == att1['subpopulation'])
                                      & (df_intersection['popu_val1'] == att1['value_population'])
                                      & (df_intersection['subpopulation2'] == att2['subpopulation'])
                                      & (df_intersection['popu_val2'] == att2['value_population'])]
            if row.shape[0] == 0:
                row = df_intersection.loc[(df_intersection['subpopulation1'] == att2['subpopulation'])
                                          & (df_intersection['popu_val1'] == att2['value_population'])
                                          & (df_intersection['subpopulation2'] == att1['subpopulation'])
                                          & (df_intersection['popu_val2'] == att1['value_population'])]
    except:
        return 0
    return row['intersection'].values[0]


def get_score(n, df_intersection, mini_intersection, group, attribute, alpha, K, lamda):
    intersection = 0
    checked_group = group.copy()
    checked_group.append(attribute)
    utility_sum = sum([ni_score(x['css'], lamda) for x in checked_group])
    utility = utility_sum / K
    if group:
        for pair in itertools.combinations(checked_group, 2):
            intersection += get_intersection(df_intersection, mini_intersection, pair[0], pair[1])
    f_intersection = ((n*K*K) - intersection) / (n*K*K)
    score = (alpha * utility) + ((1 - alpha) * f_intersection)
    return pd.Series({"utility_sum": utility_sum, "utility": utility, "intersection_sum": intersection,
                      "final_intersection": f_intersection, "score": score})


def greedy(df_facts, n, df_intersection, mini_intersection, alpha, lamda):
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
                score_dict = get_score(n, df_intersection, mini_intersection, group, row, alpha, K, lamda)
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
    N = clean_df.shape[0]
    df_intersection = pd.read_csv(f"../outputs/{PROJECT_DIRECTORY}/intersection.csv")
    mini_intersection = pd.read_csv(f"../outputs/{PROJECT_DIRECTORY}/mini_intersection.csv")
    for lamda in [0.0001, 0.0003]:
        for alpha in [0, 0.25, 0.5, 0.75, 1]:
            group, scores = greedy(df_facts, N, df_intersection, mini_intersection, alpha, lamda)
            df_calc = pd.concat(group, axis=1)
            transposed_df1 = df_calc.T
            transposed_df1.to_csv(f"../outputs/{PROJECT_DIRECTORY}/find_k/{K}_{alpha}_{lamda}.csv", index=False)
            df_scores = pd.concat(scores, axis=1)
            transposed_df2 = df_scores.T
            transposed_df2.to_csv(f"../outputs/{PROJECT_DIRECTORY}/scores/{K}_{alpha}_{lamda}.csv", index=False)


if __name__ == '__main__':
    df = pd.read_csv(CLEAN_DF_PATH)
    df = df[ATTS]
    calc_facts_metrics(df, get_secured_population)
    intersection(df, PROTECTED_COLUMN)
    find_group(df)
