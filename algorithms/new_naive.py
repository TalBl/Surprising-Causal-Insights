import pandas as pd
import numpy as np
from dowhy import CausalModel
import networkx as nx
import itertools
import ast
from tqdm import tqdm
from time import time
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIRECTORY = "so"
K = 5
LAMDA = 0.00009
CLEAN_DF_PATH = f"../outputs/{PROJECT_DIRECTORY}/clean_data.csv"
OUTCOME_COLUMN = "ConvertedCompYearly"
CLEAN_DF = pd.read_csv(CLEAN_DF_PATH)
CALCED_INTERSECTIONS = {}
CALCED_UTILITY = {}

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


def get_intersection(att1, att2):
    key = str((tuple(att1["subpopulation"]), tuple(att2["subpopulation"])))
    key2 = str((tuple(att2["subpopulation"]), tuple(att1["subpopulation"])))
    if key in CALCED_INTERSECTIONS:
        return CALCED_INTERSECTIONS[key]
    if key2 in CALCED_INTERSECTIONS:
        return CALCED_INTERSECTIONS[key2]
    population = CLEAN_DF.copy()
    for d in att1['subpopulation']:
        population = population[population[d["att"]] == d["value"]]
        if population.shape[0] == 0:
            CALCED_INTERSECTIONS[key] = 0
            return 0
    for d in att2['subpopulation']:
        population = population[population[d["att"]] == d["value"]]
    if population.shape[0] == 0:
        CALCED_INTERSECTIONS[key] = 0
        return 0
    r = population.shape[0]
    CALCED_INTERSECTIONS[key] = r
    return r


def calc_cate(group: list, df: pd.DataFrame, outcome_column, treatment: list):
    filtered_df = df.copy()
    for d in group:
        filtered_df = filtered_df[filtered_df[d["att"]] == d['value']]
    if filtered_df.shape[0] == 0:
        return None, None
    # Initialize the new_treatment column to 1
    filtered_df["new_treatment"] = 1
    for d in treatment:
        filtered_df["new_treatment"] = filtered_df["new_treatment"] & filtered_df[d["att"]].apply(lambda x: 1 if x == d["value"] else 0).astype(int)

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


def get_utility(data, group1, group2, subpopulation, treatment):
    key = str((tuple(subpopulation), tuple(treatment)))
    if key in CALCED_UTILITY:
        return CALCED_UTILITY[key]
    for d in subpopulation:
        data = data[data[d["att"]] == d["value"]]
    if data.shape[0] == 0:
        return None
    ate1, p_v1 = calc_cate(group1, data, OUTCOME_COLUMN, treatment)
    ate2, p_v2 = calc_cate(group2, data, OUTCOME_COLUMN, treatment)
    if ate1 and ate2:
        iscore = abs(ate1 - ate2)
        utility = 1 - (1 / (np.exp(LAMDA * iscore)))
        CALCED_UTILITY[key] = utility
        return utility
    return None


def get_score(data, group1, group2, max_subpopulation, checked_group, alpha=0.5):
    intersection = 0
    utilities = [get_utility(data, group1, group2, d["subpopulation"], d["treatment"]) for d in checked_group]
    if None in utilities:
        return {"score": float("-inf")}
    utility = sum(utilities) / len(checked_group)
    for pair in itertools.combinations(checked_group, 2):
        intersection += get_intersection(pair[0], pair[1])
    f_intersection = ((max_subpopulation*len(checked_group)*len(checked_group)) - intersection) / (max_subpopulation*len(checked_group)*len(checked_group))
    score = (alpha * utility) + ((1 - alpha) * f_intersection)
    return {"utility_sum": sum(utilities), "utility": utility, "intersection_sum": intersection,
            "final_intersection": f_intersection, "score": score}


def naive(df, group1, group2, subpopulations, treatments):
    r = []
    for population in subpopulations:
        df_c = df.copy()
        for d in population:
            df_c = df_c[df_c[d["att"]] == d["value"]]
        r.append(df_c.shape[0])
    max_subpopulation = max(r)
    max_score = 0
    res_group = pd.DataFrame()
    groups_and_treatments = []
    for s in subpopulations:
        for t in treatments:
            groups_and_treatments.append({"subpopulation": s, "treatment": t})
    for group_and_treat in tqdm(itertools.combinations(groups_and_treatments, K), total=98280):
        score_dict = get_score(df, group1, group2, max_subpopulation, group_and_treat)
        if score_dict["score"] > max_score:
            max_score = score_dict["score"]
            res_group = group_and_treat
            d = score_dict
    res_group.to_csv(f"outputs/{PROJECT_DIRECTORY}/find_k/naive.csv", index=False)
    print(d)


def create_combinations(lst):
    single_elements = [[x] for x in lst]
    #two_different_elements = [list(comb) for comb in itertools.combinations(lst, 2)]
    return single_elements #+ two_different_elements


if __name__ == '__main__':
    df = pd.read_csv(CLEAN_DF_PATH)
    GROUP1 = [{"att": "United States of America", "value": 1}]
    GROUP2 = [{"att": "United Kingdom of Great Britain and Northern Ireland", "value": 1}]
    SUBPOPULATIONS = [{"att": "Gender", "value": "Woman"}, {"att": "Gender", "value": "Man"}, {"att": "Age", "value": "35-44 years old"}, {"att": "Age", "value": "25-34 years old"}
                      ,{"att": "White", "value": 1}, {"att": "Asian", "value": 1}, {"att": "Black", "value": 1}]
    TREATMENTS = [{"att": "Developer, back-end", "value": 1}, {"att": "Engineer, data", "value": 1}
                  ,{"att": "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)", "value": 1},
                  {"att": "Bachelor’s degree (B.A., B.S., B.Eng., etc.)", "value": 1}]
    naive(df, GROUP1, GROUP2, create_combinations(SUBPOPULATIONS), create_combinations(TREATMENTS))
