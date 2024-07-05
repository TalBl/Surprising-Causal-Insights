import time
from typing import Dict
import numpy as np
import pandas as pd
import copy
import math
from dowhy import CausalModel
import networkx as nx
import itertools

DICT_RULES_BY_SIZE = {}

def so_syn_dag():
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


DAG_d = so_syn_dag()
data = pd.read_csv("data/acs/acstest_mini.csv")
data["new_treatment"] = data["Educational attainment"].apply(lambda x: 1 if x == "Professional Degree" else 0).astype(int)
PROTECTED_COLUMN = "Sex"
PROTECTED_VALUE = "Woman"


class Rule:
    def __init__(self, subpopulation_list, values_list):
        self.count = 0
        self.marginal_value = 0
        self.list_populations = []
        for index, subpopulation in enumerate(subpopulation_list):
            self.list_populations.append({"subpopulation": subpopulation, "value": values_list[index]})
        self.list_populations = list(sorted(self.list_populations, key=lambda d: d['subpopulation']))
        self.weight = 0
        self.sub_rules = []


def BRS(k: int, t: pd.DataFrame, w, max_weight=10, combinations=None):
    s = []
    for i in range(k):
        r_m = best_marginal_rule(s, t, w, max_weight, combinations)
        s.append(r_m)
    return s


def is_rule_covered(row, rule):
    for index, item in enumerate(rule.list_populations):
        if row[1][item['subpopulation']] != item["value"]:
            return False
    return True


def highest_covered_rule(s, r):
    sorted_rules = sorted(s, key=lambda rule: rule.weight, reverse=True)
    for rule in sorted_rules:
        if is_rule_covered(r, rule):
            return rule


def get_super_rules_from_specific_rule(r: Rule, combinations):
    res_list = []
    pop_list = [x["subpopulation"] for x in r.list_populations]
    for prime_rule in combinations:
        if prime_rule.list_populations[0]['subpopulation'] not in pop_list:
            new_subpopulation = pop_list.copy()
            new_subpopulation.append(prime_rule.list_populations[0]['subpopulation'])
            new_values = [x["value"] for x in r.list_populations].copy()
            new_values.append(prime_rule.list_populations[0]['value'])
            rule_new = Rule(subpopulation_list=new_subpopulation, values_list=new_values)
            r_weight = weight(rule_new)
            if not r_weight:
                continue
            else:
                r.weight = r_weight
            res_list.append(rule_new)
    return res_list

# Function to convert a dictionary to a sorted tuple of items
def dict_to_tuple(d):
    return tuple(sorted(d.items()))

# Function to convert a list of dictionaries to a tuple of tuples
def list_of_dicts_to_tuple(lst):
    return tuple(dict_to_tuple(d) for d in lst)

def tuple_to_dict(t):
    return dict(t)

def tuple_of_tuples_to_list_of_dicts(t):
    return [tuple_to_dict(d) for d in t]


def get_unique_objects(rules):
    unique_rules = []
    seen = set()
    for rule in rules:
        if not unique_rules:
            seen.add(list_of_dicts_to_tuple(rule.list_populations))
            unique_rules.append(rule)
        else:
            if list_of_dicts_to_tuple(rule.list_populations) not in seen:
                seen.add(list_of_dicts_to_tuple(rule.list_populations))
                unique_rules.append(rule)
    return unique_rules


def get_all_super_rules(c, combinations):
    r_list = []
    for r in c:
        r_list.extend(get_super_rules_from_specific_rule(r, combinations))
    return get_unique_objects(r_list)


def weight(rule):
    selected_data = data.copy(deep=True)
    for item in rule.list_populations:
        selected_data = selected_data.loc[selected_data[item["subpopulation"]] == item["value"]]
    protected_population = selected_data.loc[selected_data[PROTECTED_COLUMN] == PROTECTED_VALUE]
    if selected_data.shape[0] == 0 or protected_population.shape[0] == 0:
        return
    ate = calc_cate(data)
    ate_p = calc_cate(protected_population)
    if ate and ate_p:
        iscore = abs(ate - ate_p)
        return 1 - (1 / (np.exp(0.00009 * iscore)))


def calc_cate(data):
    model = CausalModel(
        data=data,
        graph=DAG_d,
        treatment="new_treatment",
        outcome="Total person's income")
    estimands = model.identify_effect()
    try:
        causal_estimate_reg = model.estimate_effect(estimands,
                                                    method_name="backdoor.linear_regression",
                                                    target_units="ate",
                                                    effect_modifiers=[],
                                                    test_significance=True)
        p_val = causal_estimate_reg.test_stat_significance()['p_value']
        if p_val < 0.05:
            return causal_estimate_reg.value
        else:
            return
    except:
        return


def update_dict(size, rules_list):
    DICT_RULES_BY_SIZE[size] = {}
    for rule in rules_list:
        DICT_RULES_BY_SIZE[size][list_of_dicts_to_tuple(rule.list_populations)] = rule


def find_sub_rules(curr_size, rule):
    if curr_size == 1:
        return []
    sublists = []
    for length in range(1, len(rule.list_populations)):
        for combination in itertools.combinations(rule.list_populations, length):
            sublists.append(list(combination))
    sub_rules = []
    for population in sublists:
        r = DICT_RULES_BY_SIZE[curr_size-1][list_of_dicts_to_tuple(population)]
        if r.weight != float("inf"):
            sub_rules.append(r)
    return sub_rules


def best_marginal_rule(s: list, t: pd.DataFrame, w, max_weight=10000000, list_combinations=None):
    if list_combinations is None:
        list_combinations = []
    h = 0
    c = []
    c_0, c_n = [], []
    for j in range(1, 2):
        if j == 1:
            c_n = list_combinations
        else:
            c_n = get_all_super_rules(c_0, list_combinations)
        update_dict(j, c_n)
        if c_0:
            max_weight = max([x.weight for x in c_0 if x.weight != float("inf")])
        else:
            max_weight = 100000000
        copy_c_n = copy.deepcopy(c_n)
        for rule_x in copy_c_n:
            m = float('inf')
            for sub_rule in find_sub_rules(j, rule_x):
                m = min(m, sub_rule.marginal_value + sub_rule.count)*(max_weight-sub_rule.weight)
            if m < h:
                if rule_x in c_n:
                    c_n.remove(rule_x)
        if not c_n:
            break
        for rule in c_n:
            rule.count = 0
            rule.marginal_value = 0
        for row in t.iterrows():
            r_s = highest_covered_rule(s, row)
            covered_rules = [rule_2 for rule_2 in c_n if is_rule_covered(row, rule_2)]
            for rule in covered_rules:
                rule.count += 1
                if r_s:
                    rule.marginal_value += rule.weight - min(rule.weight, r_s.weight)
                else:
                    rule.marginal_value += rule.weight
        c.extend(c_n)
        c_0 = c_n
        h = max([r.marginal_value for r in c])
    sorted_rules = list(sorted(c, key=lambda rule: rule.marginal_value, reverse=True))
    for r in s:
        if r in sorted_rules:
            sorted_rules.remove(r)
    return sorted_rules[0]


def calc_combinations():
    list_combinations = []
    for column in data.columns:
        if column == "Total person's income" or column=="Educational attainment" or column == "new_treatment" or column=="Sex":
            continue
        values = list(set(data[column]))
        for value in values:
            if pd.isna(value):
                continue
            r = Rule(subpopulation_list=[column], values_list=[value])
            r_weight = weight(r)
            if not r_weight:
                continue
            r.weight = r_weight
            list_combinations.append(r)
    return list_combinations


start = time.time()
prior_combinations = calc_combinations()
res = BRS(k=5, t=data, w=weight, max_weight=10, combinations=prior_combinations)
for rule in res:
    print(f"{rule.list_populations}")
end = time.time()
print(f"took {end-start}")
