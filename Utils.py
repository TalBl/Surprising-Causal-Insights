import numpy as np
from dowhy import CausalModel
from collections import defaultdict, deque
import copy
import networkx as nx


P_VAL = 0.1


class Dataset:
    def __init__(self, name, clean_df, outcome_col, group1, group2, treatments, subpopulations):
        self.name = name
        self.clean_df = clean_df
        self.outcome_col = outcome_col
        self.group1 = group1
        self.group2 = group2
        self.treatments = treatments
        self.subpopulations = subpopulations

    def get_population(self, subpopulation):
        df_copy = self.clean_df.copy()
        for d in subpopulation:
            df_copy = df_copy[df_copy[d["att"]]==d["value"]]
        return df_copy

    def get_treatment(self, population, treatment):
        df_copy = population.copy()
        for d in treatment:
            df_copy = df_copy[df_copy[d["att"]]==d["value"]]
        return df_copy


def addTempTreatment(row, t):
    res = 1
    if type(t) == dict:
        t = [t]
    for d in t:
        if d['value'](row[d['att']]) == 0:
            res = 0
    return res


def getTreatmentCATE(df_g, DAG, treatment, target, cols_dict, graph_dict=None):
    # df_g['TempTreatment'] = df_g.apply(lambda row: addTempTreatment(row, treatment, ordinal_atts), axis=1)
    df_g['TempTreatment'] = df_g.apply(lambda row: addTempTreatment(row, treatment), axis=1)
    if type(treatment) == dict and cols_dict[treatment["att"]] in graph_dict:
        causal_graph = graph_dict[cols_dict[treatment["att"]]]
    else:
        DAG_ = changeDAG(DAG, treatment, cols_dict)
        edges = []
        for line in DAG_:
            if '->' in line:
                edges.append([line.split(" ->")[0].split("'")[1], line.split("-> ")[1].split(";'")[0]])
        causal_graph = nx.DiGraph()
        causal_graph.add_edges_from(edges)
        if type(treatment) == dict:
            graph_dict[cols_dict[treatment["att"]]] = causal_graph
    try:
        ATE, p_value = estimateATE(causal_graph, df_g, 'TempTreatment', target)
        if p_value > P_VAL:
            return 0, graph_dict
    except Exception as e:
        return 0, graph_dict
    return ATE, graph_dict


def changeDAG(dag, randomTreatment, cols_dict):
    DAG = copy.deepcopy(dag)
    toRomove = []
    toAdd = ['TempTreatment;']
    randomTreatment = [randomTreatment] if type(randomTreatment) == dict else randomTreatment
    atts_treatments = [cols_dict[x['att']] for x in randomTreatment]
    for a in atts_treatments:
        for c in DAG:
            if '->' in c:
                if a in c:
                    toRomove.append(c)
                    # left hand side
                    if a in c.split(" ->")[0]:
                        string = c.replace(a, "TempTreatment")
                        if not string in toAdd:
                            toAdd.append(string)
    for r in toRomove:
        if r in DAG:
            DAG.remove(r)
    for a in toAdd:
        if not a in DAG:
            DAG.append(a)
    return list(set(DAG))


def estimateATE(causal_graph, df, T, O):
    model = CausalModel(
        data=df,
        graph=causal_graph,
        treatment=T,
        outcome=O)
    estimands = model.identify_effect()
    causal_estimate_reg = model.estimate_effect(estimands,
                                                method_name="backdoor.linear_regression",
                                                target_units="ate",
                                                #evaluate_effect_strength=True,
                                                effect_modifiers = [],
                                                test_significance=True)
    return causal_estimate_reg.value, causal_estimate_reg.test_stat_significance()['p_value']

