import numpy as np
from dowhy import CausalModel


P_VAL = 0.05


def calc_cate(df, treatment, value, outcome, DAG):
    """df['new_treatment'] = df[treatment].apply(lambda x: 1 if x == value else 0)
    model = CausalModel(
        data=df,
        graph=DAG,
        treatment="new_treatment",
        outcome=outcome)
    estimands = model.identify_effect()
    causal_estimate_reg = model.estimate_effect(estimands,
                                                method_name="backdoor.linear_regression",
                                                target_units="ate",
                                                effect_modifiers=[],
                                                test_significance=True)
    p_val = causal_estimate_reg.test_stat_significance()['p_value']
    if p_val < P_VAL:
        return causal_estimate_reg.value
    else:
        return"""
    popu_1 = np.mean(df.loc[df[treatment] == value][outcome])
    popu_0 = np.mean(df.loc[df[treatment] != value][outcome])
    return popu_1 - popu_0


def so_dag(dag_text):
    causal_graph = """
                            digraph {
                            """
    for line in dag_text:
        causal_graph = causal_graph + line + "\n"
    causal_graph = causal_graph + "}"
    causal_dag = causal_graph.replace("\n", " ")
    return causal_dag


def ni_score(x, lamda):
    return 1 - (1 / (np.exp(lamda * x)))


def calc_support(group, data):
    new_df = data.copy()
    for row in group:
        new_df = new_df.loc[new_df[row["subpopulation"]] == row["value_population"]]
        if new_df.shape[0] == 0:
            return 0
    return new_df.shape[0] / data.shape[0]


def calc_intersection(clean_df, att1, att2):
    size = clean_df.loc[(clean_df[att1['subpopulation']] == att1['value_population'])
                        & (clean_df[att2['subpopulation']] == att2['value_population'])]
    return size.shape[0]


def mini_calc_intersection(clean_df, att1):
    size = clean_df.loc[clean_df[att1['subpopulation']] == att1['value_population']]
    return size.shape[0]


