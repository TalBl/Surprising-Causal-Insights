import pandas as pd
from cdt.causality.graph import PC
import networkx as nx


def getTreatmentCATE(df_g, DAG, treatment, target):
    # df_g['TempTreatment'] = df_g.apply(lambda row: addTempTreatment(row, treatment, ordinal_atts), axis=1)
    df_g['TempTreatment'] = df_g.apply(lambda row: addTempTreatment(row, treatment), axis=1)
    DAG_ = changeDAG(DAG, treatment)
    causal_graph = """
                        digraph {
                        """
    for line in DAG_:
        causal_graph = causal_graph + line + "\n"
    causal_graph = causal_graph + "}"
    try:
        ATE, p_value = estimateATE(causal_graph, df_g, 'TempTreatment', target)
        if p_value > P_VAL:
            print("p-val ", p_value)
    except:
        ATE = 0
    return ATE



def estimateATE(causal_graph, df, T, O):
    model = CausalModel(
        data=df,
        graph=causal_graph.replace("\n", " "),
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


