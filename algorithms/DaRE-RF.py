from Utils import *

def expand_subsets(df, N, level, support_range):
    E = []
    if not N:
        for index, row in df.iterrows():
            if isValidSupport(df, [row], support_range):
                E.append([row])
    for index1, parent1 in enumerate(N):
        for index2, parent2 in enumerate(N):
            if index2 <= index1:
                continue
            set_intersection = parent1.intersection(parent2)
            if len(set_intersection) != (level - 1) :
                continue
            list_union = parent1.union(parent2)
            if not doesSubsetAlreadyExist(list_union, E) and isSubsetRealistic(list_union):
                E.append({"group": list_union, "cate": calc_cate(list_union), "size": int(calc_support(list_union, df)*N),
                          "parent1": {"cate": parent1["cate"], "size": parent1["size"]},
                          "parent2": {"cate": parent2["cate"], "size": parent2["size"]}
                          })
    return E


def doesSubsetAlreadyExist(group, all_combinations):
    for item in all_combinations:
        if item.difference(group) == set():
            return True
    return False


def isSubsetRealistic(group):
    subgroups = list(set([','.join([x['subpopulation'], x['value_population']]) for x in group]))
    return len(subgroups) == len(group)


def isValidSupport(df, group, support_range):
    support = calc_support(group, df)
    return support_range[0] <= support <= support_range[1]


def evaluateSubset(df, group):
    cate_child_value = calc_cate(group)
    avg_child_value = cate_child_value / len(group)
    avgP1Value = group['parent1']['cate'] / group['parent1']['size']
    avgP2Value = group['parent2']['cate'] / group['parent2']['size']
    return abs(avg_child_value) < abs(avgP1Value) and abs(avg_child_value) < abs(avgP2Value)


def algorithm(df, max_literals, support_range):
    N = []
    res = []
    level = 1
    E = expand_subsets(df, N, level, support_range)
    while level <= max_literals:
        for group in E:
            if not isValidSupport(df, group, support_range):
                if calc_support(group, df) > support_range[1]:
                    N.append(group)
                else:
                    continue
            if evaluateSubset(df, group):
                N.append(group)
                res.append(group)
        level += 1
        E = expand_subsets(df, N, level, support_range)
        if not E:
            break
    return res
