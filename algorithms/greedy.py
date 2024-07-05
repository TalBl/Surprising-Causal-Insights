def algorithm(df_facts, n, df_intersection, mini_intersection, alpha, lamda, K, score_func):
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
                score_dict = score_func(n, df_intersection, mini_intersection, group, row, alpha, K, lamda)
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
