import pandas as pd
import numpy as np
import Utils
import ast
from dowhy import CausalModel
PATH = 'C:/Users/brity/OneDrive - Technion/Desktop/Anna-Yuval-Brit/'
from itertools import combinations
LAMBDA = 0.0001


def main():
    df = pd.read_csv(PATH + 'so_countries_col_new.csv')

    # # column_name = 'Gender'  # Replace with the actual name of your column
    # # valid_values = ['Male', 'Female']
    # #
    # # # Replace values not in the valid_values list with None
    # # df[column_name] = df[column_name].apply(lambda x: x if x in valid_values else None)
    #
    # df['DevType'] = df['DevType'].str.contains('Data scientist').astype(int)

    # df = Utils.get_syn_df()
    atts = ['Age','Gender','DevType','FormalEducation','ConvertedSalary']
    df = df[atts]
    df.dropna()
    print(len(df))
    n = len(df)

    atts_groups = ['Age','DevType','FormalEducation']
    dag = Utils.so_dag()

    #
    BF(atts_groups, df,dag,n)
    greedy(atts_groups, df,dag)

def greedy(atts_groups, df,dag):
    df_rules = get_all_rules(atts_groups, dag, df)

    n = len(df_rules)
    ans = df_rules.copy()
    ans = ans.iloc[0:0]
    ans_i = []
    k = 3

    s = 0
    for i in range(0,k):
        r = getBestRule(df_rules, ans, n,df, s,ans_i)
        if r >= 0:
            row_to_add = df_rules.loc[r]
            ans = pd.concat([ans, row_to_add.to_frame().T], ignore_index=True)

            utility = getUtility(ans)
            overlap = getOverlap(df, n, ans)
            s = 0.5 * overlap + 0.5 * utility
            ans_i.append(r)


    utility = getUtility(ans)
    overlap = getOverlap(df, n, ans)
    score = 0.5 * overlap + 0.5 * utility
    print("Greedy solution: ", ans_i, overlap, utility, score)

def getBestRule(df_rules, ans, n,df,s,ans_i):

    r = None
    score = 0
    for index, row in df_rules.iterrows():
        if index in ans.index.tolist():
            continue
        temp = ans.copy()
        row_to_add = df_rules.loc[index]
        temp = pd.concat([temp, row_to_add.to_frame().T], ignore_index=True)

        utility = getUtility(temp)
        overlap = getOverlap(df, n, temp)
        score_temp =  0.5 * overlap + 0.5 * utility
        if score_temp > score:
            if not index in ans_i:
                r = index
                score = score_temp
    if score > s:
        print("picked rule: ", r, "score: ", score)
        return r
    else:
        print('could not find a new rule to add')
        return -1



def BF(atts, df,dag,m):
    df_rules = get_all_rules(atts, dag, df,m)
    k = 4
    n = len(df_rules)
    print(n)
    indices = list(range(n))
    subsets = []
    for i in range(1, k):  # n + 1
        subsets.extend(combinations(indices, i))

    print("num of rule sets: ", len(subsets))
    #print(subsets)
    # Iterate over every subset of rows
    f = open("results_size3_so_new.csv", 'w')
    f.write("rules,overlap,utility,score\n")
    for subset in subsets:
        subset_df = df_rules.iloc[list(subset)]
        utility = getUtility(subset_df)
        overlap = getOverlap(df, n, subset_df)
        score = 0.5 * utility+ 0.5 * overlap
        print(subset_df.index.tolist(), overlap, utility, score)
        index_list = subset_df.index.tolist()
        index_list = [str(x) for x in index_list]
        rules_str = ';'.join(index_list)
        # print(index_list,rules_str)
        f.write(rules_str + "," + str(overlap) + "," + str(utility) + "," + str(score) + "\n")
        f.flush()
    f.close()



def getOverlap(df,n, subset_df):
    ans = len(df)*(n**2)
    # Iterate over every pair of rows
    for i in range(len(subset_df)):
        for j in range(i + 1, len(subset_df)):
            pair = (subset_df.iloc[i], subset_df.iloc[j])
            condition1 = pair[0]['Population']
            condition1 = condition1.replace('  ', ',')
            condition1 = ast.literal_eval(condition1)
            condition2 = pair[1]['Population']
            condition2 = condition2.replace('  ', ',')
            condition2 = ast.literal_eval(condition2)
            df1 = df[df[condition1[0]] == condition1[1]]
            df2 = df[df[condition2[0]] == condition2[1]]
            indices_df1 = set(df1.index)
            indices_df2 = set(df2.index)
            overlap_indices = indices_df1.intersection(indices_df2)
            # Get the size of the overlap
            overlap_size = len(overlap_indices)
            ans = ans - overlap_size
    ans = ans/(len(df)*(n**2))
    return ans

def getUtility(subset_df):
    cate_sum = subset_df['Utility'].tolist()
    cate_sum = sum(cate_sum)
    return cate_sum

def get_all_rules(atts, average_salary, dag, df,n):
    df = pd.read_csv('scores_so_new.csv')
    df = df[df['Utility'] > 0.05]
    # df['IScore'] = abs(df['CATE'] - df['CATE protected'])
    # df['NIScore'] =  1 - (1 / np.exp(LAMBDA*df['IScore']))
    # df = df[df['NIScore'] != 0]
    # df['Utility'] = df['NIScore']*(df['size Population']/n)
    # df.to_csv('scores_so_new.csv')
    print("num of rules: ", len(df))
    return df
    # f = open('scores_so.csv', 'w')
    # f.write("Population,Treatment,size Population,size Protected,CATE,CATE protected\n")
    # scores = []
    # attribute_combinations = list(combinations(atts, 2))
    # # Iterate over distinct sets of 2 predicates
    # distinct_predicates = set()
    # for att1, att2 in attribute_combinations:
    #     for _, row in df.iterrows():
    #         pred1 = (att1, row[att1])
    #         pred2 = (att2, row[att2])
    #         distinct_predicates.add((pred1, pred2))
    #         distinct_predicates.add((pred2, pred1))
    # # Print the distinct sets of 2 predicates
    # for pred_set in distinct_predicates:
    #     print(f"Predicates: {pred_set[0]}, {pred_set[1]}")
    #
    #     selected_rows = df[df[pred_set[0][0]] == pred_set[0][1]]
    #
    #     selected_rows[pred_set[1][0]] = selected_rows[pred_set[1][0]].str.contains(pred_set[1][1]).astype(int)
    #     protected_df = selected_rows[selected_rows['Gender'] != 'Male']
    #     print(len(protected_df))
    #     print(len(selected_rows))
    #
    #     if len(protected_df) > 0 and len(selected_rows) > 0:
    #         CATE, CATE_p = CSS(pred_set[1][0], 'ConvertedSalary', selected_rows, protected_df, dag)
    #         print(f"Predicates: {pred_set[0]}, {pred_set[1]}", len(selected_rows), len(protected_df), ans)
    #         p0 = str(pred_set[0]).replace(",", " ")
    #         p1 = str(pred_set[1]).replace(",", " ")
    #         f.write(str(p0) + "," + str(p1) + "," + str(len(selected_rows)) + "," +
    #                 str(len(protected_df)) + "," + str(CATE) + "," + str(CATE_p) +"\n")
    #         f.flush()
    # f.close()


def CSS(T, O, df, df_protected, dag):
    CATE = Utils.compute_ate(dag,df, T, O)
    CATE_p = Utils.compute_ate(dag,df_protected, T, O)
    #print(CATE,CATE_p)

    return CATE, CATE_p


def get_score(alpha, ATE1, ATE2, size2, size1,g):
    ans = alpha*(abs(ATE1 - ATE2)) + (1-alpha)*g*(size1/size2)
    return ans


if __name__ == '__main__':
    main()