import networkx as nx
import re
import pandas as pd
from statistics import mode
from scipy.stats import zscore
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dowhy import CausalModel
P_VAL = 0.05

def compute_ate(DAG,df, treatment,outcome):
    model = CausalModel(
            data=df,
            graph= DAG,
            treatment=treatment,
            outcome=outcome)


    estimands = model.identify_effect()
    #print(estimands)

    causal_estimate_reg = model.estimate_effect(estimands,
                                                    method_name="backdoor.linear_regression",
                                                    target_units="ate",

                                                    effect_modifiers = [],
                                                    test_significance=True)

    p_val = causal_estimate_reg.test_stat_significance()['p_value']
    if p_val < P_VAL:
        return causal_estimate_reg.value

    else:
        return 0
    #print(causal_estimate_reg)
    return causal_estimate_reg.value



def plot_causal_dag():
    DAG = [
        ('Education', 'Salary'),
        ('Age', 'Education'),
        ('Age', 'Salary'),
        ('Age', 'YearsCoding'),
        ('Education', 'YearsCoding'),
        ('Education', 'DevType'),
        ('YearsCoding', 'DevType'),
        ('DevType', 'Salary'),
        ('YearsCoding', 'Salary')
    ]

    # Create a directed graph
    G = nx.DiGraph(DAG)

    # Plot the graph
    pos = nx.spring_layout(G)  # You can use other layout algorithms
    nx.draw(G, pos, with_labels=True, arrowsize=20, node_size=700, font_size=8, font_color='black', font_weight='bold',
            node_color='skyblue', edge_color='gray')

    # Display the plot
    plt.show()


def string_to_digraph(digraph_string):
    # Extract edges from the string using regular expression
    edges_match = re.findall(r'(\w+)\s*->\s*(\w+)', digraph_string)

    # Create a directed graph
    digraph = nx.DiGraph()

    # Add edges to the graph
    digraph.add_edges_from(edges_match)

    # Extract nodes from the string
    nodes_match = re.findall(r'(\w+);', digraph_string)
    nodes = set(nodes_match)

    # Add isolated nodes to the graph
    digraph.add_nodes_from(nodes)

    return digraph

def digraph_to_string(nx_digraph):
    # Get nodes and edges from the directed graph
    nodes = set(nx_digraph.nodes())
    edges = list(nx_digraph.edges())

    # Create the string representation
    string_repr = "digraph { " + "; ".join(f"{node};" for node in nodes)

    if edges:
        string_repr += "; " + "; ".join(f"{src} -> {dest}" for src, dest in edges)

    string_repr += " }"

    return string_repr

def so_syn_dag():
    DAG = [
        'Education;',
        'Salary;',
        'Age;',
        'YearsCoding;',
        'DevType;',
        'Education -> Salary;',
        'Age -> Education;',

        'Age -> Salary;',
        'Age -> YearsCoding;',
        'Education -> YearsCoding;',
        'Education -> DevType;',
        'YearsCoding -> DevType;',
        'DevType -> Salary;',
        'YearsCoding -> Salary']

    causal_graph = """
                               digraph {
                               """
    for line in DAG:
        causal_graph = causal_graph + line + "\n"
    causal_graph = causal_graph + "}"



    return causal_graph

def syn_dag():
    DAG = [
        'A;',
        'B;',
        'C;',
        'A -> C;'
        'B -> A;',
        'B -> C;']

    causal_graph = """
                               digraph {
                               """
    for line in DAG:
        causal_graph = causal_graph + line + "\n"
    causal_graph = causal_graph + "}"



    return causal_graph

def so_dag():
    DAG = [
        'Continent;',
        'HoursComputer;',
        'UndergradMajor;',
        'FormalEducation;',
        'Age;',
        'Gender;',
        'Dependents;',
        'Country;',
        'DevType;',
        'RaceEthnicity;',
        'ConvertedSalary;',
        'Gender -> FormalEducation;',
        'Gender -> UndergradMajor;',
        'Gender -> DevType;',
        'Gender -> ConvertedSalary;',
        'Country -> ConvertedSalary;',
        'Country -> FormalEducation;',
        'Country -> RaceEthnicity;',
        'Continent -> Country; '
        'FormalEducation -> DevType;',
        'FormalEducation -> UndergradMajor;',
        'Continent -> UndergradMajor',
        'Continent -> FormalEducation;',
        'Continent -> RaceEthnicity;',
        'Continent -> ConvertedSalary;',
        'RaceEthnicity -> ConvertedSalary;',
        'UndergradMajor -> DevType;',
        'DevType -> ConvertedSalary;',
        'DevType -> HoursComputer;',
        'Age -> ConvertedSalary;',
        'Age -> DevType;',
        'Age -> Dependents;',
        'Age -> FormalEducation;',
        'Dependents -> HoursComputer;',
        'HoursComputer -> ConvertedSalary;']

    DAG = [
        'Age;',
        'Gender;',
        'DevType;',
        'FormalEducation;',
        'ConvertedSalary;',
        'Gender -> DevType;',
        'Gender -> ConvertedSalary;',
        'FormalEducation -> DevType;',
        'FormalEducation -> ConvertedSalary;',
        'DevType -> ConvertedSalary;',
        'Age -> ConvertedSalary;',
        'Age -> DevType;']

    causal_graph = """
                            digraph {
                            """
    for line in DAG:
        causal_graph = causal_graph + line + "\n"
    causal_graph = causal_graph + "}"
    causal_dag = causal_graph.replace("\n", " ")
    return causal_dag


def replicate_data(df, percentage_increase):
    # Calculate the number of tuples to replicate
    num_rows = len(df)
    replicate_rows = int(num_rows * (percentage_increase / 100))

    # Randomly select rows to replicate
    replicated_rows = df.sample(replicate_rows, replace=True)

    # Concatenate the original DataFrame with replicated rows
    replicated_df = pd.concat([df, replicated_rows], ignore_index=True)

    return replicated_df


def bin_equal_width(df, column_name, k):
    """
    Bin a given column into k equal-width bins.

    Parameters:
    - df: DataFrame
    - column_name: str, the name of the column to bin
    - k: int, the number of bins

    Returns:
    - DataFrame with an additional column 'bin' containing the bin labels
    """

    # Calculate bin edges
    _, bin_edges = pd.cut(df[column_name], bins=k, retbins=True, labels=False)

    # Bin the column
    df[column_name] = pd.cut(df[column_name], bins=bin_edges, labels=False)

    return df


def drop_columns(df, columns_to_drop):
    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Drop the specified columns from the copy
    df_copy.drop(columns=columns_to_drop, inplace=True)


    return df_copy


def remove_outliers(df, column_name, z_threshold=3):
    """
    Remove outliers from a DataFrame based on a specified column using Z-score.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column to remove outliers from.
    - z_threshold (float): Z-score threshold for identifying outliers. Default is 3.

    Returns:
    - pd.DataFrame: DataFrame with outliers removed.
    """
    # Calculate Z-score for the specified column
    z_scores = zscore(df[column_name])

    # Identify outliers based on the Z-score threshold
    outliers = (abs(z_scores) > z_threshold)

    # Remove rows with outliers
    df_no_outliers = df[~outliers]

    return df_no_outliers


def impute_mode_for_attribute(df, attribute_name):
    mode_value = mode(df[attribute_name].dropna())
    df[attribute_name] = df[attribute_name].fillna(mode_value)
    return df

def impute_mean_for_attribute(df, attribute_name):
    mean_value = df[attribute_name].mean()
    df[attribute_name] = df[attribute_name].fillna(mean_value)
    return df

def impute_min_for_attribute(df, attribute_name):
    mean_value = df[attribute_name].min()
    df[attribute_name] = df[attribute_name].fillna(mean_value)
    return df

def impute_with_value_for_attribute(df, attribute_name, impute_value):
    df[attribute_name] = df[attribute_name].fillna(impute_value)
    return df


def map_age_to_life_stage(row):
    age_range = row['Age']
    if '18 - 24 years old' in age_range:
        return 'Young Adult'
    elif '25 - 34 years old' in age_range:
        return 'Young Adult'
    elif '35 - 44 years old' in age_range:
        return 'Adult'
    elif '45 - 54 years old' in age_range:
        return 'Adult'
    elif '55 - 64 years old' in age_range:
        return 'Middle Age'
    elif '65 years or older' in age_range:
        return 'Senior'
    elif 'Under 18 years old' in age_range:
        return 'Young'
    else:
        return 'UNKNOWN'

def change_granularity(df, column_name):
    if column_name == "Age":
        df_age = df.copy()
        df_age['Age'] = df_age.apply(map_age_to_life_stage, axis=1)
        return df_age




def get_syn_df1():
    n= 10000

    df = pd.DataFrame(np.zeros((n, 3)), columns=['A', 'B', 'C'])


    df['B'] = np.random.randint(1, 5, size=n)

    df['A'] =   df['B'] + np.random.randint(1, 5, size=n)
    df['C'] =  df['B'] +np.random.randint(1, 5, size=n)

    print(df.head())
    return df
def get_syn_df():
    n= 10000

    df = pd.DataFrame(np.zeros((n, 5)), columns=['YearsCoding', 'Age', 'Education','DevType' ,'Salary'])

    df['Age'] = np.random.randint(18, 75, size=n)
    df['Education'] = 6 + 0.3 * df['Age'] - np.random.randint(1, 5, size=n)
    df['Education'] = df['Education'].astype(int)
    df['YearsCoding'] =  0.3 * df['Age'] +0.6 * df['Education'] - np.random.randint(1, 5, size = n)
    df['YearsCoding'] = df['YearsCoding'].astype(int)
    # Create the 'DevType' column based on conditions
    df['DevType'] = np.where((df['Education'] >= 15) & (df['YearsCoding'] >= 5), 1, 0)

    df['Salary'] = 1 * df['Education'] + 1 * df['Age'] + 1 * df['YearsCoding'] + 30*df['DevType']

    print(df.head())
    return df