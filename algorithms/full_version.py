import pandas as pd
import numpy as np
from time import time
from algorithms.divexplorer.divexplorer import DivergenceExplorer
from algorithms.find_best_treatment import find_best_treatment
from algorithms.new_greedy import find_group, calc_facts_metrics
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

PRECENT_CLEANING_THRESHOLD = 0.5
DIVEXPLORER_THRESHOLD_SUPPORT = 0.9
PROJECT_DIRECTORY = "so"
K = 5
LAMDA_VALUES = [0.0001, 0.00005, 0.00009]
INPUT_DF_PATH = "../data/so/2022_data.csv"
PROTECTED_COLUMN = "Gender"
OUTCOME_COLUMN = "ConvertedCompYearly"
CALCED_INTERSECTIONS = {}


def remove_high_nan_valued_columns(df):
    # Drop columns with more than 50% of the rows that are Nan
    print(f"Number of columns before dropping high nan columns: {df.shape[1]}")
    df = df.dropna(thresh=df.shape[0]*PRECENT_CLEANING_THRESHOLD, axis=1)
    print(f"Number of columns after dropping high nan columns: {df.shape[1]}")
    return df


def detect_outliers(df, threshold=8):
    outliers = pd.DataFrame(columns=df.columns)
    # Find numeric columns
    numeric_atts = df.select_dtypes(include=[np.number]).columns
    df_cleaned = df.copy()
    print(f"Number of rows before dropping outliers: {df.shape[0]}")
    for column in numeric_atts:
        # Calculate Z-score for each value in the column
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        # Find indices of outliers based on the threshold
        column_outliers = z_scores[z_scores > threshold].index
        # Add outliers to the DataFrame
        if column_outliers.shape[0] > 0:
            outliers = df.loc[column_outliers].reset_index(drop=True)
            df_cleaned = df.drop(column_outliers)
    print(f"Number of rows after dropping outliers: {df_cleaned.shape[0]}")
    return outliers, df_cleaned


def split_and_one_hot_encode_all(df, sep=';'):
    for column in df.columns:
        # Check if the column contains any semi-colon separated values
        if df[column].astype(str).str.contains(sep).any():
            # Split the column and get one-hot encoded columns
            one_hot_encoded = df[column].str.get_dummies(sep=sep)
            one_hot_encoded = one_hot_encoded.add_suffix(f"_{column}")

            # Drop the original column and join the one-hot encoded columns
            df = df.drop(column, axis=1).join(one_hot_encoded)
    print(f"now dataframe has {df.shape[1]}")
    df = df.dropna()
    # remove all rows that their value is rare in 5% of the data

    return df


def clean_data(df):
    df = df.loc[:, df.nunique() != 1]
    df = remove_high_nan_valued_columns(df)
    _, df = detect_outliers(df)
    df = split_and_one_hot_encode_all(df)
    df = filter_rare_categories(df)
    return df

def print_explanation():
    return


def filter_rare_categories(df, threshold=0.01):
    # Get all columns
    all_columns = df.columns

    # Get numeric columns
    numeric_columns = df._get_numeric_data().columns

    # Categorical columns are the ones not in numeric columns
    categorical_columns = [col for col in all_columns if col not in numeric_columns]

    # Iterate over each categorical column
    for col in categorical_columns:
        # Calculate the frequency of each category
        freq = df[col].value_counts(normalize=True)

        # Identify categories that are less frequent than the threshold
        rare_categories = freq[freq < min(threshold, 1/len(freq))].index

        # Filter out rows with rare categories
        df = df[~df[col].isin(rare_categories)]

    return df


def algorithm(df_raw):
    start = time()
    df_raw = pd.read_csv(INPUT_DF_PATH)
    df_clean = clean_data(df_raw)
    df_clean.to_csv(f"../outputs/{PROJECT_DIRECTORY}/clean_data.csv", index=False)
    end_step_0 = time()
    #print(f"step 0: {end_step_0-start}")
    DEMOGRAPHICS_COLUMNS = ["Country", "Age", "Trans","Prefer not to say_Ethnicity", "African_Ethnicity","Asian_Ethnicity","Biracial_Ethnicity","Black_Ethnicity","Caribbean_Ethnicity","Central American_Ethnicity","Central Asian_Ethnicity","East Asian_Ethnicity","Ethnoreligious group_Ethnicity","European_Ethnicity","Hispanic or Latino/a_Ethnicity","I don't know_Ethnicity","Indian_Ethnicity","Indigenous (such as Native American or Indigenous Australian)_Ethnicity","Middle Eastern_Ethnicity","Multiracial_Ethnicity","North African_Ethnicity","North American_Ethnicity","Pacific Islander_Ethnicity","South American_Ethnicity","South Asian_Ethnicity","Southeast Asian_Ethnicity","White_Ethnicity","I am blind / have difficulty seeing_Accessibility","I am deaf / hard of hearing_Accessibility","I am unable to / find it difficult to type_Accessibility","I am unable to / find it difficult to walk or stand without assistance_Accessibility","None of the above_Accessibility","I have a concentration and/or memory disorder (e.g., ADHD, etc.)_MentalHealth","I have a mood or emotional disorder (e.g., depression, bipolar disorder, etc.)_MentalHealth","I have an anxiety disorder_MentalHealth","I have autism / an autism spectrum disorder (e.g. Asperger's, etc.)_MentalHealth","I have learning differences (e.g., Dyslexic, Dyslexia, etc.)_MentalHealth","None of the above_MentalHealth"]

    # step 1 - get best subpopulation by divexplorer
    df_clean = pd.read_csv(f"../outputs/{PROJECT_DIRECTORY}/clean_data.csv")
    df_clean = df_clean.drop(DEMOGRAPHICS_COLUMNS, axis=1)
    fp_diver = DivergenceExplorer(df_clean)
    subgroups = fp_diver\
        .get_pattern_divergence(min_support=DIVEXPLORER_THRESHOLD_SUPPORT, quantitative_outcomes=[OUTCOME_COLUMN])\
        .sort_values(by=f"{OUTCOME_COLUMN}_div", ascending=False, ignore_index=True)
    subgroups = subgroups.dropna()
    subgroups.head(500).to_csv(f"../outputs/{PROJECT_DIRECTORY}/interesting_subpopulations.csv", index=False)
    end_step_1 = time()
    print(f"step 1:{end_step_1-end_step_0}")

    # step 2 - find the best treatment for each subpopulation
    df_clean = pd.read_csv(f"../outputs/{PROJECT_DIRECTORY}/clean_data.csv")
    subgroups = pd.read_csv(f"../outputs/{PROJECT_DIRECTORY}/interesting_subpopulations.csv")
    intresting_subpopulations = list(subgroups['itemset'])
    r_dict = []
    for itemset in tqdm(intresting_subpopulations, total=len(intresting_subpopulations)):
        r = find_best_treatment(df=df_clean, item_set=itemset, protected_att="Woman_Gender", protected_val=1,
                                output_att="ConvertedCompYearly", treatments=DEMOGRAPHICS_COLUMNS)
        r_dict.append({"itemset": itemset, "treatment": r})
    pd.DataFrame(data=r_dict).to_csv(f"../outputs/{PROJECT_DIRECTORY}/subpopulations_and_treatments.csv", index=False)
    end_step2 = time()
    print(f"step 2: {end_step2-end_step_1}")
    # step 3 - find the best group with greedy algorithm
    df_clean = pd.read_csv(f"../outputs/{PROJECT_DIRECTORY}/clean_data.csv")
    df_metadata = pd.read_csv(f"../outputs/{PROJECT_DIRECTORY}/subpopulations_and_treatments.csv")
    calc_facts_metrics(data=df_clean, meta_data=df_metadata)
    df_facts = pd.read_csv(f"../outputs/{PROJECT_DIRECTORY}/all_facts.csv")
    find_group(df_clean, df_facts)
    end_step3 = time()
    print(f"step 3: {end_step3-end_step2}")

algorithm(1)