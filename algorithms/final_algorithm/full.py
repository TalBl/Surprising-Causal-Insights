import pandas as pd
import numpy as np
from time import time
from algorithms.divexplorer.divexplorer import DivergenceExplorer
from algorithms.final_algorithm.find_best_treatment import find_best_treatment
from algorithms.final_algorithm.new_greedy import find_group, calc_facts_metrics
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

PRECENT_CLEANING_THRESHOLD = 0.5
DIVEXPLORER_THRESHOLD_SUPPORT = 0.01
PROJECT_DIRECTORY = "heart"
K = 5
LAMDA_VALUES = [0.0001, 0.00005, 0.00009]
INPUT_DF_PATH = "data/so/2022_data.csv"
OUTCOME_COLUMN = "HeartDisease"
CALCED_INTERSECTIONS = {}
GROUP_1 = [{"att": "ChestPainType_TA", "value": 1}]
GROUP_2 = [{"att": "ChestPainType_ASY", "value": 1}]


def detect_outliers(df, threshold=8):
    outliers = pd.DataFrame(columns=df.columns)
    numeric_atts = df.select_dtypes(include=[np.number]).columns
    df_cleaned = df.copy()
    for column in numeric_atts:
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        column_outliers = z_scores[z_scores > threshold].index
        if column_outliers.shape[0] > 0:
            outliers = df.loc[column_outliers].reset_index(drop=True)
            df_cleaned = df.drop(column_outliers)
    return outliers, df_cleaned


def split_and_one_hot_encode_all(df, sep=';'):
    # where the column is multi-value - we will split it to one-hot vector
    for column in df.columns:
        if df[column].astype(str).str.contains(sep).any():
            one_hot_encoded = df[column].str.get_dummies(sep=sep)
            one_hot_encoded = one_hot_encoded.add_suffix(f"_{column}")
            df = df.drop(column, axis=1).join(one_hot_encoded)
    df = df.dropna()
    return df


def filter_rare_categories(df, threshold=0.01):
    all_columns = df.columns
    numeric_columns = df._get_numeric_data().columns
    categorical_columns = [col for col in all_columns if col not in numeric_columns]
    for col in categorical_columns:
        freq = df[col].value_counts(normalize=True)
        rare_categories = freq[freq < min(threshold, 1/len(freq))].index
        df = df[~df[col].isin(rare_categories)]
    return df


def clean_data(df):
    df = df.loc[:, df.nunique() != 1]
    df = df.dropna(thresh=df.shape[0]*PRECENT_CLEANING_THRESHOLD, axis=1) # drop columns that have too much Null values
    _, df = detect_outliers(df)
    df = split_and_one_hot_encode_all(df)
    df = filter_rare_categories(df)
    return df


def build_mini_df():
    df_raw = pd.read_csv(INPUT_DF_PATH)
    df = df_raw["Country"].str.get_dummies()
    df_raw = df_raw.drop("Country", axis=1).join(df)
    df_raw = df_raw[[GROUP_1[0]["att"], GROUP_2[0]["att"], OUTCOME_COLUMN, "DevType", "Gender", "Ethnicity", "Age", "MentalHealth", "EdLevel"]]
    df = df_raw["DevType"].str.get_dummies(sep=";")[["Developer, back-end","Engineer, data", "Student"]]
    df_raw = df_raw.drop("DevType", axis=1).join(df)
    df = df_raw.loc[df_raw["Gender"].isin(["Man", "Woman"])]
    df1 = df["Ethnicity"].str.get_dummies(sep=";")[["White", "Asian", "Black"]]
    df = df.drop("Ethnicity", axis=1).join(df1)
    df = df.loc[df["Age"].isin(["35-44 years old", "25-34 years old", "45-54 years old"])]
    df = df.loc[df["MentalHealth"].isin(["None of the above", "I have an anxiety disorder", "I have a mood or emotional disorder (e.g., depression, bipolar disorder, etc.)"])]
    df1 = df["EdLevel"].str.get_dummies(sep=";")[["Master’s degree (M.A., M.S., M.Eng., MBA, etc.)", "Bachelor’s degree (B.A., B.S., B.Eng., etc.)", "Primary/elementary school"]]
    df = df.drop("EdLevel", axis=1).join(df1)
    df = df.dropna()
    return df

def algorithm():
    #df_raw = pd.read_csv(INPUT_DF_PATH)
    #df_clean = clean_data(df_raw)
    #df_clean.to_csv(f"outputs/{PROJECT_DIRECTORY}/clean_data.csv", index=False)
    #DEMOGRAPHICS_COLUMNS = ["Gender", "Age", "Trans","Prefer not to say_Ethnicity", "African_Ethnicity","Asian_Ethnicity","Biracial_Ethnicity","Black_Ethnicity","Caribbean_Ethnicity","Central American_Ethnicity","Central Asian_Ethnicity","East Asian_Ethnicity","Ethnoreligious group_Ethnicity","European_Ethnicity","Hispanic or Latino/a_Ethnicity","I don't know_Ethnicity","Indian_Ethnicity","Indigenous (such as Native American or Indigenous Australian)_Ethnicity","Middle Eastern_Ethnicity","Multiracial_Ethnicity","North African_Ethnicity","North American_Ethnicity","Pacific Islander_Ethnicity","South American_Ethnicity","South Asian_Ethnicity","Southeast Asian_Ethnicity","White_Ethnicity","I am blind / have difficulty seeing_Accessibility","I am deaf / hard of hearing_Accessibility","I am unable to / find it difficult to type_Accessibility","I am unable to / find it difficult to walk or stand without assistance_Accessibility","None of the above_Accessibility","I have a concentration and/or memory disorder (e.g., ADHD, etc.)_MentalHealth","I have a mood or emotional disorder (e.g., depression, bipolar disorder, etc.)_MentalHealth","I have an anxiety disorder_MentalHealth","I have autism / an autism spectrum disorder (e.g. Asperger's, etc.)_MentalHealth","I have learning differences (e.g., Dyslexic, Dyslexia, etc.)_MentalHealth","None of the above_MentalHealth"]
    SUBPOPULATIONS = [{"att": "Sex", "value": "F"}, {"att": "Sex", "value": "M"}, {"att": "Age", "value": "30-40"}, {"att": "Age", "value": "40-50"}
        , {"att": "Age", "value": "50-60"}, {"att": "Age", "value": "60-70"}, {"att": "Age", "value": "70-80"}]
    TREATMENTS = [{"att": "Cholesterol", "value": "<120"}, {"att": "Cholesterol", "value": "120-180"}, {"att": "Cholesterol", "value": "180-240"},
                  {"att": "Cholesterol", "value": "240-300"}, {"att": "Cholesterol", "value": "300-360"}, {"att": "Cholesterol", "value": ">360"}
        ,{"att": "ExerciseAngina", "value": 1},
                  {"att": "ExerciseAngina", "value": 0}]
    #df_clean = build_mini_df()
    #df_clean.to_csv(f"outputs/{PROJECT_DIRECTORY}/clean_data.csv", index=False)
    # step 1 - get best subpopulation by divexplorer
    df_clean = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/clean_data.csv")
    df_clean = df_clean.loc[(df_clean[GROUP_1[0]["att"]] == GROUP_1[0]["value"]) | (df_clean[GROUP_2[0]["att"]] == GROUP_2[0]["value"])]
    fp_diver = DivergenceExplorer(df_clean)
    subgroups = fp_diver\
        .get_pattern_divergence(min_support=DIVEXPLORER_THRESHOLD_SUPPORT, quantitative_outcomes=[OUTCOME_COLUMN],
                                group_1_column=GROUP_1[0]["att"], group_2_column=GROUP_2[0]["att"], attributes=list(set([x["att"] for x in SUBPOPULATIONS])))\
        .sort_values(by=f"{OUTCOME_COLUMN}_div", ascending=False, ignore_index=True)
    subgroups = subgroups.dropna()
    subgroups.to_csv(f"outputs/{PROJECT_DIRECTORY}/interesting_subpopulations.csv", index=False)

    # step 2 - find the best treatment for each subpopulation
    df_clean = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/clean_data.csv")
    subgroups = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/interesting_subpopulations.csv")
    intresting_subpopulations = list(subgroups['itemset'])
    r_dict = []
    for itemset in tqdm(intresting_subpopulations, total=len(intresting_subpopulations)):
        r = find_best_treatment(df=df_clean, group1=GROUP_1, group2=GROUP_2, item_set=itemset,
                                output_att="HeartDisease", treatments=TREATMENTS)
        if not r:
            continue
        r_dict.append({"itemset": itemset, "treatment": r})
    pd.DataFrame(data=r_dict).to_csv(f"outputs/{PROJECT_DIRECTORY}/subpopulations_and_treatments.csv", index=False)

    # step 3 - find the best group with greedy algorithm
    df_clean = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/clean_data.csv")
    df_metadata = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/subpopulations_and_treatments.csv")
    calc_facts_metrics(data=df_clean, group1=GROUP_1, group2=GROUP_2, meta_data=df_metadata)
    df_facts = pd.read_csv(f"outputs/{PROJECT_DIRECTORY}/all_facts.csv")
    find_group(df_clean, df_facts)


algorithm()