import pandas as pd
INPUT_DF_PATH = "data/meps/h181.csv"
import networkx as nx


def build_mini_df():
    df = pd.read_csv(INPUT_DF_PATH)
    df['group2'] = 1
    # Subpopulations
    df['Region'] = df['REGION15']
    df["Married"] = df["MARRY15X"]
    df["Student"] = df["FTSTU15X"]
    df['-14 years old'] = df['AGELAST'].apply(lambda x: 1 if x <= 14 else 0)
    df['-24 years old'] = df['AGELAST'].apply(lambda x: 1 if x <= 24 else 0)
    df['-34 years old'] = df['AGELAST'].apply(lambda x: 1 if x <= 34 else 0)
    df['-44 years old'] = df['AGELAST'].apply(lambda x: 1 if x <= 44 else 0)
    df['-54 years old'] = df['AGELAST'].apply(lambda x: 1 if x <= 54 else 0)
    df['-64 years old'] = df['AGELAST'].apply(lambda x: 1 if x <= 64 else 0)
    df['-74 years old'] = df['AGELAST'].apply(lambda x: 1 if x <= 74 else 0)
    df['-85 years old'] = df['AGELAST'].apply(lambda x: 1 if x <= 84 else 0)
    df["Race"] = df["RACEV1X"]
    df["Sex"] = df["SEX"]
    df["Education"] = df["HIDEG"]
    df["FamilyIncome"] = df["POVCAT15"]
    df["Pregnant"] = df["PREGNT53"]
    df["IsHadHeartAttack"] = df["MIDX"]
    df["IsHadStroke"] = df["STRKDX"]
    df["IsDiagnosedCancer"] = df["CANCERDX"]
    df["IsDiagnosedDiabetes"] = df["DIABDX"]
    df["IsDiagnosedAsthma"] = df["ASTHDX"]
    df["BornInUSA"] = df["BORNUSA"]
    df["ADHD/ADD_Diagnisos"] = df["ADHDADDX"]

    #Treatments
    df["DoesDoctorRecommendExercise"] = df["EXRCIS53"]
    df["LongSinceLastFluVaccination"] = df["FLUSHT53"]
    df["TakesAspirinFrequently"] = df["ASPRIN53"]
    df["Exercise"] = df["PHYEXE53"]
    df["WearsSeatBelt"] = df["SEATBE53"]
    df["FeltNervous"] = df["ADNERV42"]
    df["HoldHealthInsurance"] = df["INSURC15"]
    df["IsWorking"] = df["EMPST53"]
    df['IsUnderWeight'] = df['BMINDX53'].apply(lambda x: 1 if x <= 18.5 else 0)
    df['IsOverWeight'] = df['BMINDX53'].apply(lambda x: 1 if 25.0 <= x <= 29.9 else 0)
    df['IsObesity'] = df['BMINDX53'].apply(lambda x: 1 if x >= 30 else 0)
    df['group1'] = df['ADSMOK42'].fillna(0).apply(lambda x: 1 if x and x == 1 else 0) # smokers

    df = df.dropna()
    # filter person that diagnosed with high choloesterol
    df = df.loc[df['CHOLAGED']>0]
    return df[['group1', 'group2', 'Region', 'Married', 'Student', '-14 years old', '-24 years old', '-34 years old',
               '-44 years old', '-54 years old', '-64 years old', '-74 years old', '-85 years old', 'Race', 'Sex', 'Education',
               'FamilyIncome', 'Pregnant', 'IsHadHeartAttack', 'IsHadStroke', 'IsDiagnosedCancer',
                'IsDiagnosedAsthma', 'BornInUSA', 'ADHD/ADD_Diagnisos',
               'DoesDoctorRecommendExercise', 'LongSinceLastFluVaccination', 'TakesAspirinFrequently', 'Exercise',
               'WearsSeatBelt', 'FeltNervous', 'HoldHealthInsurance', 'IsWorking', 'IsUnderWeight',
               'IsOverWeight', 'IsObesity']]

SUBPOPULATIONS = ["Married", "Student", '-14 years old', '-24 years old', '-34 years old',
                  '-44 years old', '-54 years old', '-64 years old', '-74 years old', '-85 years old', 'Race',
                  'Sex', 'Education', 'FamilyIncome', 'Pregnant', 'IsHadHeartAttack', 'IsHadStroke',
                  'IsDiagnosedAsthma', 'BornInUSA', 'ADHD/ADD_Diagnisos']

TREATMENTS = [{"att": "DoesDoctorRecommendExercise", "value": 1}, {"att": "DoesDoctorRecommendExercise", "value": 2}, {"att": "LongSinceLastFluVaccination", "value": 1},
              {"att": "LongSinceLastFluVaccination", "value": 2}, {"att": "TakesAspirinFrequently", "value": 1},{"att": "TakesAspirinFrequently", "value": 2},
              {"att": "Exercise", "value": 1}, {"att": "Exercise", "value": 2},{"att": "WearsSeatBelt", "value": 1},{"att": "WearsSeatBelt", "value": 2},
              {"att": "FeltNervous", "value": 1},{"att": "FeltNervous", "value": 2},
              {"att": "HoldHealthInsurance", "value": 1}, {"att": "HoldHealthInsurance", "value": 2},
              {"att": "IsWorking", "value": 1}, {"att": "IsWorking", "value": 2},{"att": "IsUnderWeight", "value": 1},
              {"att": "IsOverWeight", "value": 1},{"att": "IsObesity", "value": 1},{"att": "Region", "value": 1}, {"att": "Region", "value": 2},
              {"att": "Region", "value": 3}, {"att": "Region", "value": 4}]

OUTCOME_COLUMN = "IsDiagnosedCancer"