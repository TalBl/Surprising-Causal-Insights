import pandas as pd


def convert_to_yearly(row):
    if row['SalaryType'] == 'Weekly':
        return row['ConvertedSalary'] * 52  # Assuming 52 weeks per year
    elif row['SalaryType'] == 'Monthly':
        return row['ConvertedSalary'] * 12  # Assuming 12 months in a year
    else:  # Already yearly
        return 0


INPUT_DF_PATH = "data/so/2018_data.csv"
def build_mini_df():
    df = pd.read_csv(INPUT_DF_PATH)
    df['group1'] = df['Gender'].fillna("").apply(lambda x: 1 if x == "Female" else 0)
    df['group2'] = df['Gender'].fillna("").apply(lambda x: 1 if x == "Male" else 0)
    df['ConvertedCompYearly'] = df.apply(convert_to_yearly, axis=1)
    df = df.loc[(df["group1"] == 1) | (df["group2"] == 1)]
    df = df[['group1', 'group2', 'YearsCodingProf', 'FormalEducation','DevType','Age','EducationParents','MilitaryUS','UndergradMajor',
             'RaceEthnicity', 'Country', 'JobSatisfaction', 'Hobby', 'Student','JobSearchStatus',
             'LastNewJob', 'HopeFiveYears', 'WakeTime', 'Exercise', 'ConvertedCompYearly']]
    df = df.dropna(subset=['ConvertedCompYearly'])
    return df


SUBPOPULATIONS = ['Age', 'RaceEthnicity', 'Country','Student','UndergradMajor']

TREATMENTS = [{"att": "YearsCodingProf", "value": lambda x: 1 if pd.notna(x) and x == "0-2 years" else 0},
              {"att": "YearsCodingProf", "value": lambda x: 1 if pd.notna(x) and x in ["0-2 years","3-5 years"] else 0},
              {"att": "YearsCodingProf", "value": lambda x: 1 if pd.notna(x) and x in ["0-2 years","3-5 years","6-8 years"] else 0},
              {"att": "YearsCodingProf", "value": lambda x: 1 if pd.notna(x) and x in ["0-2 years","3-5 years","6-8 years","9-11 years"] else 0},
              {"att": "FormalEducation", "value": lambda x: 1 if pd.notna(x) and 'Master' in x else 0},
              {"att": "FormalEducation", "value": lambda x: 1 if pd.notna(x) and 'Bachelor' in x else 0},
              {"att": "DevType", "value": lambda x: 1 if pd.notna(x) and 'Data scientist' in x else 0},
              {"att": "DevType", "value": lambda x: 1 if pd.notna(x) and 'Developer, back-end' in x else 0},
              {"att": "Hobby", "value": lambda x: 1 if pd.notna(x) and x == "Yes" else 0},
              {"att": "DevType", "value": lambda x: 1 if pd.notna(x) and 'Full-stack developer' in x else 0},
              {"att": "DevType", "value": lambda x: 1 if pd.notna(x) and 'Engineering manager' in x else 0},
              {"att": "JobSatisfaction", "value": lambda x: 1 if pd.notna(x) and 'Extremely satisfied' in x else 0},
              {"att": "LastNewJob", "value": lambda x: 1 if pd.notna(x) and 'More than 4 years ago' in x else 0},
              {"att": "HopeFiveYears", "value": lambda x: 1 if pd.notna(x) and 'Working as a founder' in x else 0},
              {"att": "WakeTime", "value": lambda x: 1 if pd.notna(x) and 'Between 7:01 - 8:00 AM' in x else 0},
              {"att": "WakeTime", "value": lambda x: 1 if pd.notna(x) and 'Between 5:00 - 6:00 AM' in x else 0},
              {"att": "Exercise", "value": lambda x: 1 if pd.notna(x) and '3 - 4 times per week' in x else 0}]

OUTCOME_COLUMN = "ConvertedCompYearly"

