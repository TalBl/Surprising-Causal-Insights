import os
import pandas as pd
import numpy as np
from folktables import ACSDataSource

def get_Multiple_States_2018_All(state_code_list=["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                                                  "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                                                  "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                                                  "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                                                  "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]):
    #Last Done for states:["SD","NE","ND","AL","MT","NH","UT", "MO", "WI", "FL", "OK", "AR", "KS", "MN", "IA", "CO", "VT", "MD", "ME", "ID"]

    final_df = pd.DataFrame()
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    for state_code in state_code_list:
        print(state_code)
        state_file_string=f"2018_{state_code}_data.csv"
        if not os.path.exists(state_file_string):
            state_data = data_source.get_data(states=[state_code], download=True)
            state_data.to_csv()
        else:
            state_data = pd.read_csv(state_file_string)
        final_df = pd.concat([final_df, state_data], ignore_index=True)
    final_df.to_csv(f"../data/acs/2018_all_data.csv", index=False)


def read_ACS_value_data():
    text = open('../data/acs/cepr_acs_2018_varlabels_plus.log', 'r').read().split("-" * 73)[2]
    lines = text.split("\n")
    field_to_value_dict = {}  # field name-> {field value code-> string}
    d = {}
    field = None
    for i, line in enumerate(lines):
        line_text = line.strip()
        if line_text == '':
            continue
        if line_text.endswith(":"):  # new field
            if field is not None:
                field_to_value_dict[field] = d
            field = line_text[:-1]
            d = {}
        elif line_text.startswith("> "):  # continuation of previous line
            d[k] += line_text[2:]
        else:
            try:
                parts = line_text.split()
                k = parts[0]
                v = " ".join(parts[1:])
                d[k] = v
            except:
                print(f"line num {i}: {line_text}")
                return
    if field is not None:
        field_to_value_dict[field] = d
    return field_to_value_dict


def make_translation_for_ACS(fields_list):
    # df = read_ACS_fields_data(year=2018)
    df = pd.read_csv('../data/acs/field_map.tsv', sep='\t')
    df['field label'] = df['field label'].apply(lambda s: s.strip('* '))
    trans = {}
    unmatched = []
    matched = 0
    unmatched_value_mapping = []
    field_to_value_dict = read_ACS_value_data()
    for field in fields_list:
        subset = df[df['field name'] == field]
        if len(subset) == 1:
            trans[field] = subset.iloc[0]['field label']
            matched += 1
        else:
            unmatched.append(field)
            continue
        # look for value mapping
        value_map_needed = subset.iloc[0]['field value map needed?']
        exclude = subset.iloc[0]['exclude?']
        if value_map_needed == 'no' or exclude == 'yes':
            continue
        elif value_map_needed == 'binary':
            if field == 'SEX':
                trans[(field, 1)] = 'Man'
                trans[(field, 2)] = 'Woman'
            else:
                trans[(field, 1)] = 'yes'
                trans[(field, 2)] = 'no'
        elif value_map_needed == 'binary(0,1)':
            trans[(field, 1)] = 'yes'
            trans[(field, 0)] = 'no'
        elif field.lower() in field_to_value_dict:
            value_to_meaning = field_to_value_dict[field.lower()]
            for v, meaning in value_to_meaning.items():
                value_for_trans = v
                if value_for_trans.isdigit():
                    value_for_trans = int(value_for_trans)
                trans[(field, value_for_trans)] = meaning
        else:  # mapping needed but not found
            unmatched_value_mapping.append(field)
    print(f"matched field names: {matched}/{len(fields_list)}. \nUnmatched: {unmatched}")
    print(f"missing value mapping for: {unmatched_value_mapping}")
    return trans

def convert_df_clean(df, dict_translation):
    df_new = pd.DataFrame()
    for column in list(df.columns):
        if column in dict_translation:
            new_column_name = dict_translation[column]
            matching_key = next((key for key in dict_translation.keys() if key[0] == column), None)
            if matching_key is not None:
                df_new[new_column_name] = df.apply(lambda row: dict_translation.get((column, row[column]), None), axis=1)
            else:
                df_new[new_column_name] = df[column]
    return df_new


def get_secured_population(df):
    selected_rows = df.loc[~df['Sex'].str.contains("Man")]
    selected_rows.reset_index(drop=True, inplace=True)
    return selected_rows


def cast_age(value):
    if value <= 16:
        return "0-16"
    if 17 <= value < 25:
        return "17-24"
    if 25 <= value < 35:
        return "25-34"
    if 35 <= value < 45:
        return "35-44"
    if 45 <= value < 55:
        return "45-54"
    if 55 <= value < 65:
        return "55-64"
    if 65 <= value < 75:
        return "65-74"
    if 75 <= value < 85:
        return "75-84"
    if 85 <= value < 95:
        return "85-94"


def cast_education(value):
    if value in ["Nursery school, preschool", "no degree",
                 "12th grade - no diploma", "GED or alternative credential", "Regular high school diploma", "Grade 9", "Grade 10",
                 "Grade 11", "Grade 12", "Some college, but less than 1 year", "Grade 2", "Grade 8", "Grade 6", "Grade 3", "Grade 7", "Grade 4", "Grade 1",
                 "1 or more years of college credit, no degree", "No schooling completed", "Kindergarten", "Grade 5"]:
        return "No Degree"
    if value == "Professional degree beyond a bachelor's degree":
        return "Professional Degree"
    return value


def cast_race(value):
    if value == "Black alone":
        return "Black or African American"
    if value in ["American Indian alone",
                 "Am Indian and Alaskan Native tribes specified",
                 "Alaskan Native alone"]:
        return "American Indian or Alaska Native"
    if value == "Asian alone":
        return "Asian"
    if value == "Native Hawaiian and other PI alone":
        return "Native Hawaiian or Other Pacific Islander"
    if value == "White alone":
        return "White"
    if value in ["2+ major race groups", "Some other race alone"]:
        return ""


def detect_outliers(df, threshold=8):
    outliers = pd.DataFrame(columns=df.columns)
    numeric_atts = ["Total person's income"]
    df = df.loc[df['When last worked'] != "Over 5 yrs ago or never worked"]
    for column in numeric_atts:
        # Calculate Z-score for each value in the column
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        # Find indices of outliers based on the threshold
        column_outliers = z_scores[z_scores > threshold].index
        # Add outliers to the DataFrame
        if column_outliers.shape[0] > 0:
            outliers = df.loc[column_outliers].reset_index(drop=True)
            df_cleaned = df.drop(column_outliers)
    return outliers, df_cleaned

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    get_Multiple_States_2018_All()
    fields_list = ['RT', 'SERIALNO', 'DIVISION', 'SPORDER', 'PUMA', 'REGION', 'ST', 'ADJINC', 'PWGTP', 'AGEP', 'CIT', 'CITWP', 'COW', 'DDRS', 'DEAR', 'DEYE', 'DOUT', 'DPHY', 'DRAT', 'DRATX', 'DREM', 'ENG', 'FER', 'GCL', 'GCM', 'GCR', 'HINS1', 'HINS2', 'HINS3', 'HINS4', 'HINS5', 'HINS6', 'HINS7', 'INTP', 'JWMNP', 'JWRIP', 'JWTR', 'LANX', 'MAR', 'MARHD', 'MARHM', 'MARHT', 'MARHW', 'MARHYP', 'MIG', 'MIL', 'MLPA', 'MLPB', 'MLPCD', 'MLPE', 'MLPFG', 'MLPH', 'MLPI', 'MLPJ', 'MLPK', 'NWAB', 'NWAV', 'NWLA', 'NWLK', 'NWRE', 'OIP', 'PAP', 'RELP', 'RETP', 'SCH', 'SCHG', 'SCHL', 'SEMP', 'SEX', 'SSIP', 'SSP', 'WAGP', 'WKHP', 'WKL', 'WKW', 'WRK', 'YOEP', 'ANC', 'ANC1P', 'ANC2P', 'DECADE', 'DIS', 'DRIVESP', 'ESP', 'ESR', 'FOD1P', 'FOD2P', 'HICOV', 'HISP', 'INDP', 'JWAP', 'JWDP', 'LANP', 'MIGPUMA', 'MIGSP', 'MSP', 'NAICSP', 'NAICSP_grouped', 'NATIVITY', 'NOP', 'OC', 'OCCP', 'OCCP_grouped', 'PAOC', 'PERNP', 'PINCP', 'POBP', 'POVPIP', 'POWPUMA', 'POWSP', 'PRIVCOV', 'PUBCOV', 'QTRBIR', 'RAC1P', 'RAC2P', 'RAC3P', 'RACAIAN', 'RACASN', 'RACBLK', 'RACNH', 'RACNUM', 'RACPI', 'RACSOR', 'RACWHT', 'RC', 'SCIENGP', 'SCIENGRLP', 'SFN', 'SFR', 'SOCP', 'VPS', 'WAOB', 'FAGEP', 'FANCP', 'FCITP', 'FCITWP', 'FCOWP', 'FDDRSP', 'FDEARP', 'FDEYEP', 'FDISP', 'FDOUTP', 'FDPHYP', 'FDRATP', 'FDRATXP', 'FDREMP', 'FENGP', 'FESRP', 'FFERP', 'FFODP', 'FGCLP', 'FGCMP', 'FGCRP', 'FHICOVP', 'FHINS1P', 'FHINS2P', 'FHINS3C', 'FHINS3P', 'FHINS4C', 'FHINS4P', 'FHINS5C', 'FHINS5P', 'FHINS6P', 'FHINS7P', 'FHISP', 'FINDP', 'FINTP', 'FJWDP', 'FJWMNP', 'FJWRIP', 'FJWTRP', 'FLANP', 'FLANXP', 'FMARP', 'FMARHDP', 'FMARHMP', 'FMARHTP', 'FMARHWP', 'FMARHYP', 'FMIGP', 'FMIGSP', 'FMILPP', 'FMILSP', 'FOCCP', 'FOIP', 'FPAP', 'FPERNP', 'FPINCP', 'FPOBP', 'FPOWSP', 'FPRIVCOVP', 'FPUBCOVP', 'FRACP', 'FRELP', 'FRETP', 'FSCHGP', 'FSCHLP', 'FSCHP', 'FSEMP', 'FSEXP', 'FSSIP', 'FSSP', 'FWAGP', 'FWKHP', 'FWKLP', 'FWKWP', 'FWRKP', 'FYOEP']
    fields = ['AGEP', 'CIT', 'MAR', 'SCHL', 'PINCP', 'DIS', 'RAC1P', 'WKL', "SEX"]
    dict_translation = make_translation_for_ACS(fields)
    df = pd.read_csv("../data/acs/2018_all_data.csv")
    df = df[fields]
    df_clean = convert_df_clean(df, dict_translation)
    df_clean = df_clean.dropna()
    df_clean['Age'] = df_clean['Age'].apply(lambda x: cast_age(x))
    df_clean['Educational attainment'] = df_clean['Educational attainment'].apply(lambda x: cast_education(x))
    df_clean['Race/Ethnicity'] = df_clean['Race/Ethnicity'].apply(lambda x: cast_race(x))
    df_clean.to_csv("../data/acs/2018_all_data_clean1.csv", index=False)
    df_clean = pd.read_csv("../data/acs/2018_all_data_clean1.csv")
    outliers, df_clean = detect_outliers(df_clean)
    outliers.to_csv("../data/acs/outliers.csv", index=False)
    df_clean.to_csv("../data/acs/2018_all_data_clean_final.csv", index=False)


