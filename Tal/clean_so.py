import pandas as pd

FOLDER_PATH = "data/so/"
PATH_CLEAN = "../data/so/so_clean.csv"
FULL_PATH_CLEAN = FOLDER_PATH + PATH_CLEAN


def clean_so():
    df = pd.read_csv('../data/so/2018_data.csv')
    df = df[['Country',
             'Employment',
             'CompanySize',
             'YearsCoding',
             'ConvertedSalary',
             'Gender',
             'Age']]
    df = df[(df['ConvertedSalary']>1000) & (df['ConvertedSalary']<490000)]
    df_country = df['Country'].value_counts().head(10)
    df = df[ df['Country'].isin(df_country[:10].index ) ]
    employment = ['Employed full-time',
                  'Employed part-time',
                  'Independent contractor, freelancer, or self-employed']
    df = df[df['Employment'].fillna('Employed full-time').isin(employment)]
    company_size = ['Fewer than 10 employees', '10 to 19 employees', '20 to 99 employees', '100 to 499 employees', '500 to 999 employees', '1,000 to 4,999 employees', '5,000 to 9,999 employees', '10,000 or more employees']
    mapping_company_size = {key:i for i, key in enumerate(company_size)}
    df = df.dropna(subset=['CompanySize'])
    df['Gender'] = df['Gender'].fillna('Male')
    df = df[df['Gender'].isin(['Male', 'Female'])]

    # Ordered age sacle
    age = ['Under 18 years old',
           '18 - 24 years old',
           '25 - 34 years old',
           '35 - 44 years old',
           '45 - 54 years old',
           '55 - 64 years old',
           '65 years or older']
    mapping_age = {key:i for i, key in enumerate(age)}

    # Transform category to numerical column
    df['Age'] = df['Age'].fillna('25 - 34 years old')

    df.to_csv(PATH_CLEAN, index=False)


def get_secured_population(df):
    selected_rows = df.loc[~df['Gender'].str.contains("Male")]
    selected_rows.reset_index(drop=True, inplace=True)
    return selected_rows



