import pandas as pd

generations = {
    'Gen Z': 'Z',
    'Boomer': 'B',
    'Millennial': 'M',
    'Gen X': 'X',
    'Silent': 'S'
}

genders = {
    'Female': 'F',
    'Male': 'M',
    'Nonbinary/Other': 'N',
    'MALE': 'M',
    'femal': 'F'
}

races = {
    'White/Caucasian': 'W',
    'Hispanic': 'H',
    'Black/African American': 'B',
    'Other': 'O',
    'Asian/Pacific Islander': 'A',
    'White': 'W',
    'Native American/Alaskan': 'N'
}

lgbt_identifications = {
    'Prefer not to say': 2,
    'Yes': 1,
    'No': 0
}

def clean_df(df) -> pd.DataFrame:
    mappings = {
        'generation': generations,
        'gender': genders,
        'race_ethnicity_primary': races,
        'lgbt_identification': lgbt_identifications
    }

    clean = df.copy()
    clean = clean[clean['is_parent'] == True]
    clean = clean.assign(
        is_player = lambda x: x['is_player'].map(lambda y: True if y == 'TRUE' or y == 'Yes' else False),
        age = lambda x: x['age'].clip(lower=0)
    )

    for col, mapping in mappings.items():
        if col in clean.columns:
            clean[col] = clean[col].map(mapping)
    clean = clean.astype({col: 'int' for col in clean.select_dtypes('bool').columns})
    return clean
    