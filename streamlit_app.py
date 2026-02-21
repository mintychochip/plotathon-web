import numpy as np
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

url = 'https://raw.githubusercontent.com/Noah-Gallego/Plot-A-Thon/main/gamer_study_raw.csv'

df = pd.read_csv(url)
st.write("Before Clean")
st.dataframe(df.head(10))
st.dataframe(df.describe())
st.write("After Clean")

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
mappings = {
    'generation': generations,
    'gender': genders,
    'race_ethnicity_primary': races,
    'lgbt_identfication': lgbt_identifications
}
df = df.assign(
    is_player = lambda x: x['is_player'].map(lambda y: True if y == 'TRUE' or y == 'Yes' else False),
    age = lambda x: x['age'].clip(lower=0)
)
for col, mapping in mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)
df = df.astype({col: 'int' for col in df.select_dtypes('bool').columns})
st.dataframe(df.head(10))
st.dataframe(df.describe())
#
# 
# forest 