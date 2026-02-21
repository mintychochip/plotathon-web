import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import xgboost as xgb
import classifier
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

mappings = {
    'generation': generations,
    'gender': genders,
    'race_ethnicity_primary': races
}
df = df.assign(
    is_player = lambda x: x['is_player'].map(lambda y: True if y == 'TRUE' or y == 'Yes' else False),
    age = lambda x: x['age'].clip(lower=0)
)
for col, mapping in mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)
del df['lgbt_identification']
df = df.astype({col: 'int' for col in df.select_dtypes('bool').columns})
st.dataframe(df.head(10))
st.dataframe(df.describe())
X = df.drop(columns = ['children_play_games'])
y = df['children_play_games']
st.write("X Dataset")
st.dataframe(X.head(10))
st.dataframe(X.describe())
st.write("Y Dataset")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Y Dataset")
    st.dataframe(y.head(10))
with c2:
    st.subheader("Y Summary Statistics")
    st.dataframe(y.describe())

objs = ['generation','gender','race_ethnicity_primary']
X = pd.get_dummies(X ,columns=objs, dummy_na=True)
classifier.create_classifier(X, y)
# 
# forest 
