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
df['is_player'] = df['is_player'].map(lambda x: True if x == 'TRUE' or x == 'Yes' else False)
df = df.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)
df = df.apply(lambda x: x.map(generations) if x.name == 'generation' else x)
df = df.apply(lambda x: x.map(genders) if x.name == 'gender' else x)
df = df.apply(lambda x: x.map(races) if x.name == 'race_ethnicity_primary' else x)
df = df.apply(lambda x: x.map(lgbt_identifications) if x.name == 'lgbt_identification' else x)
df['age'] = df['age'].map(lambda x: x if x > 0 else 0)
st.dataframe(df.describe())
# 
x = radius * np.cos(theta)
y = radius * np.sin(theta)

df = pd.DataFrame({
    "x": x,
    "y": y,
    "idx": indices,
    "rand": np.random.randn(num_points),
})

st.altair_chart(alt.Chart(df, height=700, width=700)
    .mark_point(filled=True)
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        color=alt.Color("idx", legend=None, scale=alt.Scale()),
        size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
    ))

# 
# forest 