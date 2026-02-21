import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype
import numpy as np
from clean import clean_df, generations, genders, races, lgbt_identifications
from classifier import Classifier
data_url = 'https://raw.githubusercontent.com/Noah-Gallego/Plot-A-Thon/main/gamer_study_raw.csv'
df = pd.read_csv(data_url)
clean = clean_df(df)
rows = 5

def graph_radar(df, features):
    df_numeric = df[features].astype(float)
    sns.set_style("whitegrid")
    yes = df_numeric[df['children_play_games'] == 1][features].mean()
    no = df_numeric[df['children_play_games'] == 0][features].mean()
    feat_min = df_numeric.min()
    feat_max = df_numeric.max()
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1
    group_yes_norm = ((yes - feat_min) / feat_range) * 100
    group_no_norm  = ((no - feat_min) / feat_range) * 100
    labels = [f.replace('_', ' ').title() for f in features]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    values_yes = group_yes_norm.tolist() + [group_yes_norm.tolist()[0]]
    values_no  = group_no_norm.tolist() + [group_no_norm.tolist()[0]]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    ax.plot(angles, values_yes, 'o-', linewidth=2, label='Parent Allows Child\nTo Game', color='#2196F3')
    ax.fill(angles, values_yes, alpha=0.15, color='#2196F3')

    ax.plot(angles, values_no, 'o-', linewidth=2, label="Parent does NOT Allow Child\nTo Game", color='#FF5722')
    ax.fill(angles, values_no, alpha=0.15, color='#FF5722')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10, ha='center')
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(['20%', '40%', '60%', '80%'], size=7, color='gray')
    ax.set_title('Contibuting Factors: \nParents Allowing Games vs Parents who Don\'t', size=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), frameon=True)

    plt.tight_layout()
    plt.savefig('radar_chart.png', dpi=150, bbox_inches='tight')
    plt.show()
    return plt

def graph_heatmap(X, features):
    sns.set_style("whitegrid")
    labels = (features.str.replace('_', ' ', regex=False).str.title())
    corr = X[features].corr()
    _, ax = plt.subplots(figsize=(10,8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1, vmax=1,
        linewidths=1,
        linecolor='white',
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
        ax=ax
    )

    ax.set_title('How Parental Gaming Behaviors Correlate', fontsize=15, fontweight='bold', pad=15)
    ax.tick_params(labelsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()
    return plt

def graph_distribution(df, bins):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        df['age'].dropna(),
        bins=bins,
        kde=True,
        color='#2196F3',
        edgecolor='white',
        linewidth=0.5,
        alpha=0.7,
        line_kws={'linewidth': 2.5, 'color': '#0D47A1'},
        ax=ax
    )

    mean_age = df['age'].mean()
    ax.axvline(mean_age, color='#FF5722', linestyle='--', label=f'Mean Age: {mean_age:.1f}')

    median_age = df['age'].median()
    ax.axvline(median_age, color='#FFC107', linestyle='--', label=f'Median Age: {median_age:.1f}')

    ax.set_title('Distribution of Parent Age', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(fontsize=11, frameon=True, facecolor='white', edgecolor='gray')
    ax.tick_params(labelsize=10)
    sns.despine()

    plt.tight_layout()
    plt.show()
    return plt

with st.sidebar:
    selected = st.radio(
        "Navigation",
        ["Home","Data Cleaning", "Feature Classification"],
        label_visibility="collapsed"
    )
if selected == 'Home':
    st.title('Home')
    st.write("""
            2/21/2026 \n
            Team #11 \n
            Plot-A-Thon @ CSUCI \n

            Welcome! This is the main page for our plot-a-thon project \n
            Members: Rodney Aguirre, Noah Gallego, Justin Lo, Brooklyn Stitt
                """)
if selected == 'Data Cleaning':
    st.title("Before Clean")
    st.dataframe(df.head(rows))
    st.write("Mappings:", generations, genders, races, lgbt_identifications)
    st.title("After Clean")
    st.dataframe(clean.head(rows))
if selected == 'Feature Classification':
    X = clean.drop(columns = ['children_play_games'])
    y = clean['children_play_games']
    c1, c2 = st.columns(2)
    with c1:
        st.header("X Dataset")
        st.dataframe(X.head(rows))
    with c2:
        st.header("Y Dataset")
        st.dataframe(y.head(rows))
    obj = ['generation','gender','race_ethnicity_primary']
    features = X[obj].copy()
    X = pd.get_dummies(X, columns=obj, dummy_na=True)
    oht_col_names = [c for c in X.columns if any(c.startswith(p) for p in obj)]
    oht_features = X[oht_col_names]
    st.write("One Hot Encoded Features:", obj)
    c1, c2 = st.columns(2)
    with c1:
        st.header("Before One-Hot Encoding")
        st.dataframe(features.head(10))
    with c2:
        st.header("After One-Hot Encoding")
        st.dataframe(oht_features.head(10))
    r_state = st.slider(label="Random State", min_value=1, max_value=100, value=42)
    cv = st.toggle(label="Enable Cross-Validation (CV)")
    clf = Classifier(X, y, random_state=r_state)
    if not cv:
        clf.train_simple()
    else:
        with st.expander("Open Hyperparameter Settings", expanded=False):
            param_dist = {
                "max_depth": st.multiselect("Max Depth", range(1,13), default=range(3,6)),
                "learning_rate": st.multiselect("Learning Rate", options=[0.001,0.01,0.05,0.1,0.2], default=[0.01,0.05,0.1]),
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0]
            }
        clf.train_cv(param_dist)    
    features = clf.get_feature_list()
    st.write("Classified Feature List:", features.head(15))
    n15 = features.head(15)
    plt.figure(figsize=(10,8))
    sns.set_theme(style='whitegrid')
    sns.barplot(
        data=n15,
        x='Importance',
        y='Feature',
        palette='viridis',
        legend=False
    )
    plt.title('What Predicts Whether Children Play Games?', fontsize=14)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    st.pyplot(plt)
    with st.status("Graphing Distribution...", expanded=False) as status:
        st.write("Initialized Distribution...")
        bins = st.slider(label="Buckets", min_value=1, max_value=25, value=25)
        st.pyplot(graph_distribution(clean, bins))
        st.write("Finished Graphing!")
        status.update(label="Completed Graphing Distribution", state="complete", expanded=True)
    with st.status("Graphing How Parental Gaming Behaviors Correlate...", expanded=False) as status:
        st.write("Initialized Heatmap...")
        feature_count = st.slider(label="Feature Count", min_value=1, max_value=20, value=10)
        st.pyplot(graph_heatmap(X,features['Feature'].head(feature_count)))
        st.write("Finished Graphing!")
        status.update(label="Completed How Parental Gaming Behaviors Correlate", state="complete", expanded=True)
    with st.status("Graphing Contributing Factors...", expanded=False) as status:
        st.write("Initialized Radar...")
        
        radar_data = X.copy()
        radar_data['children_play_games'] = y.values
        radar_data = radar_data.loc[
            :, ~radar_data.columns.str.contains("att|teach", case=False)
        ]
        
        options = [c for c in radar_data.columns if c != "children_play_games"]

        def_features = [
            'esrb_aware',
            'parent_uses_parental_controls',
            'esrb_regular_use',
            'parent_plays_with_children',
            'is_player',
            'age',
            'generation',
            'plays_console',
            'plays_pc',
            'plays_action',
        ]

        default = [d for d in def_features if d in options][:8]

        selected_features = st.multiselect(
            label="Features",
            options=options,
            default=default,
            max_selections=8,
            key="radar_features"
        )
        
        st.pyplot(graph_radar(radar_data, selected_features))
        st.write("Finished Graphing!")
        status.update(label="Completed Graphing Contributing Factors", state="complete", expanded=True)
    
