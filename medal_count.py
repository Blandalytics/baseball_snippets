import streamlit as st
import pandas as pd
import numpy as np

st.title('Olympic Medal Counts')

@st.cache_data(ttl=10*60,show_spinner="Loading medal data")
def load_data():
  tables=pd.read_html("https://en.wikipedia.org/wiki/2024_Summer_Olympics_medal_table")
  return tables[3].iloc[:-1].copy()

medal_df = load_data().rename(columns={'NOC':'Country','Total':'Medal Count'})

gold_vs_silver_weight = st.slider(
  """
  How much more valuable is a Gold Medal vs a Silver Medal?
  (2: 🥇 = 🥈🥈)
  """,
  min_value=1,
  max_value=50,
  value=2,
)
silver_vs_bronze_weight =  st.slider(
  """
  How much more valuable is a Silver Medal vs a Bronze Medal?
  (2: 🥈 = 🥉🥉)
  """,
  min_value=1,
  max_value=50,
  value=2
)
gold_vs_bronze_weight = gold_vs_silver_weight * silver_vs_bronze_weight

medal_df['weighted_Silver'] = medal_df['Silver'].div(gold_vs_silver_weight)
medal_df['weighted_Bronze'] = medal_df['Bronze'].div(gold_vs_bronze_weight)
medal_df['Weighted Count'] = medal_df[['Gold','weighted_Silver','weighted_Bronze']].astype('float').sum(axis=1)

st.dataframe(medal_df[['Country','Gold','Silver','Bronze','Medal Count','Weighted Count']].sort_values('Weighted Count',ascending=False).round({'Weighted Count':1}))
