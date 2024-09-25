import streamlit as st
import pandas as pd

player_data = pd.read_csv('https://github.com/Blandalytics/baseball_snippets/blob/main/hitter_position_eligibility.csv?raw=true',encoding='latin1')

col1, col2 = st.columns(2)
with col1:
    games_started_thresh = st.number_input('Min # of Starts:',
                                           min_value=1, 
                                           max_value=50,
                                           step=1, 
                                           value=5)

with col2:
  games_played_thresh = st.number_input('Min # of Appearances:',
                                        min_value=1, 
                                        max_value=50,
                                        step=1, 
                                        value=10)

games_started_thresh = min(games_played_thresh,games_started_thresh)

pivot_df = pd.pivot_table(
    player_data.loc[(player_data['games_started']>=games_started_thresh) | 
                    (player_data['games_played']>=games_played_thresh)].assign(eligible = lambda x: ''+x['games_started'].astype('str')+' ('+x['games_played'].astype('str')+')'),
    values='eligible', 
    index=['name','mlb_player_id'], 
    columns=['position'],
    aggfunc=lambda x: x.mode().iat[0]
)[['C', '1B', '2B', 'SS', '3B', 'OF','LF',  'CF', 'RF']].reset_index()

st.dataframe(pivot_df,
             hide_index=True,
             height=(8 + 1) * 35 + 3)

players = list(pivot_df['name'].unique())
default_val = players.index('Aaron Judge')
player_select = st.selectbox('Choose a hitter:', pivot_df['name'].unique(), index=default_val)
pos_text = ', '.join(player_data.loc[player_data['name']==player_select,'position'].to_list())
st.write(f"""
{player_select}'s Eligible Positions (min {games_started_thresh} starts or {games_played_thresh} appearances):
{pos_text}
""")
