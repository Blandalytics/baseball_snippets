import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='Hitter Eligibility', page_icon='⚾')
st.title('MLB Hitter Eligibility (2024)')

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

pivot_df = (
    pd.pivot_table(
        player_data.assign(games_started = lambda x: np.clip(games_started_thresh-x['games_started'],0,200),
                           games_played = lambda x: np.clip(games_played_thresh-x['games_played'],0,200)),
        values=['games_started','games_played'], 
        index=['name','mlb_player_id'], 
        columns=['position'],
        aggfunc='mean'
    )
    [[('games_started', 'C'),('games_played', 'C'),
     ('games_started', '1B'),('games_played', '1B'),
     ('games_started', '2B'),('games_played', '2B'), 
     ('games_started', '3B'),('games_played', '3B'),
     ('games_started', 'SS'),('games_played', 'SS'),
     ('games_started', 'OF'),('games_played', 'OF')]]
    .reset_index()
    .replace({0:'E'})
)
pivot_df.columns = ['Name','MLBAMID',
                   'st_C','pl_C',
                   'st_1B','pl_1B',
                   'st_2B','pl_2B',
                   'st_3B','pl_3B',
                   'st_SS','pl_SS',
                   'st_OF','pl_OF',]

players = list(pivot_df['Name'].unique())
default_val = players.index('Aaron Judge')
player_select = st.selectbox('Choose a hitter:', pivot_df['Name'].unique(), index=default_val)
pos_text = ', '.join(player_data.loc[player_data['name']==player_select,'position'].to_list())
st.write(f"""
{player_select}'s Eligible Positions (min {games_started_thresh} starts or {games_played_thresh} appearances):

{pos_text}
""")

st.write('Games remaining until eligible (E)')
st.dataframe(pivot_df
             .fillna('F')
             .style.format(precision=0, thousands='')
             .map(lambda x: 'color: transparent; background-color: transparent' if x=='F' else '')
             .map(lambda x: 'background-color: g' if x=='E' else ''),
             hide_index=True,
             height=(8 + 1) * 35 + 3)
