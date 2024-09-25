import streamlit as st
import pandas as pd
import numpy as np
import urllib
from PIL import Image
from scipy import stats
st.set_page_config(page_title='Hitter Eligibility', page_icon='⚾')

logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
logo = Image.open(urllib.request.urlopen(logo_loc))
st.image(logo, width=200)

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

if games_played_thresh<games_started_thresh:
    st.write('Games Played threshold lower than Games Started threshold. Using Games Played threshold for both.')
    games_started_thresh = games_played_thresh

player_data['games_played'] = player_data['games_played'].fillna(player_data['games_started'])
pivot_df = (
    pd.pivot_table(
        player_data
        .assign(games_started = lambda x: np.clip(games_started_thresh-x['games_started'],0,games_played_thresh+1),
                games_played = lambda x: np.clip(games_played_thresh-x['games_played'],0,games_played_thresh+1))
        .assign(games_started = lambda x: np.where(x['games_played']==0,0,x['games_started']),
                games_played = lambda x: np.where(x['games_started']==0,0,x['games_played'])),
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
)
pivot_df.columns = ['Name','MLBAMID',
                   'C_s','C_a',
                   '1B_s','1B_a',
                   '2B_s','2B_a',
                   '3B_s','3B_a',
                   'SS_s','SS_a',
                   'OF_s','OF_a',]

pos_sort = {
    'C':0,
    '1B':1,
    '2B':2,
    '3B':3,
    'SS':4,
    'OF':5,
    'LF':6,
    'CF':7,
    'RF':8
}

players = list(pivot_df['Name'].unique())
default_val = players.index('Aaron Judge')
player_select = st.selectbox('Choose a hitter:', pivot_df['Name'].unique(), index=default_val)
pos_list = player_data.loc[player_data['name']==player_select,'position'].to_list()
pos_list.sort(key=lambda x: pos_sort[x])
pos_text = ', '.join(pos_list)
st.write(f"""
{player_select}'s Eligible Positions (min {games_started_thresh} starts or {games_played_thresh} appearances):

{pos_text}


""")

fill_val = games_started_thresh+1
st.header('Games remaining until eligible')
st.write('s=Starts; a=Appearances')
st.dataframe(pivot_df
             .fillna(fill_val)
             .style
             .format(precision=0, thousands='')
             .background_gradient(axis=0, vmin=-3, vmax=games_started_thresh, cmap="Greens_r", subset=['C_s','1B_s','2B_s','3B_s','SS_s','OF_s'])
             .background_gradient(axis=0, vmin=-3, vmax=games_played_thresh, cmap="Greens_r", subset=['C_a','1B_a','2B_a','3B_a','SS_a','OF_a'])
             .map(lambda x: 'color: transparent; background-color: transparent' if x==fill_val else '')
             .map(lambda x: 'color: green; background-color: green' if x==0 else ''),
             hide_index=True,
             height=(8 + 1) * 35 + 3)
