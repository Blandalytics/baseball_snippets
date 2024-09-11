import streamlit as st
import datetime
import random
from datetime import timedelta
import matplotlib as mpl
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import requests
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py
import pytz

date = datetime.date(2024,9,10)
level=1

r = requests.get(f'https://statsapi.mlb.com/api/v1/schedule?sportId={level}&date={date}')
x = r.json()
if len(x['dates'])==0:
    st.write('No games today')
    st.stop()

game_list = []
for game in range(len(x['dates'][0]['games'])):
    game_list += [x['dates'][0]['games'][game]['gamePk']]

game_choice = random.choice(game_list)
r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_choice}')
x = r.json()

home_name = x['scoreboard']['teams']['home']['teamName']
away_name = x['scoreboard']['teams']['away']['teamName']

home_abbr = x['scoreboard']['teams']['home']['abbreviation']
away_abbr = x['scoreboard']['teams']['away']['abbreviation']

home_score = x['scoreboard']['linescore']['teams']['home']['runs']
away_score = x['scoreboard']['linescore']['teams']['away']['runs']

delta_home_win_exp = []
home_win_prob = []
win_prob_game_pk = []
win_prob_abs = []
    
game_date = []
pitcher_id_list = []
pitcher_name = []
pitch_id = []
inning = []
out = []
total_pitches = []
ab_index = []
game_pk = []

for game_id in game_list:
    r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_id}')
    x = r.json()
    
    for ab in range(len(x['scoreboard']['stats']['wpa']['gameWpa'])):
        win_prob_game_pk += [game_id]
        win_prob_abs += [ab]
        delta_home_win_exp += [x['scoreboard']['stats']['wpa']['gameWpa'][ab]['homeTeamWinProbabilityAdded']]
        home_win_prob += [x['scoreboard']['stats']['wpa']['gameWpa'][ab]['homeTeamWinProbability']]

    for home_away_pitcher in ['home','away']:
        if f'{home_away_pitcher}_pitchers' not in x.keys():
            continue
        for pitcher_id in list(x[f'{home_away_pitcher}_pitchers'].keys()):
            for pitch in range(len(x[f'{home_away_pitcher}_pitchers'][pitcher_id])):
                game_pk += [game_id]
                game_date += [x['gameDate']]
                pitcher_id_list += [pitcher_id]
                pitcher_name += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitcher_name']]
                inning += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['inning']]
                out += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['outs']]
                ab_index += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['ab_number']-1] 
                total_pitches += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['game_total_pitches']]

player_df = pd.DataFrame()
player_df['game_pk'] = game_pk
player_df['game_pk'] = player_df['game_pk'].astype('int')
player_df['game_date'] = game_date
player_df['year_played'] = 2024
player_df['MLBAMID'] = pitcher_id_list
player_df['Pitcher'] = pitcher_name
player_df['inning'] = inning
player_df['total_pitches'] = total_pitches
player_df['post_outs'] = out
player_df['ab_index'] = ab_index
player_df['ab_index'] = player_df['ab_index'].astype('int')

wpa_df = pd.DataFrame()
wpa_df['game_pk'] = win_prob_game_pk
wpa_df['game_pk'] = wpa_df['game_pk'].astype('int')
wpa_df['ab_index'] = win_prob_abs
wpa_df['ab_index'] = wpa_df['ab_index'].astype('int')
wpa_df['delta_home_win_exp'] = delta_home_win_exp
wpa_df['home_win_prob'] = home_win_prob

scraped_game_df = (
    player_df.astype({'game_pk':'int','ab_index':'int'})
    .merge(wpa_df.astype({'game_pk':'int','ab_index':'int'}),
           how='inner',
           on=['game_pk','ab_index'])
    .sort_values('total_pitches')
    .groupby(['game_pk','ab_index'])
    [['inning','post_outs','delta_home_win_exp','home_win_prob']]
    .last()
    .reset_index()
)

single_game_df = scraped_game_df.loc[scraped_game_df['game_pk']==game_choice].copy()
single_game_df['outs_made'] = np.where(single_game_df['post_outs'].shift(1)<3,
                                        single_game_df['post_outs'].sub(single_game_df['post_outs'].shift(1).fillna(0)),
                                        single_game_df['post_outs'])
single_game_df['game_outs'] = single_game_df['outs_made'].astype('int').expanding().sum()

single_game_df = single_game_df.groupby('game_outs').last().reset_index()
game_outs = single_game_df['game_outs'].max()

single_game_df.loc[100] = [-1,game_choice,-1,0,0,0,50,0]
single_game_df = single_game_df.sort_values('game_outs').reset_index(drop=True)
single_game_df['home_win_prob'] = np.clip(single_game_df['home_win_prob'].div(100),0,1)

# Excitement
# https://lukebenz.com/post/gei/
# 20th percentile of 2021-2023 is ~1.2, 80th is ~4
excite_index = (54/game_outs * single_game_df['delta_home_win_exp'].div(100).abs().sum() - 1) / (4-1)
excite_index = np.clip(excite_index*10,0,10)

x = single_game_df['game_outs'].values
y = single_game_df['home_win_prob'].values

xvals = np.linspace(-1, game_outs, int(game_outs+1) * 20)
yinterp = np.interp(xvals, x, y)

# Create a figure and plot the line on it
fig, ax = plt.subplots()
ax.axhline(1,color='k',alpha=0.25)
ax.axhline(0,color='k',alpha=0.25)
custom_map = colors.ListedColormap(sns.light_palette(color_dict[away_abbr], n_colors=50, reverse=True) + 
                                   sns.light_palette(color_dict[home_abbr], n_colors=50))

# fig.colorbar(lines)
ax.axhline(0.5,color='k',alpha=0.5)

for inning in range(int(single_game_df['game_outs'].max()/6)+1):
    if single_game_df['game_outs'].max()<(inning+1)*6-4:
        continue
    ax.text((inning+0.5)*6,0.5,inning+1,ha='center',va='center',
            bbox=dict(boxstyle='round', facecolor=chart_white, alpha=0.75,edgecolor='k'))
    ax.axvline((inning+1)*6,linestyle='--',alpha=0.25,ymin=(0.25+0.1)/1.4,ymax=(0.75+0.1)/1.4,color='k')

dydx = 0.5 * (yinterp[:-1] + yinterp[1:])
sns.lineplot(x=np.array(xvals), 
             y=yinterp, color='#aaaaaa', linewidth=6.5,
                    )
lines = colored_line_between_pts(np.array(xvals), 
                                 yinterp, 
                                 dydx,
                                 ax, linewidth=5,
                     cmap=custom_map,
                     norm=colors.CenteredNorm(0.5,0.45),
                    )

ax.set(xlim=(-1,game_outs+0.5),
       ylim=(1.1,-.3))
ax.axis('off')

excite_ax = fig.add_axes([0.8,0.82,0.1,0.1], anchor='NE', zorder=1)
excite_ax.text(0,0.9,'Excitement\nIndex',ha='center',va='center')
excite_ax.text(0,0,f'{excite_index:.1f}',ha='center',va='center',
               color='k' if abs(excite_index-5)<2 else chart_white,
               bbox=dict(boxstyle='circle', pad=0.5,
                         fc=sns.color_palette('vlag',n_colors=1000)[int(excite_index*100)], 
                         ec="k"))
excite_ax.axis('off')

home_team_ax = fig.add_axes([0.025,0.115,0.1,0.1], anchor='NW', zorder=1)
cairosvg.svg2png(url=logo_dict[home_abbr], 
                 write_to="home.png")
image = Image.open('home.png')
home_team_ax.imshow(image)
home_team_ax.axis('off')

away_team_ax = fig.add_axes([0.025,0.665,0.1,0.1], anchor='NW', zorder=1)
cairosvg.svg2png(url=logo_dict[away_abbr],
                 write_to="away.png")
image = Image.open('away.png')
away_team_ax.imshow(image)
away_team_ax.axis('off')

fig.suptitle(f'Win Probability - {date:%#m/%#d/%y}\n{away_name} {away_score:.0f} @ {home_name} {home_score:.0f}',
            fontsize=20,x=0.375,y=0.93)
sns.despine()
