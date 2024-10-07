import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

import urllib
from PIL import Image

st.set_page_config(page_title='MLB Series Simulator', page_icon='📊')

logo_loc = 'https://github.com/Blandalytics/baseball_snippets/blob/main/PitcherList_Full_Black.png?raw=true'
logo = Image.open(urllib.request.urlopen(logo_loc))
# st.image(logo, use_column_width=True)

st.title('MLB Series Simulator')
sims = 100000

def log_pythag_win(favorite_rs,favorite_ra,
                  underdog_rs,underdog_ra):
    favorite_factor = (favorite_rs+favorite_ra)**0.285
    favorite_win_prob = favorite_rs**favorite_factor/(favorite_rs**favorite_factor+favorite_ra**favorite_factor)
    
    underdog_factor = (underdog_rs+underdog_ra)**0.285
    underdog_win_prob = underdog_rs**underdog_factor/(underdog_rs**underdog_factor+underdog_ra**underdog_factor)

    return (favorite_win_prob-(favorite_win_prob*underdog_win_prob))/(favorite_win_prob+underdog_win_prob-(2 * favorite_win_prob*underdog_win_prob))
def best_of_prob(games, favorite_win_prob, 
                 sims=sims, hfa=0.04):
    series_schedule = [1,0,1] if games==3 else [1,1,0,0] + [1,0]*int(round((games-5)/2)) + [1]
    series_wins = []
    series_games = []
    for series in range(sims):
        favored_wins = 0
        underdog_wins = 0
        series_win = 0
        games_played = 0
        for game in series_schedule:
            win_prob = favorite_win_prob+hfa*(game-0.5)*2
            games_played += 1
            if np.random.random() <=win_prob:
                favored_wins += 1
            else:
                underdog_wins += 1
            if favored_wins == int(games/2+0.5):
                series_win = 1
                break
            if underdog_wins == int(games/2+0.5):
                break
        series_games += [games_played]
        series_wins += [series_win]
    return series_wins, series_games

team_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1zbYqHg685OyP_D-E_QO2_oENfEAKx0tpnkWhjlRBAMA/export?gid=0&format=csv')

col1, col2 = st.columns(2)
with col1:
    team_1 = st.selectbox('Choose a team:',list(team_df['Team']),index=0)
with col2:
    team_2 = st.selectbox('Choose a team:',list(team_df['Team']),index=8)

favored_team, underdog = (team_1, team_2) if team_df[team_df['Team']==team_1]['Win%'].values[0] >= team_df[team_df['Team']==team_2]['Win%'].values[0] else (team_2, team_1)

favorite_rs,favorite_ra = team_df[team_df['Team']==favored_team][['Runs Scored','Runs Allowed']].values[0]
favored_color = team_df[team_df['Team']==favored_team]['Color'].values[0]
underdog_rs,underdog_ra = team_df[team_df['Team']==underdog][['Runs Scored','Runs Allowed']].values[0]
underdog_color = team_df[team_df['Team']==underdog]['Color'].values[0]
hfa = 0.04

if team_1==team_2:
    est_win_prob = 0.5
else:
    est_win_prob = log_pythag_win(favorite_rs,favorite_ra,
                                  underdog_rs,underdog_ra)
st.write(f'The {favored_team} are expected to beat the {underdog} ~{est_win_prob:.1%} of the time at a neutral site, based on their 2024 regular season runs scored and allowed. Home Field Advantage is assumed to be worth ~{hfa:.0%}.')

@st.cache_data(ttl=10*60,show_spinner=f"Simulating {sims:,} matchups, for each series length")
def series_sims(est_win_prob,sims,hfa,series_max=11):
    fill_dict = {1:est_win_prob+hfa}
    fill_dict.update({x*2+1:sum(best_of_prob(x*2+1,est_win_prob,sims,hfa=hfa)[0])/sims for x in range(1,int(series_max/2+0.5))})
    return fill_dict

fill_dict = series_sims(est_win_prob,sims,hfa)

def series_chart(fill_dict):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.lineplot(fill_dict,color=underdog_color)
    for series_len in fill_dict.keys():
        ax.text(series_len,fill_dict[series_len],
                f'{fill_dict[series_len]:.1%}',
                fontsize=12,
                color='w',
                ha='center',va='center',
               bbox=dict(boxstyle="round",pad=0.25,alpha=1,edgecolor=underdog_color,
                         color=favored_color))
    ax.set_xticks(list(fill_dict.keys()))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set(xlim=(0,max(fill_dict.keys())+1),
           ylim=(0.5,1),
           xlabel='Series Length ("Best of X")')
    fig.suptitle(f'{favored_team} over {underdog} Series Win%\n(Single Game, Neutral Site xWin%: {est_win_prob:.1%})',
                y=1.02)
    pl_ax = fig.add_axes([0.01,0.01,0.2,0.1], anchor='NE', zorder=1)
    width, height = logo.size
    pl_ax.imshow(logo.crop((0, 0, width, height-150)))
    pl_ax.axis('off')
    sns.despine(bottom=True)
    st.pyplot(fig)

series_chart(fill_dict)

series_len = st.slider(
      "How many games can be in the series?",
      min_value=3,
      max_value=11,
      value=7,
      step=2
  )

def games_played_chart(series_len):
    games = best_of_prob(series_len,est_win_prob,
                         sims,hfa=hfa)
    font_size = np.clip(120/series_len,6,12)
    fig, ax  = plt.subplots(figsize=(6,4))
    game_space = list(set(games[1]))
    sns.histplot(x=games[1], 
                 hue=games[0],
                 palette=[underdog_color,favored_color],
                 stat='percent',multiple='stack',binrange=(min(game_space)-0.5,max(game_space)+0.5),binwidth=1,
                 alpha=1,
                 edgecolor='w')
    for p in ax.patches:
        height_check = p.get_height() + 1 > ax.get_ylim()[1]/10
        ax.annotate(f"{p.get_height():.1f}%\n" if p.get_height() >= 0.05 else '~0%\n', 
                    (p.get_x() + p.get_width() / 2, 
                     (p.get_y() + p.get_height()/2 - 1) if height_check else (p.get_y() + p.get_height() - ax.get_ylim()[1]/30)),
                    ha="center", 
                     va="center" if height_check else "bottom",
                     color='w' if height_check else 'k',
                    fontsize=font_size)
    ax.legend(ncol=2,bbox_to_anchor=(0.49,1),loc='lower center',
              labels=[favored_team+f' Win: {sum(games[0])/sims:.1%}',underdog+f' Win: {1-sum(games[0])/sims:.1%}'],edgecolor='w',framealpha=0)

    # ax2.axis('off')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100,0))
    ax.set_xticks(game_space)
    ax.set(xlabel='',ylabel='',ylim=(0,ax.get_ylim()[1]*1.02))
    fig.suptitle(f'{favored_team}/{underdog} Series\nGames Played Distribution (Best of {series_len})',y=1.06)
    sns.despine()
    st.pyplot(fig)

games_played_chart(series_len)

st.write(f'''
- [PythagenPat Run Environment exponent](https://legacy.baseballprospectus.com/glossary/index.php?mode=viewstat&stat=136)

pat_exp = [(runs_scored + runs_allowed) / game] ^ .285

''')
st.write(f'''
- [Pythagorean Team Win% Estimate](https://www.mlb.com/glossary/advanced-stats/pythagorean-winning-percentage)

team_win% = (runs_scored ^ pat_exp) / [(runs_scored ^ pat_exp) + (runs_allowed ^ pat_exp)]

''')
st.write(f'''
- [Log 5 Win% Estimate](https://web.williams.edu/Mathematics/sjmiller/public_html/103/Log5WonLoss_Paper.pdf)

combined_win% = [favored_win% - (favored_win% * underdog_win%)]/[favored_win% + underdog_win% - (2 * favored_win% * underdog_win%)]

''')
