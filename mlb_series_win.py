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
max_series_len = 15

def log_pythag_win(higher_rs,higher_ra,
                  lower_rs,lower_ra):
    higher_factor = (higher_rs+higher_ra)**0.285
    higher_win_prob = higher_rs**higher_factor/(higher_rs**higher_factor+higher_ra**higher_factor)
    
    lower_factor = (lower_rs+lower_ra)**0.285
    lower_win_prob = lower_rs**lower_factor/(lower_rs**lower_factor+lower_ra**lower_factor)

    return (higher_win_prob-(higher_win_prob*lower_win_prob))/(higher_win_prob+lower_win_prob-(2 * higher_win_prob*lower_win_prob))
                    
def best_of_prob(games, higher_seed_win_prob, 
                 sims=sims, hfa=0.04,all_home=False):
    if all_home:
        series_schedule = [1] * games
    else:
        series_schedule = [1,0,1] if games==3 else [1,1,0,0] + [1,0]*int(round((games-5)/2)) + [1]
    series_wins = []
    series_games = []
    for series in range(sims):
        higher_seed_wins = 0
        lower_seed_wins = 0
        series_win = 0
        games_played = 0
        for game in series_schedule:
            win_prob = higher_seed_win_prob+hfa*(game-0.5)*2
            games_played += 1
            if np.random.random() <=win_prob:
                higher_seed_wins += 1
            else:
                lower_seed_wins += 1
            if higher_seed_wins == int(games/2+0.5):
                series_win = 1
                break
            if lower_seed_wins == int(games/2+0.5):
                break
        series_games += [games_played]
        series_wins += [series_win]
    return series_wins, series_games

team_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1zbYqHg685OyP_D-E_QO2_oENfEAKx0tpnkWhjlRBAMA/export?gid=0&format=csv').rename(columns={
  'RS/G':'Runs Scored',
  'RA/G':'Runs Allowed'
})

col1, col2 = st.columns(2)
with col1:
    higher_seed_team = st.selectbox('Higher seed:',list(team_df['Team']),index=1)
with col2:
    lower_seed_team = st.selectbox('Lower seed:',list(team_df['Team']),index=5)

# higher_seed_team, lower_seed_team = (team_1, team_2) if team_df[team_df['Team']==team_1]['Win%'].values[0] >= team_df[team_df['Team']==team_2]['Win%'].values[0] else (team_2, team_1)

higher_seed_rs,higher_seed_ra = team_df[team_df['Team']==higher_seed_team][['Runs Scored','Runs Allowed']].values[0]
higher_seed_color = team_df[team_df['Team']==higher_seed_team]['Color'].values[0]
higher_seed_code = team_df[team_df['Team']==higher_seed_team]['Code'].values[0]
lower_seed_rs,lower_seed_ra = team_df[team_df['Team']==lower_seed_team][['Runs Scored','Runs Allowed']].values[0]
lower_seed_color = team_df[team_df['Team']==lower_seed_team]['Color'].values[0]
hfa = 0.04

if higher_seed_team==lower_seed_team:
    est_win_prob = 0.5
else:
    est_win_prob = log_pythag_win(higher_seed_rs,higher_seed_ra,
                                  lower_seed_rs,lower_seed_ra)
st.write(f'The {higher_seed_team} are expected to beat the {lower_seed_team} ~{est_win_prob:.1%} of the time at a neutral site, based on their regular season runs scored and allowed. Home Field Advantage is assumed to be worth ~{hfa:.0%}.')

@st.cache_data(ttl=10*60,show_spinner=f"Simulating {sims:,} matchups, for each series length")
def series_sims(est_win_prob,sims,hfa,series_max=max_series_len):
    fill_dict = {1:est_win_prob+hfa}
    fill_dict.update({x*2+1:sum(best_of_prob(x*2+1,est_win_prob,sims,hfa=hfa)[0])/sims for x in range(1,int(series_max/2+0.5))})
    return fill_dict

def series_chart(fill_dict):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.axhline(0.5,linestyle='--',alpha=0.25,color='k')
    sns.lineplot(fill_dict,color=lower_seed_color,linewidth=3)
    for series_len in fill_dict.keys():
        ax.text(series_len,fill_dict[series_len],
                f'{fill_dict[series_len]:.1%}' if round(fill_dict[series_len],3)<1 else '~100%',
                fontsize=10,
                color='w',
                ha='center',va='center',
               bbox=dict(boxstyle="round",pad=0.25,alpha=1,
                         edgecolor=lower_seed_color,linewidth=1,
                         color=higher_seed_color))
    ax.set_xticks(list(fill_dict.keys()))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1,decimals=0))
    ax.set(xlim=(0,max(fill_dict.keys())+1),
           ylim=(min(0.3,round(min(fill_dict.values())-0.05,1)),
                 max(0.7,round(max(fill_dict.values())+0.05,1))),
           xlabel='Series Length ("Best of X")')
    fig.suptitle(f'{higher_seed_team} over {lower_seed_team} Series Win%\n(Single Game, Neutral Site xWin%: {est_win_prob:.1%})',
                y=1.02)
    pl_ax = fig.add_axes([0.03,-0.015,0.2,0.1], anchor='NE', zorder=1)
    width, height = logo.size
    pl_ax.imshow(logo.crop((0, 0, width, height-150)))
    pl_ax.axis('off')
    sns.despine(bottom=True)
    st.pyplot(fig)

# if st.button("Simulate series"):
fill_dict = series_sims(est_win_prob,sims,hfa)
series_chart(fill_dict)

series_len = st.slider(
      "How many games can be in the series?",
      min_value=3,
      max_value=max_series_len,
      value=7,
      step=2
  )
all_home = st.checkbox("All higher seed home games")

def games_played_chart(series_len):
    series_sims = 250000
    games = best_of_prob(series_len,est_win_prob,
                         series_sims,hfa=hfa,all_home=all_home)
    font_size = np.clip(120/series_len,6,12)
    fig, ax  = plt.subplots(figsize=(6,4))
    game_space = list(set(games[1]))
    sns.histplot(x=games[1], 
                 hue=games[0],
                 palette=[lower_seed_color,higher_seed_color],
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
                     color='w' if height_check else lower_seed_color,
                    fontsize=font_size)
    ax.legend(ncol=2,bbox_to_anchor=(0.49,1),loc='lower center',
              labels=[higher_seed_team+f' Win: {sum(games[0])/series_sims:.1%}',lower_seed_team+f' Win: {1-sum(games[0])/series_sims:.1%}'],edgecolor='w',framealpha=0)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100,0))
    ax.set_xticks(game_space)
    ax.set(xlabel='Win in X Games',ylabel='',ylim=(0,ax.get_ylim()[1]*1.02))
    pl_ax = fig.add_axes([0.04,-0.015,0.2,0.1], anchor='NE', zorder=1)
    width, height = logo.size
    pl_ax.imshow(logo.crop((0, 0, width, height-150)))
    pl_ax.axis('off')
    home_text = f', all @{higher_seed_code}' if all_home else ''
    fig.suptitle(f'{higher_seed_team}/{lower_seed_team} Series\nGames Played Distribution (Best of {series_len}{home_text})',y=1.06)
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
