import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

sims = 100000

def log_pythag_win(favorite_rs,favorite_ra,
                  underdog_rs,underdog_ra):
    favorite_factor = (favorite_rs+favorite_ra)**0.287
    favorite_win_prob = favorite_rs**favorite_factor/(favorite_rs**favorite_factor+favorite_ra**favorite_factor)
    
    underdog_factor = (underdog_rs+underdog_ra)**0.287
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
            if game==0:
                win_prob = favorite_win_prob-hfa
            else:
                win_prob = favorite_win_prob+hfa
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
    team_1 = st.selectbox('Choose a team:',list(team_df['Team']))
with col2:
    team_2 = st.selectbox('Choose a team:',list(team_df['Team']),index=29)

favored_team, underdog = (team_1, team_2) if team_df[team_df['Team']==team_1]['Win%'].values[0] >= team_df[team_df['Team']==team_2]['Win%'].values[0] else (team_2, team_1)

favorite_rs,favorite_ra = team_df[team_df['Team']==favored_team][['Runs Scored','Runs Allowed']].values[0]
underdog_rs,underdog_ra = team_df[team_df['Team']==underdog][['Runs Scored','Runs Allowed']].values[0]
hfa = 0.04

if team_1==team_2:
    est_win_prob = 0.5
else:
    est_win_prob = log_pythag_win(favorite_rs,favorite_ra,
                                  underdog_rs,underdog_ra)

st.write(f'The {favored_team} are expected to beat the {underdog} ~{est_win_prob:.1%} of the time at a neutral site. Home Field Advantage is assumed to be worth ~{hfa:.0%}.')
fill_dict = {1:est_win_prob+hfa}
fill_dict.update({x*2+1:sum(best_of_prob(x*2+1,est_win_prob,sims,hfa=hfa)[0])/sims for x in range(1,6)})

def series_chart(fill_dict):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.lineplot(fill_dict,color=team_df[team_df['Team']==underdog]['Color'].values[0])
    for series_len in fill_dict.keys():
        ax.text(series_len,fill_dict[series_len],
                f'{fill_dict[series_len]:.1%}',
                fontsize=12,
                color='w',
                ha='center',va='center',
               bbox=dict(boxstyle="round",pad=0.25,alpha=1,edgecolor=team_df[team_df['Team']==underdog]['Color'].values[0],
                         color=team_df[team_df['Team']==favored_team]['Color'].values[0]))
    ax.set_xticks(list(fill_dict.keys()))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set(xlim=(0,max(fill_dict.keys())+1),
           ylim=(0.5,1),
           xlabel='Series Length ("Best of X")')
    fig.suptitle(f'{favored_team} over {underdog} Series Win%\n(Single Game, Neutral Site xWin%: {est_win_prob:.1%})',
                y=1)
    sns.despine(bottom=True)
    st.pyplot(fig)

series_chart(fill_dict)

series_wins, series_games = best_of_prob(games=7,
                                         favorite_win_prob=est_win_prob,
                                         sims=sims,
                                         hfa=hfa)
series_win_prob = sum(series_wins) / sims
