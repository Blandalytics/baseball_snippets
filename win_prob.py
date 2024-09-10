import streamlit as st
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import urllib

import datetime
import os

import io
import cairosvg
from PIL import Image

from pybaseball import statcast

date = datetime.date(2024,9,9)
def load_statcast(date):
    df = statcast(date.strftime('%Y-%m-%d'))
    df = df.sort_values(['game_pk','at_bat_number','pitch_number'])
    df['game_outs'] = (
        df['inning']
        .sub(1)
        .mul(6)
        .add(df['inning_topbot'].map({'Top':0,'Bot':3}))
        .add(df['outs_when_up'])
    )
    return df
    
single_day_games = load_statcast(date)
game_id = single_day_games['game_pk'].sample(1).item()

def win_prob_chart(game_id=game_id,games_df=single_day_games):
    single_game_df = (
        games_df
        .loc[games_df['game_pk']==game_id]
        .groupby('game_outs')
        [['delta_home_win_exp','post_away_score','post_home_score']]
        .agg({
            'delta_home_win_exp':'sum',
            'post_away_score':'max',
            'post_home_score':'max'
        })
        .reset_index()
    )
    single_game_df['home_win_prob'] = np.clip(single_game_df['delta_home_win_exp'].expanding().sum().add(0.5),0,1)

    game_outs = single_game_df['game_outs'].max()
    home_score = single_game_df['post_home_score'].max()
    away_score = single_game_df['post_away_score'].max()

    # Excitement
    # https://lukebenz.com/post/gei/
    # 20th percentile of 2021-2023 is ~1.3, 80th is ~3.9
    excite_index = (54/game_outs * single_game_df['delta_home_win_exp'].abs().sum() - 1.3) / (3.9-1.3)
    excite_index = np.clip(excite_index*10,0,9.9)

    x = single_game_df['game_outs'].values
    y = single_game_df['home_win_prob'].values

    xvals = np.linspace(0, game_outs, single_game_df['game_outs'].max() * 20)
    yinterp = np.interp(xvals, x, y)

    # Create a figure and plot the line on it
    fig, ax = plt.subplots()
    ax.axhline(1,color='k',alpha=0.25)
    ax.axhline(0,color='k',alpha=0.25)
    home_team, away_team, game_date = games_df.loc[games_df['game_pk']==game_id,['home_team','away_team','game_date']].value_counts().index[0]
    custom_map = colors.ListedColormap(sns.light_palette(color_dict[away_team], n_colors=50, reverse=True) + 
                                       sns.light_palette(color_dict[home_team], n_colors=50))

    # fig.colorbar(lines)
    ax.axhline(0.5,color='k',alpha=0.5)

    for inning in range(int(single_game_df['game_outs'].max()/6)+1):
        if single_game_df['game_outs'].max()<(inning+1)*6-4:
            continue
        ax.text((inning+0.5)*6,0.5,inning+1,ha='center',va='center',
                bbox=dict(boxstyle='round', facecolor=pl_white, alpha=0.75,edgecolor='k'))
    #     if single_game_df['game_outs'].max()<(inning+1)*6-3:
    #         continue
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
                         norm=colors.CenteredNorm(0.5,0.5),
                        )

    ax.set(xlim=(-1,game_outs+1),
           ylim=(1.1,-.3))
    ax.axis('off')

    excite_ax = fig.add_axes([0.8,0.82,0.1,0.1], anchor='NE', zorder=1)
    excite_ax.text(0,0.9,'Excitement\nIndex',ha='center',va='center')
    excite_ax.text(0,0,f'{excite_index:.1f}',ha='center',va='center',color='k',
                   bbox=dict(boxstyle='circle', pad=0.5,
                             fc=sns.color_palette('vlag',n_colors=1000)[int(excite_index*100)], ec="k"))
    excite_ax.axis('off')

    home_team_ax = fig.add_axes([0,0.115,0.1,0.1], anchor='NW', zorder=1,
                               facecolor='w')
    cairosvg.svg2png(url=logo_dict[home_team], write_to="test.png")
    image = Image.open('test.png')
    home_team_ax.imshow(image)
    home_team_ax.axis('off')

    away_team_ax = fig.add_axes([0,0.675,0.1,0.1], anchor='NW', zorder=1,
                               facecolor='w')
    cairosvg.svg2png(url=logo_dict[away_team], write_to="test.png")
    image = Image.open('test.png')
    away_team_ax.imshow(image)
    away_team_ax.axis('off')

    fig.suptitle(f'Win Probability\n{away_team} {away_score:.0f} @ {home_team} {home_score:.0f} - {game_date:%#m/%#d/%y}',
                fontsize=20,x=0.4,y=0.93)
    sns.despine()

win_prob_chart()
