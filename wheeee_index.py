import streamlit as st
import datetime
import random
from datetime import timedelta
import numpy as np
import pandas as pd
import seaborn as sns
import requests
import pytz
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.collections import LineCollection
import urllib
import cairosvg
from PIL import Image

chart_white = '#FEFEFE'
chart_accent = '#162B50'

sns.set_theme(
    style={
        'axes.edgecolor': chart_accent,
        'axes.facecolor': chart_white,
        'axes.labelcolor': chart_white,
        'xtick.color': chart_accent,
        'ytick.color': chart_accent,
        'figure.facecolor':chart_white,
        'grid.color': chart_white,
        'grid.linestyle': '-',
        'legend.facecolor':chart_white,
        'text.color': 'k'
     }
    )

color_df = pd.read_csv('https://github.com/Blandalytics/PLV_viz/blob/main/mlb_team_colors.csv?raw=true')
color_dict = color_df[['Short Code','Color 1']].set_index('Short Code').to_dict()['Color 1']
logo_dict = color_df[['Short Code','Logo']].set_index('Short Code').to_dict()['Logo']

def colored_line_between_pts(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified between (x, y) points by a third value.

    It does this by creating a collection of line segments between each pair of
    neighboring points. The color of each segment is determined by the
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should have a size one less than that of x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Check color array size (LineCollection still works, but values are unused)
    if len(c) != len(x) - 1:
        warnings.warn(
            "The c argument should have a length one less than the length of x and y. "
            "If it has the same length, use the colored_line function instead."
        )

    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, **lc_kwargs)

    # Set the values used for colormapping
    lc.set_array(c)

    return ax.add_collection(lc)

today = (datetime.datetime.now(pytz.utc)-timedelta(hours=16)).date()
date = st.date_input("Select a game date:", today, min_value=datetime.date(2024, 3, 28), max_value=today)

@st.cache_data(ttl=90,show_spinner=f"Loading data")
def load_win_prob(date):
    r = requests.get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}')
    x = r.json()
    if len(x['dates'])==0:
        st.write('No games today')
        st.stop()

    game_list = []
    for game in range(len(x['dates'][0]['games'])):
        game_list += [x['dates'][0]['games'][game]['gamePk']]

    delta_home_win_exp = []
    home_win_prob = []
    win_prob_game_pk = []
    win_prob_abs = []

    game_date = []
    home_team = []
    away_team = []
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
        home_code = x['scoreboard']['teams']['home']['teamName']
        away_code = x['scoreboard']['teams']['away']['teamName']

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
                    home_team += [home_code]
                    away_team += [away_code]

    pitch_df = pd.DataFrame()
    pitch_df['game_pk'] = game_pk
    pitch_df['home_team'] = home_team
    pitch_df['away_team'] = away_team
    pitch_df['game_date'] = game_date
    pitch_df['year_played'] = 2024
    pitch_df['MLBAMID'] = pitcher_id_list
    pitch_df['Pitcher'] = pitcher_name
    pitch_df['inning'] = inning
    pitch_df['total_pitches'] = total_pitches
    pitch_df['post_outs'] = out
    pitch_df['ab_index'] = ab_index

    wpa_df = pd.DataFrame()
    wpa_df['game_pk'] = win_prob_game_pk
    wpa_df['ab_index'] = win_prob_abs
    wpa_df['delta_home_win_exp'] = delta_home_win_exp
    wpa_df['home_win_prob'] = home_win_prob

    return game_list, (
        pitch_df.astype({'game_pk':'int','ab_index':'int'})
        .merge(wpa_df.astype({'game_pk':'int','ab_index':'int'}),
               how='inner',
               on=['game_pk','ab_index'])
        .sort_values('total_pitches')
        .groupby(['game_pk','ab_index'])
        [['home_team','away_team','inning','post_outs','delta_home_win_exp','home_win_prob']]
        .last()
        .reset_index()
    )
    
game_id_list, scraped_game_df = load_win_prob(date)
scraped_game_df['game_name'] = (
    scraped_game_df
    .assign(pad = ' @ ',dash = ' - ')
    [['away_team','pad','home_team','dash','game_pk']]
    .astype('str')
    .sum(axis=1)
)
game_list = scraped_game_df['game_name'].unique()
if scraped_game_df.shape[0]==0:
    st.write('No games played')
    st.stop()
    
all_games_df = scraped_game_df.copy()
all_games_df['outs_made'] = np.where(all_games_df['post_outs'].shift(1)<3,
                                        all_games_df['post_outs'].sub(all_games_df['post_outs'].shift(1).fillna(0)),
                                        all_games_df['post_outs'])
all_games_df['game_outs'] = all_games_df.groupby('game_name')['outs_made'].transform(lambda x: x.expanding().sum()).astype('int')
all_games_df = (all_games_df.groupby(['game_name','game_outs'])[['delta_home_win_exp','home_win_prob']].agg({
    'delta_home_win_exp':'sum','home_win_prob':'mean'
}).reset_index())
all_games_df['home_win_prob'] = all_games_df.groupby('game_name')['delta_home_win_exp'].transform(lambda x: x.expanding().sum()).add(0.5)
all_games_df['rolling_away_prob'] = all_games_df.groupby('game_name')['home_win_prob'].transform(lambda x: x.rolling(6, 6).min())
all_games_df['rolling_home_prob'] = all_games_df.groupby('game_name')['home_win_prob'].transform(lambda x: x.rolling(6, 6).max())
all_games_df = (
    all_games_df
    .assign(delta_home_win_exp = lambda x: x['delta_home_win_exp'].abs().div(100),
            win_swing = lambda x: x['rolling_home_prob'].sub(x['rolling_away_prob']).abs().div(100))
    .groupby('game_name')
    [['game_outs','delta_home_win_exp','win_swing']]
    .agg({
        'game_outs':'max',
        'delta_home_win_exp':'sum',
        'win_swing':'max'
    })
)
all_games_df['win_prob_index'] = (54/all_games_df['game_outs'] * np.log(all_games_df['delta_home_win_exp']) + 0.2) / (1 + 0.2)
all_games_df['win_swing_index'] = (np.log(all_games_df['win_swing']) + 1.6) / (-0.4 + 1.6)
all_games_df['excitement_index'] = np.clip(all_games_df[['win_prob_index','win_swing_index']].mean(axis=1),0,1)*10

games = st.dataframe(all_games_df
             .sort_values('excitement_index',ascending=False)
             .assign(delta_home_win_exp = lambda x: x['delta_home_win_exp'].mul(100),
                     win_swing = lambda x: x['win_swing'].mul(100))
             .rename(columns={
                 'delta_home_win_exp':'Total Win Exp Change (%)',
                 'win_swing':'Biggest Win Exp Swing (%)',
                 'excitement_index':'Wheeee! Index'
             })
             [['Total Win Exp Change (%)','Biggest Win Exp Swing (%)','Wheeee! Index']]
             .style
             .format(precision=1)
             .background_gradient(axis=None, vmin=0, vmax=10, cmap="vlag",
                                  subset=['Wheeee! Index']
                                 ), 
                     on_select="rerun",
             use_container_width=1)

game_choice = games.selection.rows
print(game_choice)
game_choice_id = int(all_games_df.sort_values('excitement_index').iloc[game_choice][-6:])

def game_chart(game_choice_id):
    r_game = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_choice_id}')
    x_game = r_game.json()

    home_name = x_game['scoreboard']['teams']['home']['teamName']
    away_name = x_game['scoreboard']['teams']['away']['teamName']

    home_abbr = x_game['scoreboard']['teams']['home']['abbreviation']
    away_abbr = x_game['scoreboard']['teams']['away']['abbreviation']

    home_score = x_game['scoreboard']['linescore']['teams']['home']['runs']
    away_score = x_game['scoreboard']['linescore']['teams']['away']['runs']

    single_game_df = scraped_game_df.loc[scraped_game_df['game_pk']==game_choice_id].copy()
    single_game_df['outs_made'] = np.where(single_game_df['post_outs'].shift(1)<3,
                                            single_game_df['post_outs'].sub(single_game_df['post_outs'].shift(1).fillna(0)),
                                            single_game_df['post_outs'])
    single_game_df['game_outs'] = single_game_df['outs_made'].astype('int').expanding().sum()

    single_game_df = single_game_df.groupby('game_outs').last().reset_index()
    game_outs = single_game_df['game_outs'].max()

    single_game_df.loc[100] = [-1,game_choice_id,'','',-1,0,0,0,50,0,'']
    single_game_df = single_game_df.sort_values('game_outs').reset_index(drop=True)
    single_game_df['home_win_prob'] = np.clip(single_game_df['home_win_prob'].div(100),0,1)

    ## Excitement Index
    # GEI (2021-2023 5th percentile is ~0.2, 95th is ~1, log scale)
    # https://lukebenz.com/post/gei/
    gei = 54/game_outs * single_game_df['delta_home_win_exp'].div(100).abs().sum()
    win_prob_index = (np.log(gei) + 0.2) / (1 + 0.2)

    # Biggest rolling 1 Inning Swing in Win Prob (log)
    # 2021-2023 5th percentile is -1.6, 95th is -0.4
    single_game_df['rolling_away_prob'] = single_game_df['home_win_prob'].rolling(6, 6).min()
    single_game_df['rolling_home_prob'] = single_game_df['home_win_prob'].rolling(6, 6).max()
    single_game_df['win_prob_swing'] = single_game_df['rolling_home_prob'].sub(single_game_df['rolling_away_prob']).abs()
    win_swing_index = np.log(single_game_df['win_prob_swing'].max()) + 1.6 / (-0.4 + 1.6)

    excite_index = np.clip((win_prob_index+win_swing_index)/2,0,1) * 10

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

    excite_ax = fig.add_axes([0.8,0.81,0.1,0.1], anchor='NE', zorder=1)
    excite_ax.text(0,0.9,'Wheeee!\nIndex',ha='center',va='center')
    if excite_index==10:
        excite_ax.text(0,0,f'{excite_index:.0f}',ha='center',va='center',size=14,
                       color='k' if abs(excite_index-5)<2 else chart_white,
                       bbox=dict(boxstyle='circle', pad=0.5,
                                 fc=sns.color_palette('vlag',n_colors=1001)[int(excite_index*100)], 
                                 ec="k"))
    else:
        excite_ax.text(0,0,f'{excite_index:.1f}',ha='center',va='center',
                       color='k' if abs(excite_index-5)<2 else chart_white,
                       bbox=dict(boxstyle='circle', pad=0.5,
                                 fc=sns.color_palette('vlag',n_colors=1001)[int(excite_index*100)], 
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
    st.pyplot(fig)
    
game_chart(game_choice_id)