import streamlit as st
import datetime
import random
from datetime import timedelta
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import requests
import pytz
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.collections import LineCollection
import urllib
import cairosvg
from PIL import Image

from concurrent.futures import ThreadPoolExecutor, as_completed

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, ColumnsAutoSizeMode

st.title('Wheeee! Index')
st.write('This is an attempt to quantify how much "Wheeee!" a baseball game has (s/o to Sarah Langs for [making this a thing!](https://x.com/search?q=from%3ASlangsOnSports%20wheeee&src=typed_query&f=live))')

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

color_df = pl.read_csv('https://github.com/Blandalytics/PLV_viz/blob/main/mlb_team_colors.csv?raw=true')
color_dict = color_df[['Short Code','Color 1']].rows_by_key(key=["Short Code"],unique=True)
logo_dict = color_df[['Short Code','Logo']].rows_by_key(key=["Short Code"],unique=True)

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
date = st.date_input("Select a game date:", today, min_value=datetime.date(2020, 3, 28), max_value=today)

def fetch_game_ids(date,regular_season=False):
    r = requests.get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}')
    x = r.json()
    if x['totalGames']==0:
        date_list = []
    elif regular_season:
        date_list = pl.DataFrame(x['dates'][0]['games']).filter(pl.col('gameType')=='R')['gamePk'].to_list()
    else: 
        date_list = pl.DataFrame(x['dates'][0]['games'])['gamePk'].to_list()
    return date_list
    
def fetch_pitches(game_pk):
    df_list = []
    r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_pk}')
    x = r.json()

    for team in ['home','away']:
        if f'{team}_pitchers' not in x.keys():
            continue
        if 'scoreboard' not in x.keys():
            continue
        home_code = x['scoreboard']['teams']['home']['teamName']
        home_abbrev = x['home_team_data']['abbreviation']
        home_score = x['boxscore']['teams']['home']['teamStats']['batting']['runs']
        away_code = x['scoreboard']['teams']['away']['teamName']
        away_abbrev = x['away_team_data']['abbreviation']
        away_score = x['boxscore']['teams']['away']['teamStats']['batting']['runs']
        for pitcher_id in list(x[f'{team}_pitchers'].keys()):
            df_list.append(
                pl.DataFrame(x[f'{team}_pitchers'][pitcher_id],strict=False)
                .with_columns(pl.lit(home_code).alias("home_team"),
                              pl.lit(away_code).alias("away_team"),
                              pl.lit(home_abbrev).alias("home_abbrev"),
                              pl.lit(away_abbrev).alias("away_abbrev"),
                              pl.lit(home_score).alias("home_score"),
                              pl.lit(away_score).alias("away_score"),
                              pl.lit(date.year).alias("year_played"))
            )
    if not df_list:
        pitches = pl.DataFrame()
    else:
        pitches = pl.concat(df_list, how="diagonal_relaxed").with_columns(pl.col("game_pk").cast(pl.Int32),
                                                                          pl.col("ab_number").cast(pl.Int32))
    return pitches

def fetch_win_prob(game_pk):
    wp_list = []
    r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_pk}')
    x = r.json()

    wp_list.append(
                pl.DataFrame(x['scoreboard']['stats']['wpa']['gameWpa'])
                .with_row_index("ab_number", offset=1)
                .with_columns(pl.lit(game_pk).alias("game_pk"))
            )
    if not wp_list:
        win_probs = pl.DataFrame()
    else:
        win_probs = pl.concat(wp_list, how="vertical_relaxed").with_columns(pl.col("game_pk").cast(pl.Int32),
                                                                            pl.col("ab_number").cast(pl.Int32))
    return win_probs
    
@st.cache_data(ttl=90,show_spinner=f"Loading data")
def threaded_data(game_list_input):
    pitch_data = []
    win_prob_data = []
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_pitches, game_pk): game_pk for game_pk in game_list_input}
        for future in as_completed(futures):
            pitch_data.append(future.result())
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fetch_win_prob, game_pk): game_pk for game_pk in game_list_input}
        for future in as_completed(futures):
            win_prob_data.append(future.result())

    pitch_df = pl.concat(pitch_data, how="diagonal_relaxed")
    wpa_df = pl.concat(win_prob_data, how="diagonal_relaxed")
    combined_df = (
        pitch_df
        .join(wpa_df,
               how='inner',
               on=['game_pk','ab_number'])
        .sort('game_total_pitches')
        .group_by(['game_pk','ab_number'])
        .agg(pl.last(['home_team','home_abbrev','home_score','away_team','away_abbrev','away_score','inning','outs','homeTeamWinProbabilityAdded','homeTeamWinProbability']))
        .sort(['game_pk','ab_number'])
        .with_columns(
            pl.concat_str(
                [
                    pl.col("away_team"),
                    pl.lit("@"),
                    pl.col("home_team"),
                    pl.lit("-"),
                    pl.col("game_pk"),
                ],
                separator=" ",
            ).alias("game_name"),
            (pl
             .when(pl.col('outs').shift(1)<3)
             .then(pl.col('outs') - pl.col('outs').shift(1,fill_value=0))
             .otherwise(pl.col('outs'))
             .alias('outs_made')
             )
        )
        .with_columns(
            pl.col("outs_made").cum_sum().over('game_name').alias("game_outs")
        )
    )
    
    return combined_df

win_prob_df = threaded_data(fetch_game_ids(date))

test_df = (
    win_prob_df
    .group_by(['game_name','home_team','away_team','home_abbrev','away_abbrev','game_outs'])
    .agg(
        pl.col('homeTeamWinProbabilityAdded').sum(),
        pl.col('homeTeamWinProbability').last().alias("home_win_prob"),
        pl.col('home_score').last(),
        pl.col('away_score').last()
    )
    .sort(['game_name','game_outs'])
    .rolling(index_column="game_outs", period="6i",group_by=["game_name",'home_team','away_team','home_abbrev','away_abbrev']).agg(
        pl.max("home_win_prob").alias("rolling_max_prob"),
        pl.min("home_win_prob").alias("rolling_min_prob"),
        pl.last("home_win_prob").alias("rolling_last_prob"),
        pl.last("home_score"),
        pl.last("away_score")
    )
    .with_columns(
        pl.col('game_name').str.tail(6).alias('game_pk'),
        (pl.col("rolling_max_prob") - pl.col("rolling_min_prob")).alias('win_swing'),
        (pl
         .when(pl.col('game_name')==pl.col('game_name').shift(1))
         .then(pl.col("rolling_last_prob") - pl.col("rolling_last_prob").shift(1))
         .otherwise(pl.col("rolling_last_prob")-50)
         .alias('homeTeamWinProbabilityAdded')
         )
    )
)
    
agg_df = (
    test_df
    .group_by('game_name')
    .agg(
        pl.col('game_outs').max(),
        pl.col('homeTeamWinProbabilityAdded').abs().sum(),
        pl.col('win_swing').max()
    )
    .with_columns(((np.log(54/pl.col('game_outs') * pl.col('homeTeamWinProbabilityAdded'))-4.6)/(5.86-4.6)).alias('win_prob_index'), # 100 and 350%
                  ((np.log(pl.col('win_swing'))-3)/(4.32-3)).alias('win_swing_index') # 20 and 75%
                 )
    .with_columns(pl.mean_horizontal('win_prob_index','win_swing_index').mul(10).alias('raw_excitement_index'))
    .with_columns(pl.col('raw_excitement_index').round(1).clip(0,10).alias('excitement_index'))
    
    .with_columns(pl.col('game_name').str.tail(6).alias('game_pk'),
                  pl.concat_str(pl.col('game_name').str.head(pl.col('game_name').str.len_bytes()-9),pl.lit(': '),pl.col('excitement_index')).alias('game_name'),
                 )
    .sort('raw_excitement_index',descending=True)
)


if agg_df.shape[0]==0:
    st.write('No games played')
    st.stop()

game_list = agg_df['game_name'].to_list()
game_choice = st.selectbox('Choose a game:',game_list)
game_choice_id = agg_df.row(by_predicate=(pl.col("game_name") == game_choice))[-1]

def game_chart(game_choice_id):
    single_game_df = test_df.filter(pl.col('game_pk')==game_choice_id)
    home_name = single_game_df['home_team'][0]
    away_name = single_game_df['away_team'][0]
    home_abbr = single_game_df['home_abbrev'][0]
    away_abbr = single_game_df['away_abbrev'][0]
    home_score = single_game_df['home_score'][0]
    away_score = single_game_df['away_score'][0]
    
    # Add start row for 50%
    append_row = single_game_df[0]
    append_row = append_row.with_columns(pl.lit(-1).alias('game_outs'),
                                         pl.lit(50).alias('rolling_last_prob'),
                                         pl.lit(0).alias('homeTeamWinProbabilityAdded'))
    single_game_df = pl.concat([append_row,single_game_df], how="vertical_relaxed")
    x = single_game_df.select(pl.col('game_outs')).to_numpy().ravel()
    y = single_game_df.select(pl.col('rolling_last_prob')).to_numpy().ravel() / 100
    
    gei = agg_df.filter(pl.col('game_pk')==game_choice_id)['homeTeamWinProbabilityAdded'][0] / 100
    win_prob_index = agg_df.filter(pl.col('game_pk')==game_choice_id)['win_prob_index'][0]
    biggest_win_swing = agg_df.filter(pl.col('game_pk')==game_choice_id)['win_swing'][0] / 100
    win_swing_index = agg_df.filter(pl.col('game_pk')==game_choice_id)['win_swing_index'][0]
    excite_index = agg_df.filter(pl.col('game_pk')==game_choice_id)['excitement_index'][0]
    
    game_outs = max(x)
    chart_outs = 54 if game_outs <51 else game_outs
    
    # Create a figure and plot the line on it
    fig, ax = plt.subplots(figsize=(7,5))
    ax.axhline(1,color=color_dict[home_abbr][0],alpha=1,xmin=7/(chart_outs+1.25),xmax=(chart_outs+1)/(chart_outs+1.25))
    ax.axhline(0,color=color_dict[away_abbr][0],alpha=1,xmin=7/(chart_outs+1.25),xmax=(chart_outs+1)/(chart_outs+1.25))
    for inning in range(int((chart_outs-1)/6)+1):
        ax.text((inning+0.5)*6,0.5,inning+1,ha='center',va='center',
                bbox=dict(boxstyle='round', facecolor=chart_white, alpha=0.5,edgecolor='k'))
        ax.axvline((inning+1)*6,linestyle='--',alpha=0.25,ymin=(0.25+0.1)/1.5,ymax=(0.75+0.1)/1.5,color='k')
    
    custom_map = colors.ListedColormap(sns.light_palette(color_dict[away_abbr][0], n_colors=50, reverse=True) + 
                                       sns.light_palette(color_dict[home_abbr][0], n_colors=50))
    contrast_map = colors.ListedColormap(sns.dark_palette(color_dict[away_abbr][0], n_colors=50, reverse=True) + 
                                       sns.dark_palette(color_dict[home_abbr][0], n_colors=50))
    ax.axhline(0.5,color='k',alpha=0.5)
    
    nc = 50
    xvals = np.linspace(-1, game_outs, int(game_outs+1) * 5)
    y1 = np.interp(xvals, x, y)
    y_base = np.full(len(xvals), 0.5)
    normalize = colors.Normalize(vmin=0, vmax=1)
    
    for ii in range(len(xvals)-1):
        y_n = np.linspace(y1[ii], y_base[ii], nc)
        y_n1 = np.linspace(y1[ii+1], y_base[ii+1], nc)
        for kk in range(nc - 1):
            p = patches.Polygon([[xvals[ii], y_n[kk]], 
                                 [xvals[ii+1], y_n1[kk]], 
                                 [xvals[ii+1], y_n1[kk+1]], 
                                 [xvals[ii], y_n[kk+1]]], color=custom_map(normalize((y_n[kk]+y_n1[kk])/2)))
            ax.add_patch(p)
    
    plt.plot(xvals, y1, alpha=0)
    xvals_line = np.linspace(-1, game_outs, int(game_outs+1) * 50)
    yinterp_line = np.interp(xvals_line, x, y)
    dydx = 0.5 * (yinterp_line[:-1] + yinterp_line[1:])
    shadow = colored_line_between_pts(np.array(xvals_line),
                                      yinterp_line, 
                                      dydx,
                                      ax, 
                                      linewidth=2.25,
                                      cmap=contrast_map,
                                      norm=normalize, 
                                      alpha=1/3
                            )
    
    ax.set(xlim=(-1,chart_outs+0.25),
           ylim=(1.1,-.4))
    ax.axis('off')
    
    excite_ax = fig.add_axes([0.82,0.81,0.1,0.1], anchor='NE', zorder=1)
    excite_ax.text(0,0.9,'Wheeee!\nIndex',ha='center',va='center',fontsize=14)
    if excite_index==10:
        excite_ax.text(0,-0.15,f'{excite_index:.0f}',ha='center',va='center',size=16,
                       color='k' if abs(excite_index-5)<2 else chart_white,
                       bbox=dict(boxstyle='circle', pad=0.5,
                                 fc=sns.color_palette('vlag',n_colors=1001)[int(excite_index*100)], 
                                 ec="k"))
    else:
        excite_ax.text(0,-0.15,f'{excite_index:.1f}',ha='center',va='center',size=14,
                       color='k' if abs(excite_index-5)<2 else chart_white,
                       bbox=dict(boxstyle='circle', pad=0.5,
                                 fc=sns.color_palette('vlag',n_colors=1001)[int(excite_index*100)], 
                                 ec="k"))
    excite_ax.axis('off')
    
    home_team_ax = fig.add_axes([0.12,0.115,0.1,0.12], anchor='NW', zorder=1)
    cairosvg.svg2png(url=logo_dict[home_abbr][0], 
                     write_to="home.png")
    image = Image.open('home.png')
    home_team_ax.imshow(image,aspect='equal')
    home_team_ax.axis('off')
    
    away_team_ax = fig.add_axes([0.12,0.625,0.1,0.12], anchor='NW', zorder=1)
    cairosvg.svg2png(url=logo_dict[away_abbr][0],
                     write_to="away.png")
    image = Image.open('away.png')
    away_team_ax.imshow(image,aspect='equal')
    away_team_ax.axis('off')
    
    
    fig.suptitle(f'Win Probability - {date:%-m/%-d/%y}\n{away_name} {away_score:.0f} @ {home_name} {home_score:.0f}',
                fontsize=20,x=0.45,y=0.95)
    fig.text(0.32,0.785,'Δ Win Prob/54 Outs',
             ha='center', fontsize=12)
    fig.text(0.32,0.735,f'{gei:.1f} Wins',
             ha='center', fontsize=12,
             color='k' if abs(win_prob_index-.5)<.2 else chart_white,
             bbox=dict(boxstyle='round', pad=0.25,
                       fc=sns.color_palette('vlag',n_colors=1001)[int(np.clip(win_prob_index*1000,0,1000))], 
                       ec="k"))
    
    fig.text(0.62,0.785,'Biggest Swing',
             ha='center', fontsize=12)
    fig.text(0.62,0.735,f'{biggest_win_swing:.0%}',
             ha='center', fontsize=12,
             color='k' if abs(win_swing_index-.5)<.2 else chart_white,
             bbox=dict(boxstyle='round', pad=0.25,
                       fc=sns.color_palette('vlag',n_colors=1001)[int(np.clip(win_swing_index*1000,0,1000))], 
                       ec="k"))
    
    fig.text(0.41,0.12,'mlb-win-prob.streamlit.app',
             ha='center', fontsize=12)
    
    logo_loc = 'https://github.com/Blandalytics/baseball_snippets/blob/main/PitcherList_Full_Black.png?raw=true'
    logo = Image.open(urllib.request.urlopen(logo_loc))
    
    # Add PL logo
    pl_ax = fig.add_axes([0.675,0.085,0.2,0.1], anchor='NE', zorder=1)
    width, height = logo.size
    pl_ax.imshow(logo.crop((0, 0, width, height-150)))
    pl_ax.axis('off')
    
    sns.despine()
    st.pyplot(fig)
    
game_chart(game_choice_id)

st.dataframe(agg_df)

st.header('Glossary')
st.write(f'''
- **Δ Win Prob/54 Outs**: The total change in win probability for the game, normalized to 54 outs (a median game is ~1.8 wins). Derived from [Luke Benz's Game Excitement Index](https://lukebenz.com/post/gei/)
''')
st.write(f'''
- **Biggest Swing**: Largest swing in win probability across 6 outs (after the 1st inning; a median game is ~40%))
''')
st.write(f'''
- **Wheeee Index**: Combination of Δ Win Prob/54 Outs and Biggest Swing, scaled 0-10
''')
