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

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from concurrent.futures import ThreadPoolExecutor, as_completed

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, ColumnsAutoSizeMode

from pyfonts import set_default_font, load_google_font
import matplotlib.font_manager as fm

@st.cache_data(ttl=3600)
def load_logo():
    logo_loc = 'https://res.cloudinary.com/dduabusaf/image/upload/v1772839288/PitcherList_Stats_watermark_with_logo_k9e3xa.webp'
    logo = Image.open(urllib.request.urlopen(logo_loc))
    return logo

logo = load_logo()

@st.cache_data(ttl=3600)
def letter_logo():
    logo_loc = 'https://res.cloudinary.com/dduabusaf/image/upload/v1772839606/teal_letter_logo_owufaj.png'
    logo = Image.open(urllib.request.urlopen(logo_loc))
    return logo

letter_logo = letter_logo()

def load_team_logo(team_abbr):
    cairosvg.svg2png(url=logo_dict[team_abbr][0],
                     write_to="team_logo.png")
    image = Image.open('team_logo.png')
    return image

base_font = 'DM Sans'
font = load_google_font(base_font, weight=700)
fm.fontManager.addfont(str(font.get_file()))

## Set Styling
# Plot Style
pl_white = '#FFFFFF'
pl_background = '#292C42'
pl_text = '#00D4FF'#'#72CBFD'
pl_line_color = '#8D96B3'
pl_highlight = '#F1C647'
pl_highlight_gradient = ['#F1C647','#F5A05E']
pl_highlight_cmap = sns.color_palette(f'blend:{pl_highlight_gradient[0]},{pl_highlight_gradient[1]}', as_cmap=True)

sns.set_theme(
    style={
        'axes.edgecolor': pl_line_color,
        'axes.facecolor': pl_white,
        'axes.labelcolor': pl_white,
        'xtick.color': pl_line_color,
        'ytick.color': pl_line_color,
        'figure.facecolor':pl_white,
        'grid.color': pl_background,
        'grid.linestyle': '-',
        'legend.facecolor':pl_background,
        'text.color': 'k'
     },
    font=base_font
    )
mpl.rcParams.update({"font.weight": 700})

st.set_page_config(page_title='MLB Excitement Stats', page_icon=letter_logo)

new_title = '<p style="color:#72CBFD; font-weight: bold; font-size: 42px; text-align:center;">MLB Excitement Index</p>'
st.markdown(new_title, unsafe_allow_html=True)
st.write('Excitement Index quantifies how much excitement a baseball game has, based on three components:')
st.write('- [Volatility](https://inpredictable.substack.com/p/quantifying-excitement): how much does the win probability change throughout the game (0-10)')
st.write('- [Tension](https://www.inpredictable.com/2020/04/an-update-to-tension-index-with-assist_11.html): how uncertain the outcome of the game is (0-10)')
st.write('- Biggest Swing: largest 6-out swing in win probability')

color_df = pl.read_csv('https://github.com/Blandalytics/PLV_viz/blob/main/mlb_team_colors.csv?raw=true')
color_dict = color_df[['Short Code','Color 1']].rows_by_key(key=["Short Code"],unique=True)
logo_dict = color_df[['Short Code','Logo']].rows_by_key(key=["Short Code"],unique=True)

col1, col2 = st.columns(2)
with col1:
    today = (datetime.datetime.now(pytz.utc)-timedelta(hours=16)).date()
    date = st.date_input("Select a game date:", today, min_value=datetime.date(2020, 3, 28), max_value=today)

def fetch_game_ids(date):
    r = requests.get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}')
    x = r.json()
    if x['totalGames']==0:
        date_list = []
    else:
        date_list = []
        for game in x['dates'][0]['games']:
            if 'rescheduleGameDate' in list(game.keys()):
                if game['rescheduleGameDate']!=date:
                    continue
            if game['status']['detailedState'] in ['Pre-Game','Warmup']:
                continue
            date_list += [game['gamePk']]
    return date_list

def fetch_pitches(game_pk):
    df_list = [pl.DataFrame({
        'game_pk':game_pk,
        'ab_number':0,
        'game_total_pitches':0,
        'outs':0
    })]
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
        pitches = pl.DataFrame(df_list)
    else:
        pitches = pl.concat(df_list, how="diagonal_relaxed").with_columns(pl.col("game_pk").cast(pl.Int32),
                                                                          pl.col("ab_number").cast(pl.Int32))
    return pitches

def fetch_win_prob(game_pk):
    wp_list = [pl.DataFrame({
        'game_pk':game_pk,
        'ab_number':0,
        'homeTeamWinProbability':50,
        'awayTeamWinProbability':50,
        'hwp':0,
        'awp':0,
        'homeTeamWinProbabilityAdded':0
    })]
    r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_pk}')
    x = r.json()

    wp_list.append(
                pl.DataFrame(x['scoreboard']['stats']['wpa']['gameWpa'])
                .with_row_index("ab_number", offset=1)
                .with_columns(pl.lit(game_pk).alias("game_pk"))
            )
    if not wp_list:
        win_probs = pl.DataFrame(wp_list)
    else:
        win_probs = pl.concat(wp_list, how="diagonal_relaxed").with_columns(pl.col("game_pk").cast(pl.Int32),
                                                                            pl.col("ab_number").cast(pl.Int32))
    return win_probs

def merge_dfs(game_pk):
    pitch_df = fetch_pitches(game_pk)
    wpa_df = fetch_win_prob(game_pk)
    if pitch_df.shape[0]==0:
        combined_df = pl.DataFrame()
    else:
        combined_df = (
            pitch_df
            .join(wpa_df,
                   how='inner',
                   on=['game_pk','ab_number'])
            .sort('game_total_pitches')
            .group_by(['game_pk','ab_number'])
            .agg(pl.last(['home_team','home_abbrev','home_score','away_team','away_abbrev','away_score','inning','outs','homeTeamWinProbabilityAdded','homeTeamWinProbability','awayTeamWinProbability','events']))
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
                 ),
                (pl
                 .when(pl.col('events').str.contains("Double Play|GIDP"))
                 .then(pl.lit(2))
                 .when(pl.col('events').str.contains("Triple Play"))
                 .then(pl.lit(3))
                 .when(pl.col('events').str.contains("Out|out|Caught|Sac|Pickoff"))
                 .then(pl.lit(1))
                 .otherwise(pl.lit(0))
                 .alias('ab_outs')
                 )
            )
            .with_columns(
                pl.col("ab_outs").cum_sum().over('game_name').alias("game_outs"),
                (-(pl.max_horizontal("homeTeamWinProbability", "awayTeamWinProbability")/100*np.log2(pl.max_horizontal("homeTeamWinProbability", "awayTeamWinProbability")/100).replace(float("-inf"), 0))-(pl.min_horizontal("homeTeamWinProbability", "awayTeamWinProbability")/100*np.log2(pl.min_horizontal("homeTeamWinProbability", "awayTeamWinProbability")/100).replace(float("-inf"), 0))).alias('tension'),
                ((pl.col('homeTeamWinProbability') / 100 * np.log2(pl.col('homeTeamWinProbability') / pl.col('homeTeamWinProbability').shift(1)).replace(float("-inf"), 0) + pl.col('awayTeamWinProbability') / 100 * np.log2(pl.col('awayTeamWinProbability') / pl.col('awayTeamWinProbability').shift(1)).replace(float("-inf"), 0))).alias('k_l_excite')
            )
        )
    return combined_df

def threaded_data(game_list_input):
    games_data = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(merge_dfs, game_pk): game_pk for game_pk in game_list_input}
        for future in as_completed(futures):
            games_data.append(future.result())
    combined_df = pl.concat(games_data, how="diagonal_relaxed")
    return combined_df.filter(pl.col("game_name").is_not_null())

def game_table(win_prob_df):
    play_df = (
        win_prob_df
        .group_by(['game_name','game_pk','home_team','away_team','home_abbrev','away_abbrev','inning','ab_number','events'])
        .agg(
            pl.col('k_l_excite').last(),
            pl.col('tension').last(),
            pl.col('homeTeamWinProbabilityAdded').sum(),
            pl.col('homeTeamWinProbability').last().alias("home_win_prob"),
            pl.col('home_score').last(),
            pl.col('away_score').last(),
            pl.col('game_outs').last()
        )
        .sort(['game_pk','ab_number'])
    )

    swing_df = (
        win_prob_df
        .group_by(['game_name','game_pk','home_team','away_team','home_abbrev','away_abbrev','game_outs'])
        .agg(
            pl.col('homeTeamWinProbability').last().alias("home_win_prob")
        )
        .sort(['game_pk','game_outs'])
        .rolling(index_column="game_outs", period="6i",group_by=["game_name",'home_team','away_team','home_abbrev','away_abbrev']).agg(
            pl.max("home_win_prob").alias("rolling_max_prob"),
            pl.min("home_win_prob").alias("rolling_min_prob")
        )
        .with_columns((pl.col("rolling_max_prob") - pl.col("rolling_min_prob")).alias('win_swing'))
    )

    agg_df = (
        play_df
        .group_by(['game_name','game_pk'])
        .agg(
            pl.col('game_outs').max(),
            pl.col('k_l_excite').abs().sum(),
            pl.col('tension').mean()
        )
        .with_columns(
            (pl.col('tension')*100).alias('tension_adj'),
            (2**pl.col('k_l_excite')).alias('k_l_excite_adj'),
        )
    )
    agg_df = (
        pl.concat([agg_df,
                   swing_df.group_by('game_name').agg(pl.col('win_swing').max().alias("win_swing"))
                   ], how="align_inner")
        .with_columns(
            ((pl.col('k_l_excite') - 0.25)/2).alias('excite_scale'),
            ((pl.col('tension') - 0.35)/0.575).alias('tension_scale'),
            ((pl.col('win_swing') - 21)/60).alias('swing_scale')
        )
        .with_columns((((pl.col('excite_scale').clip(0,2.5)+pl.col('tension_scale').clip(0,2.5)+pl.col('swing_scale').clip(0,2.5))/3)**(0.75)*10).alias('watch_scale'))
        .with_columns(pl.col('watch_scale').clip(0,10).alias('watch_score'))
    )

    return play_df, agg_df.sort('watch_scale',descending=True)

days_games = fetch_game_ids(date)
game_df, table_df = game_table(threaded_data(days_games))

if table_df.shape[0]==0:
    st.write('No games played')
    st.stop()

with col2:
    game_list = table_df['game_name'].to_list()
    game_choice = st.selectbox('Choose a game:',game_list)
    game_choice_id = table_df.row(by_predicate=(pl.col("game_name") == game_choice))[1]

def game_chart(game_choice_id):
    r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_choice_id}')
    x = r.json()
    game_date = x['gameDate']#datetime.datetime.strptime(x['gameDate'],'%Y-%m-%d').strftime('%-m/%-d/%y')
    hline_y = 0.5
    single_game_df = game_df.filter(pl.col('game_pk')==game_choice_id)
    home_name = single_game_df['home_team'][0]
    away_name = single_game_df['away_team'][0]
    home_abbr = single_game_df['home_abbrev'][0]
    away_abbr = single_game_df['away_abbrev'][0]
    home_score = single_game_df['home_score'][0]
    away_score = single_game_df['away_score'][0]
    home_color = 'k' if color_dict[home_abbr][0]=='#FFFFFF' else color_dict[home_abbr][0]
    away_color = 'k' if color_dict[away_abbr][0]=='#FFFFFF' else color_dict[away_abbr][0]

    # Add start row for 50%
    append_row = single_game_df[0]
    append_row = append_row.with_columns(pl.lit(-1).alias('ab_number'),
                                         pl.lit(50).alias('home_win_prob'),
                                         pl.lit(0).alias('homeTeamWinProbabilityAdded'))
    single_game_df = pl.concat([append_row,single_game_df], how="vertical_relaxed")
    x = single_game_df.select(pl.col('ab_number')).to_numpy().ravel()
    y = single_game_df.select(pl.col('home_win_prob')).to_numpy().ravel() / 100

    excite_index = table_df.filter(pl.col('game_pk')==game_choice_id)['excite_scale'].clip(0,1)[0]*10
    tension_index = table_df.filter(pl.col('game_pk')==game_choice_id)['tension_scale'].clip(0,1)[0]*10
    biggest_swing = table_df.filter(pl.col('game_pk')==game_choice_id)['win_swing'][0]
    win_swing_index = table_df.filter(pl.col('game_pk')==game_choice_id)['swing_scale'].clip(0,1)[0]*10
    watch_index = table_df.filter(pl.col('game_pk')==game_choice_id)['watch_score'].clip(0,10)[0]

    game_abs = max(x)
    # chart_outs = 54 if game_outs <51 else game_outs

    # Create a figure and plot the line on it
    fig, ax = plt.subplots(figsize=(7,4))
    ax.axhline(1.005,color=home_color,alpha=1,xmin=(game_abs/4.25)/(game_abs+1.25),xmax=(game_abs+1)/(game_abs+1.25))
    ax.axhline(-0.005,color=away_color,alpha=1,xmin=(game_abs/4.25)/(game_abs+1.25),xmax=(game_abs+1)/(game_abs+1.25))
    inning_text_dict = game_df.filter(pl.col('game_pk')==game_choice_id).group_by('inning').agg(pl.median('ab_number'),
                                                                                            pl.max('ab_number').alias('ab_max')).sort('inning').to_dict(as_series=False)
    for inning in inning_text_dict['inning']:
        ab_number = inning_text_dict['ab_number'][inning_text_dict['inning'].index(inning)]
        ab_max = inning_text_dict['ab_max'][inning_text_dict['inning'].index(inning)]
        ax.text(ab_number-0.5,0.5,inning,ha='center',va='center',color=pl_background,
                bbox=dict(boxstyle='round', facecolor='w', alpha=0.75,edgecolor=pl_background))
        if inning != inning_text_dict['inning'][-1]:
            ax.axvline(ab_max,linestyle='--',alpha=0.25,ymin=(0.25+0.1)/1.5,ymax=(0.75+0.1)/1.5,color=pl_background)

    custom_map = colors.ListedColormap(sns.light_palette(color_dict[away_abbr][0], n_colors=50, reverse=True) +
                                       sns.light_palette(color_dict[home_abbr][0], n_colors=50))
    contrast_map = colors.ListedColormap(sns.dark_palette(color_dict[away_abbr][0], n_colors=50, reverse=True) +
                                       sns.dark_palette(color_dict[home_abbr][0], n_colors=50))
    ax.axhline(0.5,color=pl_background,alpha=0.5)

    sns.lineplot(x=x,y=y,color=pl_background)
    verts = np.column_stack([x, y]).tolist()
    verts += [[x[-1], hline_y], [x[0], hline_y]]
    clip_path = Path(verts + [verts[0]])
    clip_patch = PathPatch(clip_path, transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip_patch)

    # Gradient image spanning the fill's vertical range, colored by y-value
    # (i.e. distance from the reference line), then clipped to that polygon.
    y_min = 0
    y_max = 1
    gradient = np.linspace(y_min, y_max, 256).reshape(-1, 1)

    im = ax.imshow(
        gradient,
        extent=[x[0], x[-1], y_min, y_max],
        origin="lower",
        aspect="auto",
        cmap=sns.blend_palette([away_color,'w',home_color],as_cmap=True),
        vmin=y_min,
        vmax=y_max,
        zorder=1,
    )
    im.set_clip_path(clip_patch)

    ax.set(xlim=(-1,game_abs+0.25),
           ylim=(1.1,-.4))
    ax.axis('off')

    excite_ax = fig.add_axes([0.82,0.8,0.1,0.1], anchor='NE', zorder=1)
    excite_ax.text(0,0.9,'Excitement\nIndex',ha='center',va='center',fontsize=15)
    if abs(watch_index-5)==5:
        excite_ax.text(0,-0.4,f'{watch_index:.0f}',ha='center',va='center',size=24,
                       color='k' if abs(watch_index-5)<2.5 else 'w',
                       bbox=dict(boxstyle='circle', pad=0.3,
                                 fc=sns.color_palette('vlag',n_colors=1001)[int(watch_index*100)],
                                 ec=pl_background))
    else:
        excite_ax.text(0,-0.4,f'{watch_index:.1f}',ha='center',va='center',size=22,
                       color='k' if abs(watch_index-5)<2.5 else 'w',
                       bbox=dict(boxstyle='circle', pad=0.3,
                                 fc=sns.color_palette('vlag',n_colors=1001)[int(watch_index*100)],
                                 ec=pl_background))
    excite_ax.axis('off')

    home_team_ax = fig.add_axes([0.12,0.115,0.1,0.12], anchor='NW', zorder=1)
    image = load_team_logo(home_abbr)
    home_team_ax.imshow(image,aspect='equal')
    home_team_ax.axis('off')
    ax.text((game_abs/7),1,f'{home_score:.0f}',
            color=pl_highlight if home_score > away_score else 'k',
            fontsize=30,ha='center',va='center')

    away_team_ax = fig.add_axes([0.12,0.625,0.1,0.12], anchor='NW', zorder=1)
    image = load_team_logo(away_abbr)
    away_team_ax.imshow(image,aspect='equal')
    away_team_ax.axis('off')
    ax.text((game_abs/7),0,f'{away_score:.0f}',
            color=pl_highlight if away_score > home_score else 'k',
            fontsize=30,ha='center',va='center')


    fig.suptitle(f'{away_name} @ {home_name}',
                fontsize=25,x=0.415,y=0.92)
    fig.text(0.415,0.77,f'{game_date} - Game ID: {game_choice_id:.0f}',
                fontsize=12,ha='center')
    fig.text(0.375,0.1,'Volatility',
             ha='center', fontsize=16)
    fig.text(0.375,0.015,f'{excite_index:.0f}' if abs(excite_index-5)==5 else f'{excite_index:.1f}',
             ha='center', fontsize=20 if abs(excite_index-5)==5 else 16,
             color='k' if abs(excite_index-5)<2.5 else 'w',
             bbox=dict(boxstyle='round', pad=0.25,
                       fc=sns.color_palette('vlag',n_colors=1001)[int(np.clip(excite_index*100,0,1000))],
                       ec='k'))

    fig.text(0.55,0.1,'Tension',
             ha='center', fontsize=16)
    fig.text(0.55,0.015,f'{tension_index:.0f}' if abs(tension_index-5)==5 else f'{tension_index:.1f}',
             ha='center', fontsize=20 if abs(tension_index-5)==5 else 16,
             color='k' if abs(tension_index-5)<2.5 else 'w',
             bbox=dict(boxstyle='round', pad=0.25,
                       fc=sns.color_palette('vlag',n_colors=1001)[int(np.clip(tension_index*100,0,1000))],
                       ec='k'))

    fig.text(0.775,0.1,'Biggest Swing',
             ha='center', fontsize=16)
    fig.text(0.775,0.015,f'{biggest_swing:.0f}%',
             ha='center', fontsize=16,
             color='k' if abs(win_swing_index-5)<2.5 else 'w',
             bbox=dict(boxstyle='round', pad=0.25,
                       fc=sns.color_palette('vlag',n_colors=1001)[int(np.clip(win_swing_index*100,0,1000))],
                       ec='k'))
    fig.text(0.55,0.69,'Game Win Probability',
             ha='center', fontsize=14)
    fig.text(0.14,0.02,'Data: MLB',
             ha='left', fontsize=10)

    # Add PL logo
    # pl_ax = fig.add_axes([0.675,0.04,0.2,0.1], anchor='NE', zorder=1)
    # pl_ax.set_facecolor(pl_background)
    # # width, height = logo.size
    # # pl_ax.imshow(logo.crop((0, 0, width, height-150)))
    # pl_ax.imshow(logo)
    # pl_ax.axis('off')

    sns.despine()
    st.pyplot(fig)

game_chart(game_choice_id)

st.header(f'Excitement stats for {date:%-m/%-d/%y} games:')
st.dataframe(table_df.with_columns(
    (pl.col('excite_scale').clip(0,1).round(2)*10).alias('Volatility'),
    (pl.col('tension_scale').clip(0,1).round(2)*10).alias('Tension'),
    (pl.col('watch_score').round(1)).alias('Excitement Index'),
).rename({'game_name':'Game',
          'win_swing':'Biggest Swing'})[['Game','Volatility','Tension','Biggest Swing','Excitement Index']].sort('Excitement Index',descending=True))

# st.header('Glossary')
# st.write(f'''
# - **Δ Win Prob/54 Outs**: The total change in win probability for the game, normalized to 54 outs (a median game is ~180%). Derived from [Luke Benz's Game Excitement Index](https://lukebenz.com/post/gei/)
# ''')
# st.write(f'''
# - **Biggest Swing**: Largest swing in win probability across 6 outs (after the 1st inning; a median game is ~40%)
# ''')
# st.write(f'''
# - **Wheeee Index**: Combination of Δ Win Prob/54 Outs and Biggest Swing, scaled 0-10
# ''')
