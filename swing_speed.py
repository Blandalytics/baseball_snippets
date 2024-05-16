import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import urllib

from PIL import Image
from statsmodels.nonparametric.kernel_regression import KernelReg

st.title('MLB Swing Speed App')
st.write('Find me [@Blandalytics](https://twitter.com/blandalytics) and check out the data at [Baseball Savant](https://baseballsavant.mlb.com/leaderboard/bat-tracking)')

pl_white = '#FEFEFE'
pl_background = '#162B50'
pl_text = '#72a3f7'
pl_line_color = '#293a6b'

sns.set_theme(
    style={
        'axes.edgecolor': pl_line_color,
        'axes.facecolor': pl_background,
        'axes.labelcolor': pl_white,
        'xtick.color': pl_white,
        'ytick.color': pl_white,
        'figure.facecolor':pl_background,
        'grid.color': pl_background,
        'grid.linestyle': '-',
        'legend.facecolor':pl_background,
        'text.color': pl_white
     }
    )

count_rates = {
    (0, 0): 0.2562946734218814,
    (0, 1): 0.1299246884841846,
    (1, 1): 0.10081336437080651,
    (1, 0): 0.09744488566342599,
    (1, 2): 0.09660687388744352,
    (2, 2): 0.08257428454059976,
    (0, 2): 0.06793372586608243,
    (2, 1): 0.052126523346569906,
    (3, 2): 0.05111871833493085,
    (2, 0): 0.03293441051622621,
    (3, 1): 0.022078597836505547,
    (3, 0): 0.010149253731343283
}

all_swings = st.toggle('Include Non-Competitive swings?')

swing_data = pd.read_csv('https://github.com/Blandalytics/baseball_snippets/blob/main/swing_speed_data.csv?raw=true',encoding='latin1')
if all_swings==False:
    swing_data = swing_data.loc[(swing_data['bat_speed']>=40) &
                                (swing_data['bat_speed']>swing_data['bat_speed'].groupby(swing_data['Hitter']).transform(lambda x: x.quantile(0.1)))].copy()
swing_data['squared_up_frac'] = swing_data['squared_up_frac'].mul(100)
swing_data['blastitos'] = swing_data['blastitos'].mul(100)
swing_data['swing_time'] = swing_data['swing_time'].mul(1000)
swing_data['game_date'] = pd.to_datetime(swing_data['game_date'])
# swing_data['game_date'] = swing_data['game_date'].dt.date

col1, col2 = st.columns(2)

with col1:
    swing_threshold = st.number_input(f'Min # of Swings (in all situations):',
                                      min_value=0, 
                                      max_value=swing_data.groupby('Hitter')['Swings'].sum().max(),
                                      step=25, 
                                      value=100)
with col2:
    team = st.selectbox('Team:',
                        ['All','ATL', 'AZ', 'BAL', 'BOS', 'CHC', 'CIN', 'CLE', 'COL', 'CWS',
                         'DET', 'HOU', 'KC', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM',
                         'NYY', 'OAK', 'PHI', 'PIT', 'SD', 'SEA', 'SF', 'STL', 'TB', 'TEX',
                         'TOR', 'WSH'])

count_select = st.radio('Count Group', 
                        ['All','Hitter-Friendly','Pitcher-Friendly','Even','2-Strike','3-Ball','Custom'],
                        index=0,
                        horizontal=True
                       )
 
if count_select=='All':
    selected_options = ['0-0', '1-0', '2-0', '3-0', '0-1', '1-1', '2-1', '3-1', '0-2', '1-2', '2-2', '3-2']
elif count_select=='Hitter-Friendly':
    selected_options = ['1-0', '2-0', '3-0', '2-1', '3-1']
elif count_select=='Pitcher-Friendly':
    selected_options = ['0-1','0-2','1-2','2-2']
elif count_select=='Even':
    selected_options = ['0-0','1-1','3-2']
elif count_select=='2-Strike':
    selected_options = ['0-2','1-2','2-2','3-2']
elif count_select=='3-Ball':
    selected_options = ['3-0','3-1','3-2']
else:
    selected_options = st.multiselect('Select the count(s):',
                                       ['0-0', '1-0', '2-0', '3-0', '0-1', '1-1', '2-1', '3-1', '0-2', '1-2', '2-2', '3-2'],
                                       ['0-0', '1-0', '2-0', '3-0', '0-1', '1-1', '2-1', '3-1', '0-2', '1-2', '2-2', '3-2'])

count_threshold = sum([count_rates[(int(x[0]),int(x[-1]))] for x in selected_options])
updated_threshold = int(round(swing_threshold*count_threshold,0))

stat_name_dict = {
    'bat_speed':'Swing Speed (mph)',
    'swing_length':'Swing Length (ft)',
    'swing_time':'Swing Time (ms)',
    'swing_acceleration':'Swing Acceleration (ft/s^2)',
    'squared_up_frac':'Squared Up%',
    # 'blastitos':'Partial Blasts / Swing'
}

df_stat_dict = {
    'bat_speed':'Speed (mph)',
    'swing_length':'Length (ft)',
    'swing_time':'Time (ms)',
    'swing_acceleration':'Acceleration (ft/s^2)',
    'squared_up_frac':'SU%',
    'blastitos':'PB%',
}

st.write('Swing Metrics')
st.dataframe((swing_data if team=='All' else swing_data.loc[swing_data['Team']==team])
             .loc[swing_data['count'].isin(selected_options)]
             .fillna({'blastitos':0})
             .groupby(['Hitter'])
             [['Team','Swings','bat_speed','swing_length','swing_time','swing_acceleration','squared_up_frac',
               'blastitos'
              ]]
             .agg({
                 'Team':lambda x: pd.Series.unique(x)[-1],
                 'Swings':'count',
                 'bat_speed':'mean',
                 'swing_length':'mean',
                 'swing_time':'mean',
                 'swing_acceleration':'mean',
                 'squared_up_frac':'mean',
                 'blastitos':'mean'
             })
             .query(f'Swings >={updated_threshold}')
             .sort_values('swing_acceleration',ascending=False)
             .round(1)
             .rename(columns=df_stat_dict)
)

players = list(swing_data
               .groupby('Hitter')
               [['Swings','swing_acceleration']]
               .agg({'Swings':'count','swing_acceleration':'mean'})
               .query(f'Swings >={swing_threshold}')
               .reset_index()
               .sort_values('swing_acceleration', ascending=False)
               ['Hitter']
              )

col1, col2 = st.columns(2)

with col1:
    player = st.selectbox('Choose a player:', players)
with col2:
    stat = st.selectbox('Choose a metric:', list(stat_name_dict.values()))
    stat = list(stat_name_dict.keys())[list(stat_name_dict.values()).index(stat)]
    players = list(swing_data
                   .loc[swing_data['count'].isin(selected_options)]
               .groupby('Hitter')
               [['Swings',stat]]
               .agg({'Swings':'count',stat:'mean'})
               .query(f'Swings >={updated_threshold}')
               .reset_index()
               .sort_values(stat, ascending=False)
               ['Hitter']
              )

col1, col2 = st.columns(2)
season_start = swing_data.loc[swing_data['Hitter']==player,'game_date'].min()
season_end = swing_data.loc[swing_data['Hitter']==player,'game_date'].max()
with col1:
    start_date = st.date_input(f"Start Date (Season started: {season_start:%b %d})", 
                               season_start,
                               min_value=season_start,
                               max_value=season_end,
                               format="MM/DD/YYYY")
with col2:
    end_date = st.date_input(f"End Date (Season ended: {season_end:%b %d})", 
                             season_end,
                             min_value=season_start,
                             max_value=season_end,
                             format="MM/DD/YYYY")
    
def speed_dist(swing_data,player,stat):
    fig, ax = plt.subplots(figsize=(6,3))
    swing_data = swing_data.loc[(swing_data['game_date'].dt.date>=start_date) &
                                (swing_data['game_date'].dt.date<=end_date) &
                                (swing_data['count'].isin(selected_options))].copy()

    val = swing_data.loc[swing_data['Hitter']==player,stat].mean()
    color_list = sns.color_palette('vlag',n_colors=len(players))
    player_color = color_list[len(players)-players.index(player)-1] if stat not in ['swing_length','swing_time'] else color_list[players.index(player)-1]
    sns.kdeplot(swing_data.loc[swing_data['Hitter']==player,stat],
                    color=player_color,
                    fill=True,
                    cut=0)
                    
    g = sns.kdeplot(swing_data[stat],
                    linestyle='--',
                    color='w',
                    alpha=0.5,
                    cut=0)
    
    league_val = swing_data[stat].mean()
    kdeline_g = g.lines[0]
    xs_g = kdeline_g.get_xdata()
    ys_g = kdeline_g.get_ydata()
    height_g = np.interp(league_val, xs_g, ys_g)
    ax.vlines(league_val, 0, height_g, color='w',alpha=0.5,linestyle='--')

    p = sns.kdeplot(swing_data.loc[swing_data['Hitter']==player,stat],
                    color=player_color,
                    cut=0)
    if all_swings==True:
        xlim = (min(swing_data[stat].quantile(0.01),swing_data.loc[swing_data['Hitter']==player,stat].quantile(0.02)),
                 max(200,swing_data.loc[swing_data['Hitter']==player,stat].quantile(0.99)) if stat == 'swing_time' else swing_data[stat].max())
    else:
        xlim = (ax.get_xlim()[0],ax.get_xlim()[1])

    if stat=='squared_up_frac':
        xlim = (38,xlim[1])
    
    ax.set(xlim=xlim,
           xlabel=stat_name_dict[stat],
           ylabel='',
          ylim=(0,ax.get_ylim()[1]*1.15))

    plt.legend(labels=[player,'league Avg','MLB'][::2],
               loc='upper center',
               ncol=2,
               edgecolor=pl_background)
    
    kdeline_p = p.lines[1]
    xs_p = kdeline_p.get_xdata()
    ys_p = kdeline_p.get_ydata()
    height_p = np.interp(val, xs_p, ys_p)
    ax.vlines(val, ax.get_ylim()[1]*0.1, height_p, color=player_color)
    
    measure = '%' if stat == 'squared_up_frac' else stat_name_dict[stat].split(' ')[-1][1:-1]
    ax.text(val,ax.get_ylim()[1]*0.1,f'{val:.1f}{measure}',va='center',ha='center',color=player_color,
            bbox=dict(facecolor=pl_background, alpha=0.9,edgecolor=player_color))

    if stat=='squared_up_frac':
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(100,decimals=0))
    ax.set_yticks([])
    title_stat = 'Squared Up%' if stat == 'squared_up_frac' else ' '.join(stat_name_dict[stat].split(' ')[:-1])
    apostrophe_text = "'" if player[-1]=='s' else "'s"
    date_text = '' if (start_date==season_start) & (end_date==season_end) else f' ({start_date:%b %-d} - {end_date:%b %-d})'
    count_text = '' if count_select=='All' else f'\nin {count_select} counts'
    fig.suptitle(f"{player}{apostrophe_text}\n{title_stat}{date_text}{count_text}",y=1.025 if count_select=='All' else 1.075)
    sns.despine(left=True)
    fig.text(0.8,-0.15,'@blandalytics\nData: Savant',ha='center',fontsize=10)
    fig.text(0.125,-0.14,'mlb-swing-speed.streamlit.app',ha='left',fontsize=10)
    st.pyplot(fig)
speed_dist(swing_data,player,stat)

def rolling_chart(df,player,stat):
    rolling_df = (df
                  .loc[(df['Hitter']==player) &
                          (df['count'].isin(selected_options)),
                       ['Hitter','game_date',stat]]
                  .dropna()
                  .reset_index(drop=True)
                  .reset_index()
                  .rename(columns={'index':'swings'})
                 )
    
    swing_thresh = max(25,int(round(swing_threshold/2/5,0))*5)
    season_avg = rolling_df[stat].mean()
    rolling_df['Rolling_Stat'] = rolling_df[stat].rolling(swing_thresh).mean()
    rolling_df = rolling_df.loc[rolling_df['swings']==rolling_df['swings'].groupby(rolling_df['game_date']).transform('max')].copy()
    
    chart_thresh_list = (df
                         .loc[(df['count'].isin(selected_options))]
                         .assign(Swing=1)
                         .groupby('Hitter')
                         [['Swing',stat]]
                         .agg({
                             'Swing':'count',
                             stat:'mean'
                         })
                         .query(f'Swing >= {updated_threshold}')
                         .copy()
                        )
    chart_avg = chart_thresh_list[stat].mean()
    chart_stdev = chart_thresh_list[stat].std()

    chart_90 = chart_thresh_list[stat].quantile(0.9)
    chart_75 = chart_thresh_list[stat].quantile(0.75)
    chart_25 = chart_thresh_list[stat].quantile(0.25)
    chart_10 = chart_thresh_list[stat].quantile(0.1)
    
    y_pad = (chart_90-chart_10)/5
    chart_min = min(rolling_df['Rolling_Stat'].min(),chart_10) - y_pad
    chart_max = max(rolling_df['Rolling_Stat'].max(),chart_90) + y_pad

    if stat in ['swing_length','swing_time']:
        chart_90 = chart_thresh_list[stat].quantile(0.1)
        chart_75 = chart_thresh_list[stat].quantile(0.25)
        chart_25 = chart_thresh_list[stat].quantile(0.75)
        chart_10 = chart_thresh_list[stat].quantile(0.9)
    
    line_text_loc = rolling_df['game_date'].min() + pd.Timedelta(days=(rolling_df['game_date'].max() - rolling_df['game_date'].min()).days * 1.05)
    
    fig, ax = plt.subplots(figsize=(6,6))
    sns.lineplot(data=rolling_df,
                     x='game_date',
                     y='Rolling_Stat',
                     color='w'
                       )
    
    ax.axhline(season_avg, 
               color='w',
               linestyle='--')
    ax.text(line_text_loc,
            season_avg,
            'Szn Avg',
            va='center',
            color='w')
    
    # Threshold Lines
    ax.axhline(chart_90,
               color=sns.color_palette('vlag', n_colors=100)[99],
               alpha=0.6)
    ax.axhline(chart_75,
               color=sns.color_palette('vlag', n_colors=100)[79],
               linestyle='--',
               alpha=0.5)
    ax.axhline(chart_avg,
               color='w',
               alpha=0.5)
    ax.axhline(chart_25,
               color=sns.color_palette('vlag', n_colors=100)[19],
               linestyle='--',
               alpha=0.5)
    ax.axhline(chart_10,
               color=sns.color_palette('vlag', n_colors=100)[0],
               alpha=0.6)
    
    ax.text(line_text_loc,
            chart_90,
            '90th %' if abs(chart_90 - season_avg) > (ax.get_ylim()[1] - ax.get_ylim()[0])/25 else '',
            va='center',
            color=sns.color_palette('vlag', n_colors=100)[99],
            alpha=1)
    ax.text(line_text_loc,
            chart_75,
            '75th %' if abs(chart_75 - season_avg) > (ax.get_ylim()[1] - ax.get_ylim()[0])/25 else '',
            va='center',
            color=sns.color_palette('vlag', n_colors=100)[74],
            alpha=1)
    ax.text(line_text_loc,
            chart_avg,
            'MLB Avg' if abs(chart_avg - season_avg) > (ax.get_ylim()[1] - ax.get_ylim()[0])/25 else '',
            va='center',
            color='w',
            alpha=0.75)
    ax.text(line_text_loc,
            chart_25,
            '25th %' if abs(chart_25 - season_avg) > (ax.get_ylim()[1] - ax.get_ylim()[0])/25 else '',
            va='center',
            color=sns.color_palette('vlag', n_colors=100)[24],
            alpha=1)
    ax.text(line_text_loc,
            chart_10,
            '10th %' if abs(chart_10 - season_avg) > (ax.get_ylim()[1] - ax.get_ylim()[0])/25 else '',
            va='center',
            color=sns.color_palette('vlag', n_colors=100)[9],
            alpha=1)
    
    ax.set(xlabel='Game Date',
           ylabel=stat_name_dict[stat],
           ylim=(chart_min, 
                 chart_max)           
          )
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator,
                                            show_offset=False,
                                           formats=['%Y', '%-m/1', '%-m/%-d', '%H:%M', '%H:%M', '%S.%f'])
    if stat=='squared_up_frac':
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(100,decimals=0))
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    metric_text = 'Squared Up%' if stat == 'squared_up_frac' else ' '.join(stat_name_dict [stat].split(' ')[:-1])
    apostrophe_text = "'" if player[-1]=='s' else "'s"
    count_text = '' if count_select=='All' else f'; in {count_select} counts'
    fig.suptitle(f"{player}{apostrophe_text} {metric_text}\nRolling {swing_thresh} Swings{count_text}",
                 fontsize=14,
                 y=0.95
                 )
    fig.text(0.9,-0.025,'@blandalytics\nData: Savant',ha='center',fontsize=10)
    fig.text(0.025,-0.02,'mlb-swing-speed.streamlit.app',ha='left',fontsize=10)
    sns.despine()
    st.pyplot(fig)
st.write('The rolling chart uses either 25 swings or ~1/2 of the Swings seen in that count, whichever is larger')
rolling_chart(swing_data,player,stat)

zone_df = pd.DataFrame(columns=['x','z'])
for x in range(-20,21):
    for y in range(0,55):
        zone_df.loc[len(zone_df)] = [x/12,y/12]

heatmap_stat_dict = {
        'bat_speed':['bs_oa','Bat Speed'],
        'swing_length':['sl_oa','Swing Length'],
        'swing_acceleration':['sa_oa','Swing Acceleration'],
        'swing_time':['st_oa','Swing Time'],
        'squared_up_frac':['su_oa','Squared Up%']
    }

def heatmap_data(df,stat):
    heatmap_df = df.loc[(df['plate_x'].abs()<=2) &
                        (df['sz_z'].abs()<=1.5)].dropna(subset=['bat_speed']).copy()
    heatmap_df.loc[heatmap_df['plate_x'].notna(),'kde_x'] = np.clip(heatmap_df.loc[heatmap_df['plate_x'].notna(),'plate_x'].astype('float').mul(12).round(0).astype('int').div(12),
                                                -20/12,
                                                20/12)
    heatmap_df.loc[heatmap_df['sz_z'].notna(),'kde_z'] = np.clip(heatmap_df.loc[heatmap_df['sz_z'].notna(),'sz_z'].astype('float').mul(24).round(0).astype('int').div(24),
                                                 -1.5,
                                                 1.25)
    
    heatmap_df['base_'+stat] = heatmap_df[stat].groupby([heatmap_df['stand'],
                                                    heatmap_df['kde_x'],
                                                    heatmap_df['kde_z'],
                                                    heatmap_df['count']]).transform('mean')

    
    heatmap_df[heatmap_stat_dict[stat][0]] = heatmap_df['base_'+stat].sub(heatmap_df[stat]) if stat in ['swing_time','swing_length'] else heatmap_df[stat].sub(heatmap_df['base_'+stat])
    heatmap_df.loc[heatmap_df['sz_z'].notna(),'kde_z'] = np.clip(heatmap_df.loc[heatmap_df['sz_z'].notna(),'plate_z'].astype('float').mul(12).round(0).astype('int').div(12),
                                                 0,
                                                 4.5)
    return heatmap_df[['Hitter','stand',heatmap_stat_dict[stat][0],'plate_x','sz_z','kde_x','kde_z','sz_top','sz_bot']]

def swing_heatmap(df,hitter,base_stat):
    b_hand = df.loc[(df['Hitter']==hitter),'stand'].value_counts().index[0]
    stat_dict = {
        heatmap_stat_dict[base_stat][0]:[heatmap_stat_dict[base_stat][1],swing_data[base_stat].mean()/(20 if stat=='swing_acceleration' else 40]
    }
    
    bandwidth = np.clip(df
                        .loc[(df['Hitter']==hitter)]
                        .shape[0]/100,
                        0.2,
                        0.25)
    
    sz_top = round(df.loc[df['Hitter']==hitter,'sz_top'].median()*12)
    sz_bot = round(df.loc[df['Hitter']==hitter,'sz_bot'].median()*12)
    sz_range = sz_top-sz_bot
    sz_mid = sz_bot + sz_range/2
    
    for stat in stat_dict.keys():
        fig, ax = plt.subplots(figsize=(5,6))
        v_center = df[stat].mean()
        kde_df = pd.merge(zone_df,
                          (df
                           .loc[(df['Hitter']==hitter)]
                           .dropna(subset=[stat,'plate_x','sz_z'])
                           [['kde_x','kde_z',stat]]
                          ),
                          how='left',
                          left_on=['x','z'],
                          right_on=['kde_x','kde_z']).fillna({stat:v_center})
        
        kernel_regression = KernelReg(endog=kde_df[stat], 
                                      exog= [kde_df['x'], kde_df['z']], 
                                      bw=[bandwidth,bandwidth],
                                      var_type='cc')
        kde_df['kernel_stat'] = kernel_regression.fit([kde_df['x'], kde_df['z']])[0]
        del kernel_regression
        kde_df = kde_df.pivot_table(columns='x',index='z',values=['kernel_stat'], aggfunc='mean')

        sns.heatmap(data=kde_df['kernel_stat'].astype('float'),
                    cmap='vlag',
                    center=v_center,
                    vmin=v_center-stat_dict[stat][1],
                    vmax=v_center+stat_dict[stat][1],
                    ax=ax,
                    cbar=False
                   )

        ax.set(xlabel=None, ylabel=None)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)

        ax.set(xlim=(40,0),
               ylim=(6,54),
               aspect=1)

        # Strikezone
        ax.axhline(sz_bot, xmin=1/4, xmax=3/4, color='black', linewidth=2)
        ax.axhline(sz_top, xmin=1/4, xmax=3/4, color='black', linewidth=2)
        ax.axvline(10, ymin=(sz_bot-6)/48, ymax=(sz_top-6)/48, color='black', linewidth=2)
        ax.axvline(30, ymin=(sz_bot-6)/48, ymax=(sz_top-6)/48, color='black', linewidth=2)

        # Inner Strikezone
        ax.axhline(sz_bot+sz_range/3, xmin=1/4, xmax=3/4, color='black', linewidth=1)
        ax.axhline(sz_bot+2*sz_range/3, xmin=1/4, xmax=3/4, color='black', linewidth=1)
        ax.axvline(10+20/3, ymin=(sz_bot-6)/48, ymax=(sz_top-6)/48, color='black', linewidth=1)
        ax.axvline(30-20/3, ymin=(sz_bot-6)/48, ymax=(sz_top-6)/48, color='black', linewidth=1)

        # Plate
        ax.plot([11.27,27.73], [7,7], color='k', linewidth=1)
        ax.plot([11.25,11.5], [7,8], color='k', linewidth=1)
        ax.plot([27.75,27.5], [7,8], color='k', linewidth=1)
        ax.plot([27.43,20], [8,9], color='k', linewidth=1)
        ax.plot([11.57,20], [8,9], color='k', linewidth=1)
        
        ax.text(37.5 if b_hand=='L' else 2.5,
                                sz_mid,
                                'Stands Here',
                                rotation=270 if b_hand=='L' else 90,
                                fontsize=14,
                                color='k',
                                ha='center',
                                va='center',
                                bbox=dict(boxstyle='round',
                                          color='w',
                                          alpha=0.5,
                                          pad=0.2))
        
        kde_thresh=0.05
        # Add PL logo
        logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
        logo = Image.open(urllib.request.urlopen(logo_loc))
        pl_ax = fig.add_axes([0.36,-0.19,0.32,0.32], anchor='NE', zorder=1)
        pl_ax.imshow(logo)
        pl_ax.set(ylim=(390,0))
        pl_ax.axis('off')
        apostrophe_text = "'" if hitter[-1]=='s' else "'s"
        fig.suptitle(f"{hitter}{apostrophe_text}\n{stat_dict[stat][0]} Heatmap",y=0.96)
        sns.despine(left=True,bottom=True)
        st.pyplot(fig)
swing_heatmap(heatmap_data(swing_data,stat),player,stat)

st.header('Assumptions & Formulas')
st.write('Assumptions:')
st.write('- Initial speed is 0mph')
st.write('- Swing Speed is recorded at the same point & time as Swing Length')
st.write('- Speed of pitch at plate = [~0.92 * Release Speed](https://twitter.com/tangotiger/status/1790432119275082139)')
st.write('- Collision Efficiency = [0.23](http://tangotiger.com/index.php/site/article/statcast-lab-collisions-and-the-perfect-swing)')
st.write("- Squared Up% can't be >100%")
st.write('Formulas:')
st.write('- Initial Speed (v_0; in ft/s) = 0 ')
st.write('- Final Speed (v_f; in ft/s) = Swing Speed * 1.46667 (from Savant; converted from mph to ft/s)')
st.write('- Swing Length (d; in ft) = Swing Length (from Savant)')
st.write('- Average Speed (v_avg; in ft/s) = (v_f - v_i)/2')
st.write('- Swing Time (t; in s) = d / v_avg')
st.write('- Swing Acceleration (a; in ft/s^2) = v_f / t')
st.write('- Collision Efficiency (q; unitless) = 0.23')
st.write('- Max Possible EV (in mph) = (Pitch Speed at Plate * q) + [Swing Speed * (1 + q]')
st.write('- Squared Up% (SU%; % of Max Possible EV) = Exit Velocity / Max Possible EV')
st.write('- Partial Blasts (PB) = 1/(1 + exp(-(-90.6 + Swing Speed * 0.57 + Squared Up% * 53.52)))')
st.write('- Partial Blast Rate (PB%) = PB / Swing')
