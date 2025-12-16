import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import urllib
import datetime
from PIL import Image
from matplotlib import ticker
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from matplotlib import colors
import matplotlib.font_manager as fm
import os

from pyfonts import set_default_font, load_google_font

font = load_google_font("Alexandria")
fm.fontManager.addfont(str(font.get_file()))

logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
logo = Image.open(urllib.request.urlopen(logo_loc))
st.image(logo, width=200)

# st.title("NFBC Draft Data, over Time")
new_title = '<p style="color:#72CBFD; font-weight: bold; font-size: 42px;">NFBC Draft Data, over Time</p>'
st.markdown(new_title, unsafe_allow_html=True)

@st.cache_data(ttl=600,show_spinner=f"Loading draft data")
def load_data():
  df = pd.read_parquet('https://github.com/Blandalytics/baseball_snippets/blob/main/nfbc_adp_data.parquet?raw=true')
  df['start_date'] = pd.to_datetime(df['start_date']).dt.date
  df['end_date'] = pd.to_datetime(df['end_date']).dt.date
  df['yahoo_pos'] = np.where(df['Position(s)'].apply(lambda x: 'P' in ', '.join(x)),
                             df['yahoo_pos'].str.split(', '),#df['yahoo_pos'].apply(lambda x: [y.replace('DH','UT') for y in x]),
                             df['Position(s)'].str.split(', '))
  return df

nfbc_adp_df = load_data()
default_player = 'Ben Rice'
default_player_pos = nfbc_adp_df.loc[nfbc_adp_df['Player']==default_player,'yahoo_pos'].iloc[0]
default_player_group = ['H'] if 'P' not in ', '.join(default_player_pos) else ['P']

update_date = nfbc_adp_df['end_date'].max().strftime('%-m/%-d/%y')
st.write(f'Data is through {update_date}')
## Set Styling
# Plot Style
pl_white = '#FEFEFE'
pl_background = '#292C42'
pl_text = '#72a3f7'
pl_line_color = '#293a6b'

sns.set_theme(
    style={
        'axes.edgecolor': pl_white,
        'axes.facecolor': pl_background,
        'axes.labelcolor': pl_white,
        'xtick.color': pl_white,
        'ytick.color': pl_white,
        'figure.facecolor':pl_background,
        'grid.color': pl_background,
        'grid.linestyle': '-',
        'legend.facecolor':pl_background,
        'text.color': pl_white
     },
    font='Alexandria'
    )

chart_red = '#D74C36'#sns.color_palette('vlag',n_colors=10000)[-1]
chart_blue = '#72CBFD'#sns.color_palette('vlag',n_colors=10000)[0]

def format_dollar_amount(amount):
    formatted_absolute_amount = '${:.1f}'.format(abs(amount))
    if round(amount, 2) < 0:
        return f'-{formatted_absolute_amount}'
    return formatted_absolute_amount

col1, col2 = st.columns(2)
with col1:
    adp_start_date = st.date_input("ADP Start Date", 
                                   datetime.date(2025,10,15),
                                   min_value=datetime.date(2025,10,1),
                                   max_value=datetime.date.today() - datetime.timedelta(days=7),
                                   format="MM/DD/YYYY")
    adp_thresh = 1000
    start_string = adp_start_date.strftime('%-m/%-d')
with col2:
    pos_filters = [
        'All',
        'H','P',
        'C','1B','2B','SS','3B','OF','UT',
        'SP','RP'
    ]
    
    pos_filter = st.selectbox('Select a position filter:',pos_filters)
    if pos_filter=='All':
        nfbc_adp_df = nfbc_adp_df.copy()
    elif pos_filter in ['H','P']:
        if pos_filter=='H':
            position_mask = nfbc_adp_df['yahoo_pos'].apply(lambda x: 'P' not in ', '.join(x))
            nfbc_adp_df = nfbc_adp_df.loc[position_mask].copy()
        else:
            position_mask = nfbc_adp_df['yahoo_pos'].apply(lambda x: 'P' in ', '.join(x))
            nfbc_adp_df = nfbc_adp_df.loc[position_mask].copy()
    else:
        position_mask = nfbc_adp_df['yahoo_pos'].apply(lambda x: pos_filter in x)
        nfbc_adp_df = nfbc_adp_df.loc[position_mask].copy()
    
    pos_text = '' if pos_filter =='All' else f' ({pos_filter}-Eligible)'

@st.cache_data(show_spinner=f"Generating positional chart")
def position_chart(adp_start_date,nfbc_adp_df=nfbc_adp_df):
    start_string = adp_start_date.strftime('%-m/%-d')
    med_values = {x:[] for x in position_list}
    for end_date in pd.date_range(start=adp_start_date, end=nfbc_adp_df['end_date'].max()):
        for pos in med_values.keys():
            med_values[pos] += [(pd
                               .merge(nfbc_adp_df.loc[(pd.to_datetime(nfbc_adp_df['end_date']).dt.date == adp_start_date) & (nfbc_adp_df['yahoo_pos'].str.contains(pos)),['Player ID','Player','yahoo_pos','ADP']].query('ADP <= 400'),
                                      nfbc_adp_df.loc[(nfbc_adp_df['end_date'] == end_date) & (nfbc_adp_df['yahoo_pos'].str.contains(pos)),['Player ID','Player','ADP']].query('ADP <= 400'),
                                      how='inner',
                                      on=['Player ID','Player'],
                                      suffixes=['_early','_current'])
                               .assign(perc_diff = lambda x: (-(x['ADP_current']-x['ADP_early'])/x['ADP_early']) * 100)
                               ['perc_diff']
                               .median()
                               )]
    all_values = []
    for end_date in pd.date_range(start=adp_start_date, end=nfbc_adp_df['end_date'].max()):
        all_values += [(pd
                               .merge(nfbc_adp_df.loc[(pd.to_datetime(nfbc_adp_df['end_date']).dt.date == adp_start_date),['Player ID','Player','yahoo_pos','ADP']].query('ADP <= 400'),
                                      nfbc_adp_df.loc[(nfbc_adp_df['end_date'] == end_date),['Player ID','Player','ADP']].query('ADP <= 400'),
                                      how='inner',
                                      on=['Player ID','Player'],
                                      suffixes=['_early','_current'])
                               .assign(perc_diff = lambda x: (-(x['ADP_current']-x['ADP_early'])/x['ADP_early']) * 100)
                               ['perc_diff']
                               .median()
                               )]
    
    med_values.update({'All':all_values})
    color_map = {
        'SP':'#0C3E9B', 
        'RP':'#2674C5', 
        'C':'#696666', 
        '1B':'#794C30', 
        '2B':'#B5168A', 
        '3B':'#286F04', 
        'SS':'#872B2C', 
        'OF':'#B1550D', 
        'All':'#ffffff'
    }
    fig, ax = plt.subplots(figsize=(6,5))
    sns.lineplot(pd.DataFrame(med_values,index=pd.date_range(start=adp_start_date, end=nfbc_adp_df['end_date'].max())).melt(value_name='Cost',ignore_index=False).reset_index().rename(columns={'index':'Date'}),
                 x='Date',
                 y='Cost',
                 hue='variable',
                palette=['#ffffff']*9,
                linewidth=3,legend=False)
    sns.lineplot(pd.DataFrame(med_values,index=pd.date_range(start=adp_start_date, end=nfbc_adp_df['end_date'].max())).melt(value_name='Cost',ignore_index=False).reset_index().rename(columns={'index':'Date'}),
                 x='Date',
                 y='Cost',
                 hue='variable',
                palette=color_map,
                linewidth=2.5)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator,
                                            show_offset=False,
                                           formats=['%-m/%-d/%y', '%-m/%-d', '%-m/%-d', '%H:%M', '%H:%M', '%S.%f'])
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set(ylim=(min(-6,ax.get_ylim()[0]),ax.get_ylim()[1]))
    ax.set_yticklabels([f'{y/100:+.0%}' for y in ax.get_yticks()])
    ax.legend(title='',edgecolor=pl_background)
    ax.axhline(0,color='w',alpha=0.5,linestyle='--')
    fig.suptitle('Median Change in Draft Pick Cost, by Position')
    sns.despine()
    st.pyplot(fig)

position_chart(adp_start_date)

adp_diff_df = (pd
               .merge(nfbc_adp_df.loc[nfbc_adp_df['end_date'] == adp_start_date,['Player ID','Player','yahoo_pos','ADP']],
                      nfbc_adp_df.loc[nfbc_adp_df['end_date'] == nfbc_adp_df['end_date'].max(),['Player ID','Player','ADP']],
                      how='right',
                      on=['Player ID','Player'],
                      suffixes=['_early','_current'])
               .assign(perc_diff = lambda x: (x['ADP_current']-x['ADP_early'])/x['ADP_early'] * 100,
                       val_diff = lambda x: -10.19 * (np.log(x['ADP_current']) - np.log(x['ADP_early'])))
               .query(f'ADP_current <= {adp_thresh}')
               .sort_values('perc_diff',ascending=True)
               .round(1)
               .rename(columns={'yahoo_pos':'Pos',
                                'ADP_early':start_string,
                                'ADP_current':'Current',
                                'perc_diff':'% Diff',
                                'val_diff':'Val Diff'})
               .drop(columns=['Player ID'])
               )

st.write('Value Diff is the modeled Auction Value of the Current Rank minus the modeled Auction Value of the Early Rank')
col1, col2 = st.columns(2)
with col1:
    st.write(f'Biggest risers since {adp_start_date.strftime('%-m/%-d/%y')}{pos_text}')
    st.dataframe(adp_diff_df.drop(columns=['Pos']).sort_values('% Diff',ascending=True).head(25)
                 .style
                 .format(precision=1, thousands='')
                 .format(format_dollar_amount,subset=['Val Diff'])
                 .background_gradient(axis=0, vmin=-50, vmax=50,
                                      cmap="vlag_r", subset=['% Diff'])
                 .background_gradient(axis=0, vmin=-7, vmax=7,
                                      cmap="vlag", subset=['Val Diff']),
                 hide_index=True
                 )

with col2:
    st.write(f'Biggest fallers since {adp_start_date.strftime('%-m/%-d/%y')}{pos_text}')
    st.dataframe(adp_diff_df.drop(columns=['Pos']).sort_values('% Diff',ascending=False).head(25)
                 .style
                 .format(precision=1, thousands='')
                 .format(format_dollar_amount,subset=['Val Diff'])
                 .background_gradient(axis=0, vmin=-50, vmax=50,
                                      cmap="vlag_r", subset=['% Diff'])
                 .background_gradient(axis=0, vmin=-7, vmax=7, 
                                      cmap="vlag", subset=['Val Diff']),
                 hide_index=True
                 )

player_list = list(
  nfbc_adp_df
  .assign(weighted_adp = lambda x: x['# Picks'].mul(x['ADP']))
  .groupby('Player')
  [['# Picks','weighted_adp']]
  .sum()
  .assign(adp = lambda x: x['weighted_adp'].div(x['# Picks']))
  .sort_values('adp')
  .rename(columns={'# Picks':'num_picks'})
  .query('num_picks >= 500')
  .index
)

if pos_filter in ['All']+default_player_group+default_player_pos:
    player_index = player_list.index(default_player)
else:
    player_index = 0

player = st.selectbox('Choose a player:', player_list,
                      index=player_index)

start_date = datetime.date(2024,10,21)
def plot_draft_data(df,player,start_date):
  chart_df = df.loc[(df['Player']==player) & (df['end_date'] >= start_date)].copy()
  chart_start = start_date.strftime('%-m/%-d/%y')
  chart_end = chart_df['end_date'].max()
  chart_end = chart_end.strftime('%-m/%-d/%y')
  
  fig, ax = plt.subplots(figsize=(6,4))
  sns.lineplot(chart_df,
               x='end_date',
               y='ADP',
               color='#F1C647')
  sns.lineplot(chart_df,
               x='end_date',
               y='Min Pick',color=chart_red,
              )
  sns.lineplot(chart_df,
               x='end_date',
               y='Max Pick',color=chart_blue,
              )
  
  ax.fill_between(chart_df['end_date'], 
                  np.clip(chart_df['ADP'] - chart_df['StDev Est'],chart_df['Min Pick'],chart_df['Max Pick']), 
                  np.clip(chart_df['ADP'] + chart_df['StDev Est'],chart_df['Min Pick'],chart_df['Max Pick']), 
                  color='w',
                  alpha=0.2,
                 )
  
  adp_val = chart_df.iloc[-1]['ADP']
  chart_range = ax.get_ylim()[1] - ax.get_ylim()[0]
  ax.text(chart_df.iloc[-1]['end_date'] + datetime.timedelta(days=2),
          adp_val,
          f'ADP: {adp_val:.1f}', color='#F1C647',
           ha='left',va='center')
  
  min_val = chart_df.iloc[-1]['Min Pick']
  min_adj = min(min_val,adp_val - chart_range*0.055)
  ax.text(chart_df.iloc[-1]['end_date'] + datetime.timedelta(days=2),
          min_adj,
          f'Min Pick: {min_val:.0f}',
           ha='left',va='center', color=chart_red)
  
  max_val = chart_df.iloc[-1]['Max Pick']
  max_adj = max(max_val,adp_val + chart_range*0.055)
  ax.text(chart_df.iloc[-1]['end_date'] + datetime.timedelta(days=2),
          max_adj,
          f'Max Pick: {max_val:.0f}',
           ha='left',va='center',color=chart_blue)
   
  text_mask = max_val - (adp_val + chart_df.iloc[-1]['StDev Est'])  >= chart_range * 0.05
  ax.text(chart_df.iloc[-1]['end_date'] + datetime.timedelta(days=2),
          chart_df.iloc[-1]['ADP'] + chart_df.iloc[-1]['StDev Est'],
          'St Dev (est)' if text_mask else '',
           ha='left',va='center',color='#aaaaaa')
  
  ax.set(xlim=(ax.get_xlim()[0],chart_df['end_date'].max()),
         ylim=(ax.get_ylim()[1],max(0,ax.get_ylim()[0])),
         xlabel='')
  
  locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
  formatter = mdates.ConciseDateFormatter(locator,
                                          show_offset=False,
                                         formats=['%-m/%-d/%y', '%-m/%-d', '%-m/%-d', '%H:%M', '%H:%M', '%S.%f'])
  ax.xaxis.set_major_locator(locator)
  ax.xaxis.set_major_formatter(formatter)
  
  # Add PL logo
  logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
  logo = Image.open(urllib.request.urlopen(logo_loc)).crop((50,150,1300, 400))
  pl_ax = fig.add_axes([0.44,-0.04,0.2,0.2], anchor='S', zorder=1)
  pl_ax.imshow(logo)
  pl_ax.axis('off')
  
  line_one_text = f"{player}'s NFBC Draft Data"
  line_two_text = f"Rolling 14-Days: {chart_start} - {chart_end}"
  fig.text(0.25,1,line_one_text,ha='left',va='top',color='w',fontsize=14)
  fig.text(0.25,1,f"{player}",ha='left',va='top',color='#F1C647',fontsize=14)
  fig.text(0.25,0.95,line_two_text,ha='left',va='top',color='w',fontsize=14)
  sns.despine(left=True)
  st.pyplot(fig)

plot_draft_data(nfbc_adp_df,player,adp_start_date)
st.write(f'ADP differences since {adp_start_date.strftime('%-m/%-d/%y')}{pos_text}')
st.dataframe(adp_diff_df.sort_values('Current').reset_index(drop=True).reset_index().assign(index=lambda x: x['index'].add(1)).rename(columns={'index':'ADP Rank'})
                 .style
                 .format(precision=1, thousands='')
                 .format(format_dollar_amount,subset=['Val Diff'])
                 .background_gradient(axis=0, vmin=-50, vmax=50,
                                      cmap="vlag_r", subset=['% Diff'])
                 .background_gradient(axis=0, vmin=-7, vmax=7, 
                                      cmap="vlag", subset=['Val Diff']),
                 hide_index=True,
             width=750
             )
