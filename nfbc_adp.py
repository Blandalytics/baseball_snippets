import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import urllib
import datetime
from PIL import Image
from matplotlib import ticker
import matplotlib.dates as mdates
from matplotlib import colors

logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
logo = Image.open(urllib.request.urlopen(logo_loc))
st.image(logo, width=200)

st.title("NFBC Draft Data, over Time")

@st.cache_data(ttl=6000,show_spinner=f"Loading draft data")
def load_data():
  df = pd.read_parquet('https://github.com/Blandalytics/baseball_snippets/blob/main/nfbc_adp_data.parquet?raw=true')
  df['start_date'] = pd.to_datetime(df['start_date']).dt.date
  df['end_date'] = pd.to_datetime(df['end_date']).dt.date
  return df

nfbc_adp_df = load_data()
update_date = nfbc_adp_df['end_date'].max().strftime('%-m/%-d/%y')
st.write(f'Data is through {update_date}')
## Set Styling
# Plot Style
pl_white = '#FEFEFE'
pl_background = '#162B50'
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
     }
    )

chart_red = sns.color_palette('vlag',n_colors=10000)[-1]
chart_blue = sns.color_palette('vlag',n_colors=10000)[0]

def format_dollar_amount(amount):
    formatted_absolute_amount = '${:.1f}'.format(abs(amount))
    if round(amount, 2) < 0:
        return f'-{formatted_absolute_amount}'
    return formatted_absolute_amount

col1, col2 = st.columns(2)
with col1:
    adp_start_date = st.date_input("ADP Start Date", 
                                   datetime.date(2025,10,1),
                                   min_value=datetime.date(2024,10,20),
                                   max_value=datetime.date.today() - datetime.timedelta(days=7),
                                   format="MM/DD/YYYY")
    adp_thresh = 1000
    start_string = adp_start_date.strftime('%-m/%-d')
with col2:
    pos_filters = [
        'All',
        'H','P',
        'C','1B','2B','SS','3B','OF','DH',
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

if pos_filter in ['All','H','OF']:
    player_index = player_list.index('Ben Rice')
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
               color='w')
  sns.lineplot(chart_df,
               x='end_date',
               y='Min Pick',color=chart_red,
              )
  sns.lineplot(chart_df,
               x='end_date',
               y='Max Pick',color=chart_blue,
              )
  
  ax.fill_between(chart_df['end_date'], 
                  chart_df['ADP'] - chart_df['StDev Est'], 
                  chart_df['ADP'] + chart_df['StDev Est'], 
                  color='w',
                  alpha=0.2,
                 )
  
  adp_val = chart_df.iloc[-1]['ADP']
  ax.text(chart_df.iloc[-1]['end_date'] + datetime.timedelta(days=2),
          adp_val,
          f'ADP: {adp_val:.1f}',
           ha='left',va='center')
  
  min_val = chart_df.iloc[-1]['Min Pick']
  if min_val <= adp_val*0.95:
    ax.text(chart_df.iloc[-1]['end_date'] + datetime.timedelta(days=2),
            min_val,
            f'Min Pick: {min_val:.0f}',
             ha='left',va='center', color=chart_red)
  
  max_val = chart_df.iloc[-1]['Max Pick']
  if max_val >= adp_val*1.1:
    ax.text(chart_df.iloc[-1]['end_date'] + datetime.timedelta(days=2),
            max_val,
            f'Max Pick: {max_val:.0f}',
             ha='left',va='center',color=chart_blue)
    
  if chart_df.iloc[-1]['ADP'] + chart_df.iloc[-1]['StDev Est'] >= adp_val*1.05:
    ax.text(chart_df.iloc[-1]['end_date'] + datetime.timedelta(days=2),
            chart_df.iloc[-1]['ADP'] + chart_df.iloc[-1]['StDev Est'],
            'St Dev (est)',
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
  
  fig.suptitle(f"{player}'s NFBC Draft Data\nRolling 14-Days: {chart_start} - {chart_end}",x=0.55,y=1)
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
