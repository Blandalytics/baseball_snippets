import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

st.title('MLB Swing Speed App')
st.write('Find me [@Blandalytics](https://twitter.com/blandalytics)')

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

swing_data = pd.read_csv('https://github.com/Blandalytics/baseball_snippets/blob/main/swing_speed_data.csv?raw=true',encoding='latin1')

swing_threshold = st.number_input(f'Min # of Swings:',
                                  min_value=0, 
                                  max_value=swing_data.groupby('Hitter')['Swings'].sum().max(),
                                  step=25, 
                                  value=100)

stat_name_dict = {
    'bat_speed':'Swing Speed (mph)',
    'swing_length':'Swing Length (ft)',
    'swing_acceleration':'Swing Acceleration (ft/s^2)'
}

st.dataframe(swing_data
             .groupby(['Hitter'])
             [['Swings','bat_speed','swing_length','swing_acceleration']]
             .agg({
                 'Swings':'count',
                 'bat_speed':'mean',
                 'swing_length':'mean',
                 'swing_acceleration':'mean'
             })
             .query(f'Swings >={swing_threshold}')
             .sort_values('swing_acceleration',ascending=False)
             .rename(columns=stat_name_dict)
             .round(1)
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
               .groupby('Hitter')
               [['Swings',stat]]
               .agg({'Swings':'count',stat:'mean'})
               .query(f'Swings >={swing_threshold}')
               .reset_index()
               .sort_values(stat, ascending=False)
               ['Hitter']
              )
    
def speed_dist(player,stat):
    fig, ax = plt.subplots(figsize=(6,3))
    g = sns.kdeplot(swing_data[stat],
                linestyle='--',
                color='w',
                cut=0)

    val = swing_data.loc[swing_data['Hitter']==player,stat].mean()
    player_color = sns.color_palette('vlag',n_colors=len(players))[len(players)-players.index(player)-1]
    p = sns.kdeplot(swing_data.loc[swing_data['Hitter']==player,stat],
                    color=player_color,
                    cut=0)
    
    ax.set(xlim=(swing_data[stat].quantile(0.03),swing_data[stat].max()),
           xlabel=stat_name_dict[stat],
           # ylim=(0,ax.get_ylim()[1]*1),
           ylabel='')

    plt.legend(labels=['MLB',player],
               loc='lower center')
    
    kdeline_p = p.lines[1]
    xs_p = kdeline_p.get_xdata()
    ys_p = kdeline_p.get_ydata()
    height_p = np.interp(val, xs_p, ys_p)
    ax.vlines(val, 0, height_p, color=player_color)
    
    kdeline = g.lines[0]
    xs = kdeline.get_xdata()
    ys = kdeline.get_ydata()
    height = np.interp(swing_data[stat].median(), xs, ys)
    ax.vlines(swing_data[stat].median(), 0, height, color='w', ls='--')
    
    ax.set_yticks([])
    title_stat = ' '.join(stat_name_dict[stat].split(' ')[:-1])
    fig.suptitle(f"{player}'s\n{title_stat} Distribution",y=1)
    sns.despine(left=True)
    fig.text(0.83,-0.15,'@blandalytics',ha='center',fontsize=10)
    fig.text(0.125,-0.15,'mlb-swing-speed.streamlit.app',ha='left',fontsize=10)
    st.pyplot(fig)
speed_dist(player,stat)
