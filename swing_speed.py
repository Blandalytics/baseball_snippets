import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np


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

swing_threshold = st.number_input(f'Min # of Pitches:',
                                  min_value=0, 
                                  max_value=swing_data.groupby('Hitter')['Swings'].sum().max(),
                                  step=25, 
                                  value=200)

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
             .rename(columns={'bat_speed':'Swing Speed (mph)',
                              'swing_length':'Swing Length (ft)',
                              'swing_acceleration':'Swing Acceleration (ft/s^2)'
                             })
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
player = st.selectbox('Choose a player:', players)

def speed_dist(player):
    fig, ax = plt.subplots(figsize=(6,3))
    sns.kdeplot(pitch_data['bat_speed'],
                linestyle='--',
                color='w',
                cut=0)
    sns.kdeplot(pitch_data.loc[pitch_data['hittername']==player,'bat_speed'],
                color=sns.color_palette('vlag',n_colors=1000)[-1],
               cut=0)
    ax.set(xlim=(50,90),
           xlabel='Bat Speed (mph)',
           ylabel='')
    plt.legend(labels=['MLB',player],
               loc='lower center')
    ax.set_yticks([])
    fig.suptitle(f"{player}'s\nBat Speed Distribution",y=1)
    sns.despine(left=True)
    st.pyplot(fig)
speed_dist(player)
