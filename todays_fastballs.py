import streamlit as st
import datetime
from datetime import timedelta
import matplotlib as mpl
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import requests
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py
import pytz
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression

def adjusted_vaa(dataframe):
    ## Physical characteristics of pitch
    # Pitch velocity (to plate) at plate
    dataframe['vyf'] = -1 * (dataframe['vy0']**2 - (2 * dataframe['ay']*(50-17/12)))**0.5
    # Pitch time in air (50ft to home plate)
    dataframe['pitch_time_50ft'] = (dataframe['vyf'] - dataframe['vy0'])/dataframe['ay']
    # Pitch velocity (vertical) at plate
    dataframe['vzf'] = dataframe['vz0'] + dataframe['az'] * dataframe['pitch_time_50ft']

    ## raw and height-adjusted VAA
    # Raw VAA 
    dataframe['raw_vaa'] = -1 * np.arctan(dataframe['vzf']/dataframe['vyf']) * (180/np.pi)
    # VAA of all pitches at that height
    dataframe['vaa_z_adj'] = np.where(dataframe['p_z']<3.5,
                                      dataframe['p_z'].mul(1.5635).add(-10.092),
                                      dataframe['p_z'].pow(2).mul(-0.1996).add(dataframe['p_z'].mul(2.704)).add(-11.69))
    dataframe['adj_vaa'] = dataframe['raw_vaa'].sub(dataframe['vaa_z_adj'])
    # Adjusted VAA, based on height
    return dataframe[['raw_vaa','adj_vaa']]

def spin_calcs(data):
    needed_cols = ['Velo','p_x','p_z','vx0','vy0',
                   'vz0','ax','ay','az','Ext']
    data[needed_cols] = data[needed_cols].astype('float')
    
    ## Formulas
    # Release location
    data['yR'] = 60.5 - data['Ext']
    
    # Time since release
    data['tR'] = (-data['vy0']-(data['vy0']**2 - 2*data['ay']*(50-data['yR']))**0.5)/data['ay']
    
    # Release velo
    data['vxR'] = data['vx0']+data['ax']*data['tR']
    data['vyR'] = data['vy0']+data['ay']*data['tR']
    data['vzR'] = data['vz0']+data['az']*data['tR']
    
    # Delta release speed
    data['dv0'] = data['Velo'] - (data['vxR']**2 + data['vyR']**2 + data['vzR']**2)**0.5/1.467

    # pitch flight time
    data['tf'] = (-data['vyR']-(data['vyR']**2 - 2*data['ay']*(data['yR']-17/12))**0.5)/data['ay']

    # Average velocity
    data['v_xbar'] = (2*data['vxR']+data['ax']*data['tf'])/2
    data['v_ybar'] = (2*data['vyR']+data['ay']*data['tf'])/2
    data['v_zbar'] = (2*data['vzR']+data['az']*data['tf'])/2
    data['v_bar'] = (data['v_xbar']**2 + data['v_ybar']**2 + data['v_zbar']**2)**0.5

    # Drag Acceleration
    data['a_drag'] = -(data['ax']*data['v_xbar'] + data['ay']*data['v_ybar'] + (data['az']+32.174)*data['v_zbar'])/data['v_bar']

    # Magnus Accelerations
    data['a_magx'] = data['ax'] + data['a_drag']*data['v_xbar']/data['v_bar']
    data['a_magy'] = data['ay'] + data['a_drag']*data['v_ybar']/data['v_bar']
    data['a_magz'] = data['az'] + data['a_drag']*data['v_zbar']/data['v_bar'] + 32.174
    data['a_mag'] = (data['a_magx']**2 + data['a_magy']**2 + data['a_magz']**2)**0.5

    data['IHB'] = 0.5*data['a_magx']*data['tf']**2*12
    data['IVB'] = 0.5*data['a_magz']*data['tf']**2*12
    
    return data[['IHB','IVB']]
                                                                   
### Standardized Strikezone (z-location, in 'strikezones')
def strikezone_z(dataframe,top_column,bottom_column):
    dataframe[['p_z',top_column,bottom_column]] = dataframe[['p_z',top_column,bottom_column]].astype('float')
    
    # Ratio of 'strikezones' above/below midpoint of strikezone
    dataframe['sz_mid'] = dataframe[[top_column,bottom_column]].mean(axis=1)
    dataframe['sz_height'] = dataframe[top_column].sub(dataframe[bottom_column])
    
    return dataframe['p_z'].sub(dataframe['sz_mid']).div(dataframe['sz_height'])

def loc_model(df,year=2024):
    df['balls_before_pitch'] = np.clip(df['balls'], 0, 3)
    df['strikes_before_pitch'] = np.clip(df['strikes'], 0, 2)
    df['pitcherside'] = df['P Hand'].copy()

    df = pd.get_dummies(df, columns=['pitcherside','hitterside','balls_before_pitch','strikes_before_pitch'])
    for hand in ['L','R']:
        if f'pitcherside_{hand}' not in df.columns.values:
            df[f'pitcherside_{hand}'] = 0

    df[['take_input','swing_input','called_strike_raw','ball_raw',
                'hit_by_pitch_raw','swinging_strike_raw','contact_raw',
                'foul_strike_raw','in_play_raw','10deg_raw','10-20deg_raw',
                '20-30deg_raw','30-40deg_raw','40-50deg_raw','50+deg_raw',
                'called_strike_pred','ball_pred','hit_by_pitch_pred','contact_input',
                'swinging_strike_pred','foul_strike_pred','in_play_input','50+deg_pred',
                'out_pred', 'single_pred', 'double_pred', 'triple_pred', 'home_run_pred']] = None

    for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
        df[[launch_angle+'_input',launch_angle+': <90mph_raw',
                 launch_angle+': 90-95mph_raw',launch_angle+': 95-100mph_raw',
                 launch_angle+': 100-105mph_raw',launch_angle+': 105+mph_raw',
                 launch_angle+': <90mph_pred',launch_angle+': 90-95mph_pred',
                 launch_angle+': 95-100mph_pred',launch_angle+': 100-105mph_pred',
                 launch_angle+': 105+mph_pred']] = None

    # Swing Decision
    with open('model_files/2024_pl_swing_model_Fastball_loc.pkl', 'rb') as f:
        decision_model = pickle.load(f)

    df[['take_input','swing_input']] = decision_model.predict_proba(df[decision_model.feature_names_in_])

    # Take Result
    with open('model_files/2024_pl_take_model_Fastball_loc.pkl', 'rb') as f:
        take_model = pickle.load(f)

    df[['called_strike_raw','ball_raw','hit_by_pitch_raw']] = take_model.predict_proba(df[take_model.feature_names_in_])
    df['called_strike_pred'] = df['called_strike_raw'].mul(df['take_input'])
    df['ball_pred'] = df['ball_raw'].mul(df['take_input'])
    df['hit_by_pitch_pred'] = df['hit_by_pitch_raw'].mul(df['take_input'])

    # Swing Result
    with open('model_files/2024_pl_contact_model_Fastball_loc.pkl', 'rb') as f:
        swing_result_model = pickle.load(f)

    df[['swinging_strike_raw','contact_raw']] = swing_result_model.predict_proba(df[swing_result_model.feature_names_in_])
    df['contact_input'] = df['contact_raw'].mul(df['swing_input'])
    df['swinging_strike_pred'] = df['swinging_strike_raw'].mul(df['swing_input'])

    # Contact Result
    with open('model_files/2024_pl_in_play_model_Fastball_loc.pkl', 'rb') as f:
        contact_model = pickle.load(f)

    df[['foul_strike_raw','in_play_raw']] = contact_model.predict_proba(df[contact_model.feature_names_in_])
    df['foul_strike_pred'] = df['foul_strike_raw'].mul(df['contact_input'])
    df['in_play_input'] = df['in_play_raw'].mul(df['contact_input'])

    # Launch Angle Result
    with open('model_files/2024_pl_launch_angle_model_Fastball_loc.pkl', 'rb') as f:
        launch_angle_model = pickle.load(f)

    df[['10deg_raw','10-20deg_raw','20-30deg_raw','30-40deg_raw','40-50deg_raw','50+deg_raw']] = launch_angle_model.predict_proba(df[launch_angle_model.feature_names_in_])
    for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
        df[launch_angle+'_input'] = df[launch_angle+'_raw'].mul(df['in_play_input'])
    df['50+deg_pred'] = df['50+deg_raw'].mul(df['in_play_input'])

    # Launch Velo Result
    for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
        with open('model_files/2024_pl_{}_model_Fastball_loc.pkl'.format(launch_angle), 'rb') as f:
            launch_velo_model = pickle.load(f)

        df[[launch_angle+': <90mph_raw',launch_angle+': 90-95mph_raw',launch_angle+': 95-100mph_raw',launch_angle+': 100-105mph_raw',launch_angle+': 105+mph_raw']] = launch_velo_model.predict_proba(df[launch_velo_model.feature_names_in_])
        for bucket in [launch_angle+': '+x for x in ['<90mph','90-95mph','95-100mph','100-105mph','105+mph']]:
            df[bucket+'_pred'] = df[bucket+'_raw'].mul(df[launch_angle+'_input'])

    bip_result_dict = (
        pd.read_csv('model_files/data_bip_result.csv')
        .set_index(['year_played','bb_bucket'])
        .to_dict(orient='index')
    )

    # Apply averages to each predicted grouping
    for outcome in ['out', 'single', 'double', 'triple', 'home_run']:
        # Start with 50+ degrees (popups)
        df[outcome+'_pred'] = df['50+deg_pred']*bip_result_dict[(year,'50+deg')][outcome]

        for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
            for bucket in [launch_angle+': '+x for x in ['<90mph','90-95mph','95-100mph','100-105mph','105+mph']]:
                df[outcome+'_pred'] += df[bucket+'_pred']*bip_result_dict[(year,bucket)][outcome]

    ### Find the estimated change in wOBA/runs for each pitch
    # wOBA value of an outcome, based on the count that it came in
    outcome_wOBAs = pd.read_csv('model_files/data_woba_outcome.csv').set_index(['year_played','balls','strikes'])

    df = df.merge(outcome_wOBAs,
                  how='left',
                  on=['year_played','balls','strikes'])

    # wOBA_effect is how the pitch is expected to affect wOBA
    # (either by moving the count, or by ending the PA)
    df['wOBA_effect'] = 0

    for stat in [x[:-5] for x in list(outcome_wOBAs.columns)]:
        df['wOBA_effect'] = df['wOBA_effect'].add(df[stat+'_pred'].fillna(df[stat+'_pred'].median()).mul(df[stat+'_wOBA'].fillna(df[stat+'_wOBA'].median())))

    return df['wOBA_effect'].sub(-0.004253050593194383).div(0.05179234832326223).mul(-50).add(100)

today = (datetime.datetime.now(pytz.utc)-timedelta(hours=16)).date()
st.header(f"4-Seam Fastballs by Starters")
col1, col2, col3 = st.columns([1/3,1/3,1/3])
with col1:
    level = st.selectbox('Choose a level:', ['MLB','AAA','A (FSL)','AFL','NCAA'])
    level_dict = {'MLB':1,'AAA':11,'A (FSL)':14, 'AFL':17, 'NCAA':'22'}
    level_code = level_dict[level]
with col2:
    date = st.date_input("Select a game date:", today, min_value=datetime.date(2024, 3, 28), max_value=today)

@st.cache_data(ttl=90,show_spinner=f"Loading data")
def load_savant(date,level):
    r = requests.get(f'https://statsapi.mlb.com/api/v1/schedule?sportId={level}&date={date}')
    x = r.json()
    if len(x['dates'])==0:
        st.write('No games today')
        st.stop()
    
    game_list = []
    for game in range(len(x['dates'][0]['games'])):
        game_list += [x['dates'][0]['games'][game]['gamePk']]
    
    game_date = []
    pitcher_id_list = []
    pitcher_name = []
    throws = []
    stands = []
    pitch_id = []
    pitch_type = []
    inning = []
    out = []
    balls = []
    strikes = []
    velo = []
    extension = []
    ivb = []
    vx0 = []
    vy0 = []
    vz0 = []
    ax = []
    ay = []
    az = []
    px = []
    pz = []
    sz_top = []
    sz_bot = []
    for game_id in game_list:
        r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_id}')
        x = r.json()
        # if x['game_status_code'] in ['P','S']:
        #     continue
        for home_away_pitcher in ['home','away']:
            if f'{home_away_pitcher}_pitchers' not in x.keys():
                continue
            for pitcher_id in list(x[f'{home_away_pitcher}_pitchers'].keys()):
                for pitch in range(len(x[f'{home_away_pitcher}_pitchers'][pitcher_id])):
                    game_date += [x['gameDate']]
                    pitcher_id_list += [pitcher_id]
                    p_name = x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitcher_name']
                    pitcher_name += [p_name]
                    throws += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['p_throws']]
                    stands += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['stand']]
                    inning += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['inning']]
                    out += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['outs']]
                    balls += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['balls']]
                    strikes += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['strikes']]
                    pitch_id += [pitch]
                    try:
                        velo += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['start_speed']]
                        pitch_type += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitch_type']]
                        ivb += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['inducedBreakZ']]
                        extension += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['extension'] if 'extension' in x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch].keys() else None]
                        vx0 += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['vx0']]
                        vy0 += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['vy0']]
                        vz0 += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['vz0']]
                        ax += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['ax']]
                        ay += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['ay']]
                        az += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['az']]
                        px += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['px']]
                        pz += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pz']]
                        sz_top += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['sz_top']]
                        sz_bot += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['sz_bot']]
                    except KeyError:
                        pitch_type += ['UN']
                        velo += [None]
                        ivb += [None]
                        extension += [None]
                        vx0 += [None]
                        vy0 += [None]
                        vz0 += [None]
                        ax += [None]
                        ay += [None]
                        az += [None]
                        px += [None]
                        pz += [None]
                        sz_top += [None]
                        sz_bot += [None]
    
    player_df = pd.DataFrame()
    player_df['game_date'] = game_date
    player_df['year_played'] = 2024
    player_df['MLBAMID'] = pitcher_id_list
    player_df['Pitcher'] = pitcher_name
    player_df['P Hand'] = throws
    player_df['hitterside'] = stands
    player_df['Inning'] = inning
    player_df['Out'] = out
    player_df['balls'] = balls
    player_df['strikes'] = strikes
    player_df['#'] = pitch_id
    player_df['pitch_type'] = pitch_type
    player_df['Velo'] = velo
    player_df['Ext'] = extension
    player_df['IVB_sav'] = ivb
    player_df['vx0'] = vx0
    player_df['vy0'] = vy0
    player_df['vz0'] = vz0
    player_df['ax'] = ax
    player_df['ay'] = ay
    player_df['az'] = az
    player_df['p_x'] = px
    player_df['p_z'] = pz
    player_df['sz_top'] = sz_top
    player_df['sz_bot'] = sz_bot
    player_df[['IHB','IVB']] = spin_calcs(player_df)
    player_df['IHB'] = np.where(player_df['P Hand']=='R',player_df['IHB'].mul(-1),player_df['IHB'])
    player_df[['VAA','HAVAA']] = adjusted_vaa(player_df)
    player_df['sz_z'] = strikezone_z(player_df,'sz_top','sz_bot')
    if player_df.shape[0]==0:
        player_df['plvLoc+'] = None
    else:
        player_df['plvLoc+'] = loc_model(player_df)
  
    return player_df.loc[(player_df['pitch_type']=='FF')] if level == 17 else player_df.loc[(player_df['pitch_type']=='FF') & (player_df['Inning'].groupby(player_df['MLBAMID']).transform('min')==1)]

chart_df = load_savant(date,level_code)

if chart_df.shape[0]==0:
    st.write('No fastballs thrown')
    st.stop()

st.write('**Fan 4+**: modeled Whiff% of a pitch (based on the "Fan-Tastic 4" stats: Velo, Extension, IVB, and HAVAA), compared to an average 4-Seam Fastball')

model_df = (chart_df
            .groupby(['Pitcher'])
            [['#','Velo','Ext','IVB','HAVAA','IHB','VAA','plvLoc+']]
            .agg({
                '#':'count',
                'Velo':'mean',
                'Ext':'mean',
                'IVB':'mean',
                'HAVAA':'mean',
                'IHB':'mean',
                'VAA':'mean',
                'plvLoc+':'mean'
              })
            .sort_values('#',ascending=False)
           )

with open('model_files/fan-4_contact_model.pkl', 'rb') as f:
    whiff_model = pickle.load(f)

model_df = model_df.dropna(subset=whiff_model.feature_names_in_)
model_df['swinging_strike_pred'] = whiff_model.predict(model_df[whiff_model.feature_names_in_])
model_df['Fan 4+'] = model_df['swinging_strike_pred'].div(0.1542).mul(100).astype('int')

st.dataframe(model_df[['#','Velo','Ext','IVB','HAVAA','IHB','VAA','Fan 4+','plvLoc+']].sort_values('Fan 4+',ascending=False)
             .style
             .format(precision=1, thousands=',')
             .background_gradient(axis=0, vmin=91.4, vmax=96.6, cmap="vlag", subset=['Velo'])
             .background_gradient(axis=0, vmin=5.95, vmax=6.95, cmap="vlag", subset=['Ext'])
             .background_gradient(axis=0, vmin=12.4, vmax=18.2, cmap="vlag", subset=['IVB'])
             .background_gradient(axis=0, vmin=0.4, vmax=1.5, cmap="vlag", subset=['HAVAA'])
             .background_gradient(axis=0, vmin=70, vmax=130, cmap="vlag", subset=['Fan 4+'])
             .background_gradient(axis=0, vmin=85, vmax=115, cmap="vlag", subset=['plvLoc+']),
            column_config={"Pitcher": st.column_config.Column(width="medium")}
)

players = list(model_df.sort_values('plvLoc+',ascending=False).index)
player = st.selectbox('Choose a starter:', players)

def location_chart(df,player):
    chart_df = df.loc[(df['Pitcher']==player)].copy()
    chart_df['smoothed_csw'] = 0.288
    chart_df['smoothed_wOBAcon'] = 0.3284

    plate_y = -.25
    
    layout = go.Layout(height = 600,width = 500,xaxis_range=[-2.5,2.5], yaxis_range=[-1,6])

    labels = chart_df['plvLoc+']
    hover_text = '<b>plvLoc+: %{marker.color:.1f}</b><br>Count: %{customdata[0]}-%{customdata[1]}<br>Hitter Hand: %{text}<br>X Loc: %{x:.1f}ft<br>Y Loc: %{y:.1f}ft<extra></extra>'
    marker_dict = dict(color = labels, size= 15, line=dict(width=0),
                               cmin=50,cmax=150,
                               colorscale=[[x/100,'rgb'+str(tuple([int(y*255) for y in sns.color_palette('vlag',n_colors=101)[x]]))] for x in range(101)], 
                               colorbar=dict(
                                   title="plvLocation+\n",
                                   # titleside="top",
                                   tickmode="array",
                                   tickvals=[50, 75, 100, 125, 150],
                                   ticks="outside"
                                   ))
    
    fig = go.Figure(layout = layout)
    fig.add_trace(go.Scatter(x=[10/12,10/12], y=[1.5,3.5],
                             mode='lines',
                             line=dict(color='black', width=4),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[-10/12,-10/12], y=[1.5,3.5],
                             mode='lines',
                             line=dict(color='black', width=4),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[-10/12,10/12], y=[1.5,1.5],
                             mode='lines',
                             line=dict(color='black', width=4),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[-10/12,10/12], y=[3.5,3.5],
                             mode='lines',
                             line=dict(color='black', width=4),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[10/36,10/36], y=[1.5,3.5],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[-10/36,-10/36], y=[1.5,3.5],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[-10/12,10/12], y=[1.5+2/3,1.5+2/3],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[-10/12,10/12], y=[3.5-2/3,3.5-2/3],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    
    # Plate
    fig.add_trace(go.Scatter(x=[-8.5/12,8.5/12], y=[plate_y,plate_y],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[-8.5/12,-8.25/12], y=[plate_y,plate_y+0.15],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[8.5/12,8.25/12], y=[plate_y,plate_y+0.15],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[8.28/12,0], y=[plate_y+0.15,plate_y+0.25],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[-8.28/12,0], y=[plate_y+0.15,plate_y+0.25],
                             mode='lines',
                             line=dict(color='black', width=2),
                             showlegend=False))

    bonus_text = chart_df['hitterside']
    fig.add_trace(go.Scatter(x=chart_df['p_x'].mul(-1), y=chart_df['p_z'], mode='markers', 
                       marker=marker_dict, text=bonus_text,
                       customdata=chart_df[['balls','strikes']],
                       hovertemplate=hover_text,
                        showlegend=False))
    
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_yaxes(visible=False, showticklabels=False)
    
    overall_loc = chart_df['plvLoc+'].mean()
    fig.update_layout(
            margin=dict(
                l=200,
                r=200,
                b=100,
                t=100,
                pad=4
                ),
            template='simple_white',
            title={
                'text': f"{player}'s Four-Seam<br>plvLocation+: {overall_loc:.1f}",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            legend={
                "x": 0.8,
                "y": 0.67}
        )
    fig.show()
    st.plotly_chart(fig,
                    theme=None
                   )
location_chart(chart_df, player)
