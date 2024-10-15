import streamlit as st
import datetime
from datetime import timedelta
import matplotlib as mpl
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import urllib
import requests
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.offline as py
from PIL import Image
# import pytz
import xgboost as xgb
# import pkg_resources
from xgboost import XGBClassifier, XGBRegressor

st.set_page_config(page_title='Live Pitcher List Feed', page_icon='⚾')

logo_loc = 'https://github.com/Blandalytics/PLV_viz/blob/main/data/PL-text-wht.png?raw=true'
logo = Image.open(urllib.request.urlopen(logo_loc))
st.image(logo, width=200)

st.title('Live Pitcher List Feed')

bip_result_dict = {'10-20deg: 100-105mph': {'out': 0.34884200398598036,
  'single': 0.35708885987217376,
  'double': 0.2679540925022335,
  'triple': 0.021304377705999588,
  'home_run': 0.00481066593361281},
 '10-20deg: 105+mph': {'out': 0.25057373565660856,
  'single': 0.2883977900552486,
  'double': 0.3817254568635784,
  'triple': 0.021419464513387165,
  'home_run': 0.05788355291117722},
 '10-20deg: 90-95mph': {'out': 0.37410824693236944,
  'single': 0.4976695519832588,
  'double': 0.11490535527442214,
  'triple': 0.011890040901740702,
  'home_run': 0.0014268049082088842},
 '10-20deg: 95-100mph': {'out': 0.37082113695886615,
  'single': 0.4355261130796487,
  'double': 0.17632105992913263,
  'triple': 0.01694654136496688,
  'home_run': 0.00038514866738561084},
 '10-20deg: <90mph': {'out': 0.3501221655450298,
  'single': 0.5780787860602683,
  'double': 0.0677268635603755,
  'triple': 0.00402931973080715,
  'home_run': 4.2865103519225e-05},
 '10deg: 100-105mph': {'out': 0.5587734845799361,
  'single': 0.38957816377171217,
  'double': 0.048670684154555124,
  'triple': 0.002977667493796526,
  'home_run': 0.0},
 '10deg: 105+mph': {'out': 0.4712564543889845,
  'single': 0.4567986230636833,
  'double': 0.06838783706253586,
  'triple': 0.0035570854847963283,
  'home_run': 0.0},
 '10deg: 90-95mph': {'out': 0.7267213427427046,
  'single': 0.2403585733358764,
  'double': 0.030783902345985124,
  'triple': 0.002136181575433912,
  'home_run': 0.0},
 '10deg: 95-100mph': {'out': 0.6433748435132107,
  'single': 0.31603742505106414,
  'double': 0.03772155234894907,
  'triple': 0.002866179086776043,
  'home_run': 0.0},
 '10deg: <90mph': {'out': 0.8408481743999047,
  'single': 0.14585939206226298,
  'double': 0.012746441122163322,
  'triple': 0.0005459924156689896,
  'home_run': 0.0},
 '20-30deg: 100-105mph': {'out': 0.3043250784792466,
  'single': 0.013951866062085804,
  'double': 0.22575863271712593,
  'triple': 0.030258109522148587,
  'home_run': 0.4257063132193931},
 '20-30deg: 105+mph': {'out': 0.05316757042719217,
  'single': 0.01203544504695146,
  'double': 0.114138341489221,
  'triple': 0.009654807565136887,
  'home_run': 0.8110038354714985},
 '20-30deg: 90-95mph': {'out': 0.8495277385734448,
  'single': 0.01975898382368907,
  'double': 0.10520030398436651,
  'triple': 0.010422321137770058,
  'home_run': 0.015090652480729563},
 '20-30deg: 95-100mph': {'out': 0.6488510638297872,
  'single': 0.013617021276595745,
  'double': 0.19727659574468084,
  'triple': 0.02102127659574468,
  'home_run': 0.11923404255319149},
 '20-30deg: <90mph': {'out': 0.5922371406959153,
  'single': 0.34010968229954613,
  'double': 0.06202723146747353,
  'triple': 0.00548411497730711,
  'home_run': 0.00014183055975794251},
 '30-40deg: 100-105mph': {'out': 0.4961502925777641,
  'single': 0.001539882968894364,
  'double': 0.030951647674776716,
  'triple': 0.012319063751154912,
  'home_run': 0.4590391130274099},
 '30-40deg: 105+mph': {'out': 0.16513409961685824,
  'single': 0.0019157088122605363,
  'double': 0.013793103448275862,
  'triple': 0.0030651340996168583,
  'home_run': 0.8160919540229885},
 '30-40deg: 90-95mph': {'out': 0.9368251210127839,
  'single': 0.0013652724339084025,
  'double': 0.024699019486161104,
  'triple': 0.004840511356584336,
  'home_run': 0.03227007571056224},
 '30-40deg: 95-100mph': {'out': 0.7922282120395328,
  'single': 0.0017969451931716084,
  'double': 0.0330188679245283,
  'triple': 0.0110062893081761,
  'home_run': 0.1619496855345912},
 '30-40deg: <90mph': {'out': 0.8379536556978098,
  'single': 0.1317320918421331,
  'double': 0.02766902973230346,
  'triple': 0.0020103692730927946,
  'home_run': 0.0006348534546608824},
 '40-50deg: 100-105mph': {'out': 0.8959088555152771,
  'single': 0.0015535991714137752,
  'double': 0.01605385810460901,
  'triple': 0.004142931123770067,
  'home_run': 0.08234075608493009},
 '40-50deg: 105+mph': {'out': 0.619277108433735,
  'single': 0.00963855421686747,
  'double': 0.014457831325301205,
  'triple': 0.004819277108433735,
  'home_run': 0.35180722891566263},
 '40-50deg: 90-95mph': {'out': 0.9927727833597744,
  'single': 0.0012339150361360832,
  'double': 0.0033491979552265115,
  'triple': 0.0012339150361360832,
  'home_run': 0.0014101886127269522},
 '40-50deg: 95-100mph': {'out': 0.9700921969867327,
  'single': 0.0008994827973914999,
  'double': 0.008994827973915,
  'triple': 0.0029233190915223745,
  'home_run': 0.0170901731504385},
 '40-50deg: <90mph': {'out': 0.9217500495671139,
  'single': 0.05954662613178243,
  'double': 0.018174608419800408,
  'triple': 0.0004626263961403741,
  'home_run': 6.608948516291058e-05},
 '50+deg': {'out': 0.9848613269294156,
  'single': 0.009700051796393088,
  'double': 0.0052267269388331684,
  'triple': 0.00018835052031831239,
  'home_run': 2.3543815039789048e-05}}

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
    dataframe['vaa_z_adj'] = np.where(dataframe['pz']<3.5,
                                      dataframe['pz'].mul(1.5635).add(-10.092),
                                      dataframe['pz'].pow(2).mul(-0.1996).add(dataframe['pz'].mul(2.704)).add(-11.69))
    dataframe['adj_vaa'] = dataframe['raw_vaa'].sub(dataframe['vaa_z_adj'])
    # Adjusted VAA, based on height
    return dataframe[['raw_vaa','adj_vaa']]

def spin_calcs(data):
    needed_cols = ['velo','px','pz','vx0','vy0',
                   'vz0','ax','ay','az','extension']
    data[needed_cols] = data[needed_cols].astype('float')
    
    ## Formulas
    # Release location
    data['yR'] = 60.5 - data['extension']
    
    # Time since release
    data['tR'] = (-data['vy0']-(data['vy0']**2 - 2*data['ay']*(50-data['yR']))**0.5)/data['ay']
    
    # Release velo
    data['vxR'] = data['vx0']+data['ax']*data['tR']
    data['vyR'] = data['vy0']+data['ay']*data['tR']
    data['vzR'] = data['vz0']+data['az']*data['tR']
    
    # Delta release speed
    data['dv0'] = data['velo'] - (data['vxR']**2 + data['vyR']**2 + data['vzR']**2)**0.5/1.467

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
    data['total_IB'] = (data['IHB'].astype('float')**2+data['IVB'].astype('float')**2)**0.5
    
    return data[['IHB','IVB','total_IB']]
                                                                   
### Standardized Strikezone (z-location, in 'strikezones')
def strikezone_z(dataframe,top_column,bottom_column):
    dataframe[['p_z',top_column,bottom_column]] = dataframe[['p_z',top_column,bottom_column]].astype('float')
    
    # Ratio of 'strikezones' above/below midpoint of strikezone
    dataframe['sz_mid'] = dataframe[[top_column,bottom_column]].mean(axis=1)
    dataframe['sz_height'] = dataframe[top_column].sub(dataframe[bottom_column])
    
    return dataframe['p_z'].sub(dataframe['sz_mid']).div(dataframe['sz_height'])

# def loc_model(df,year=2024):
#     df['balls_before_pitch'] = np.clip(df['balls'], 0, 3)
#     df['strikes_before_pitch'] = np.clip(df['strikes'], 0, 2)
#     df['pitcherside'] = df['P Hand'].copy()

#     df = pd.get_dummies(df, columns=['pitcherside','hitterside','balls_before_pitch','strikes_before_pitch'])
#     for hand in ['L','R']:
#         if f'pitcherside_{hand}' not in df.columns.values:
#             df[f'pitcherside_{hand}'] = 0

#     df[['take_input','swing_input','called_strike_raw','ball_raw',
#                 'hit_by_pitch_raw','swinging_strike_raw','contact_raw',
#                 'foul_strike_raw','in_play_raw','10deg_raw','10-20deg_raw',
#                 '20-30deg_raw','30-40deg_raw','40-50deg_raw','50+deg_raw',
#                 'called_strike_pred','ball_pred','hit_by_pitch_pred','contact_input',
#                 'swinging_strike_pred','foul_strike_pred','in_play_input','50+deg_pred',
#                 'out_pred', 'single_pred', 'double_pred', 'triple_pred', 'home_run_pred']] = None

#     for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
#         df[[launch_angle+'_input',launch_angle+': <90mph_raw',
#                  launch_angle+': 90-95mph_raw',launch_angle+': 95-100mph_raw',
#                  launch_angle+': 100-105mph_raw',launch_angle+': 105+mph_raw',
#                  launch_angle+': <90mph_pred',launch_angle+': 90-95mph_pred',
#                  launch_angle+': 95-100mph_pred',launch_angle+': 100-105mph_pred',
#                  launch_angle+': 105+mph_pred']] = None

#     for pitch_type in ['Fastball','Breaking Ball','Offspeed']:
#         # Swing Decision
#         with open('2024_pl_swing_model_{}_loc.pkl'.format(pitch_type), 'rb') as f:
#             decision_model = pickle.load(f)
    
#         df.loc[df['pitch_type_bucket']==pitch_type,['take_input','swing_input']] = decision_model.predict_proba(df.loc[df['pitch_type_bucket']==pitch_type,decision_model.feature_names_in_])
    
#         # Take Result
#         with open('2024_pl_take_model_{}_loc.pkl'.format(pitch_type), 'rb') as f:
#             take_model = pickle.load(f)
    
#         df.loc[df['pitch_type_bucket']==pitch_type,['called_strike_raw','ball_raw','hit_by_pitch_raw']] = take_model.predict_proba(df.loc[df['pitch_type_bucket']==pitch_type,take_model.feature_names_in_])
#         df.loc[df['pitch_type_bucket']==pitch_type,'called_strike_pred_loc'] = df.loc[df['pitch_type_bucket']==pitch_type,'called_strike_raw'].mul(df.loc[df['pitch_type_bucket']==pitch_type,'take_input'])
#         df.loc[df['pitch_type_bucket']==pitch_type,'ball_pred_loc'] = df.loc[df['pitch_type_bucket']==pitch_type,'ball_raw'].mul(df.loc[df['pitch_type_bucket']==pitch_type,'take_input'])
#         df.loc[df['pitch_type_bucket']==pitch_type,'hit_by_pitch_pred_loc'] = df.loc[df['pitch_type_bucket']==pitch_type,'hit_by_pitch_raw'].mul(df.loc[df['pitch_type_bucket']==pitch_type,'take_input'])
    
#         # Swing Result
#         with open('2024_pl_contact_model_{}_loc.pkl'.format(pitch_type), 'rb') as f:
#             swing_result_model = pickle.load(f)
    
#         df.loc[df['pitch_type_bucket']==pitch_type,['swinging_strike_raw','contact_raw']] = swing_result_model.predict_proba(df.loc[df['pitch_type_bucket']==pitch_type,swing_result_model.feature_names_in_])
#         df.loc[df['pitch_type_bucket']==pitch_type,'contact_input'] = df.loc[df['pitch_type_bucket']==pitch_type,'contact_raw'].mul(df.loc[df['pitch_type_bucket']==pitch_type,'swing_input'])
#         df.loc[df['pitch_type_bucket']==pitch_type,'swinging_strike_pred_loc'] = df.loc[df['pitch_type_bucket']==pitch_type,'swinging_strike_raw'].mul(df.loc[df['pitch_type_bucket']==pitch_type,'swing_input'])
    
#         # Contact Result
#         with open('2024_pl_in_play_model_{}_loc.pkl'.format(pitch_type), 'rb') as f:
#             contact_model = pickle.load(f)
    
#         df.loc[df['pitch_type_bucket']==pitch_type,['foul_strike_raw','in_play_raw']] = contact_model.predict_proba(df.loc[df['pitch_type_bucket']==pitch_type,contact_model.feature_names_in_])
#         df.loc[df['pitch_type_bucket']==pitch_type,'foul_strike_pred_loc'] = df.loc[df['pitch_type_bucket']==pitch_type,'foul_strike_raw'].mul(df.loc[df['pitch_type_bucket']==pitch_type,'contact_input'])
#         df.loc[df['pitch_type_bucket']==pitch_type,'in_play_input'] = df.loc[df['pitch_type_bucket']==pitch_type,'in_play_raw'].mul(df.loc[df['pitch_type_bucket']==pitch_type,'contact_input'])
    
#         # Launch Angle Result
#         with open('2024_pl_launch_angle_model_{}_loc.pkl'.format(pitch_type), 'rb') as f:
#             launch_angle_model = pickle.load(f)
    
#         df.loc[df['pitch_type_bucket']==pitch_type,['10deg_raw','10-20deg_raw','20-30deg_raw','30-40deg_raw','40-50deg_raw','50+deg_raw']] = launch_angle_model.predict_proba(df.loc[df['pitch_type_bucket']==pitch_type,launch_angle_model.feature_names_in_])
#         for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
#             df.loc[df['pitch_type_bucket']==pitch_type,launch_angle+'_input'] = df.loc[df['pitch_type_bucket']==pitch_type,launch_angle+'_raw'].mul(df.loc[df['pitch_type_bucket']==pitch_type,'in_play_input'])
#         df.loc[df['pitch_type_bucket']==pitch_type,'50+deg_pred_loc'] = df.loc[df['pitch_type_bucket']==pitch_type,'50+deg_raw'].mul(df.loc[df['pitch_type_bucket']==pitch_type,'in_play_input'])
    
#         # Launch Velo Result
#         for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
#             with open('2024_pl_{}_model_{}_loc.pkl'.format(launch_angle,pitch_type), 'rb') as f:
#                 launch_velo_model = pickle.load(f)
    
#             df.loc[df['pitch_type_bucket']==pitch_type,[launch_angle+': <90mph_raw',launch_angle+': 90-95mph_raw',launch_angle+': 95-100mph_raw',launch_angle+': 100-105mph_raw',launch_angle+': 105+mph_raw']] = launch_velo_model.predict_proba(df.loc[df['pitch_type_bucket']==pitch_type,launch_velo_model.feature_names_in_])
#             for bucket in [launch_angle+': '+x for x in ['<90mph','90-95mph','95-100mph','100-105mph','105+mph']]:
#                 df.loc[df['pitch_type_bucket']==pitch_type,bucket+'_pred_loc'] = df.loc[df['pitch_type_bucket']==pitch_type,bucket+'_raw'].mul(df.loc[df['pitch_type_bucket']==pitch_type,launch_angle+'_input'])

#     # Apply averages to each predicted grouping
#     for outcome in ['out', 'single', 'double', 'triple', 'home_run']:
#         # Start with 50+ degrees (popups)
#         df[outcome+'_pred'] = df['50+deg_pred']*bip_result_dict['50+deg'][outcome]

#         for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
#             for bucket in [launch_angle+': '+x for x in ['<90mph','90-95mph','95-100mph','100-105mph','105+mph']]:
#                 df[outcome+'_pred'] += df[bucket+'_pred']*bip_result_dict[bucket][outcome]

#     ### Find the estimated change in wOBA/runs for each pitch
#     # wOBA value of an outcome, based on the count that it came in
#     outcome_wOBAs = pd.read_csv('data_woba_outcome.csv').set_index(['year_played','balls','strikes'])

#     df = df.merge(outcome_wOBAs,
#                   how='left',
#                   on=['year_played','balls','strikes'])

#     # wOBA_effect is how the pitch is expected to affect wOBA
#     # (either by moving the count, or by ending the PA)
#     df['wOBA_effect'] = 0

#     for stat in [x[:-5] for x in list(outcome_wOBAs.columns)]:
#         df['wOBA_effect'] = df['wOBA_effect'].add(df[stat+'_pred'].fillna(df[stat+'_pred'].median()).mul(df[stat+'_wOBA'].fillna(df[stat+'_wOBA'].median())))

#     return df['wOBA_effect'].sub(-0.004253050593194383).div(0.05179234832326223).mul(-50).add(100)

def fastball_differences(dataframe,stat):
    dataframe[stat] = dataframe[stat].astype('float')
    temp_df = dataframe.loc[dataframe['pitch_type']==dataframe['fastball_type']].groupby(['pitcher_id','game_date','pitch_type','stand'], as_index=False)[stat].mean().rename(columns={stat:'fb_'+stat})
    dataframe = dataframe.merge(temp_df,
                                left_on=['pitcher_id','game_date','fastball_type'],
                                right_on=['pitcher_id','game_date','pitch_type']).drop(columns=['pitch_type_y']).rename(columns={'pitch_type_x':'pitch_type'})
    return dataframe[stat].sub(dataframe['fb_'+stat])
    
def get_mode(series):
    mode = series.mode()
    if mode.size==0:
        return None
    else:
        return series.mode()[0]


today = (datetime.datetime.now()-timedelta(hours=16)).date()
st.header(f"PL Live Feed")
col1, col2, col3 = st.columns([1/3,1/3,1/3])
with col1:
    level = st.selectbox('Choose a level:', ['MLB','AAA','A (FSL)','AFL'])
    level_dict = {'MLB':1,'AAA':11,'A (FSL)':14, 'AFL':17}
    level_code = level_dict[level]
with col2:
    date = st.date_input("Select a game date:", today, min_value=datetime.date(2021, 3, 1), max_value=today)

@st.cache_data(ttl=90,show_spinner=f"Loading data")
def scrape_pitch_data(date,level):
    pitch_data={}
    r = requests.get(f'https://statsapi.mlb.com/api/v1/schedule?sportId={level}&date={date}')
    x = r.json()
    if len(x['dates'])==0:
        st.write('No games today')
        st.stop()
    
    game_list = []
    for game in range(len(x['dates'][0]['games'])):
        game_list += [x['dates'][0]['games'][game]['gamePk']]
    
    for game_id in game_list:
        r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_id}')
        x = r.json()
        for home_away_pitcher in ['home','away']:
            if f'{home_away_pitcher}_pitchers' not in x.keys():
                continue
            for pitcher_id in list(x[f'{home_away_pitcher}_pitchers'].keys()):
                for pitch in range(len(x[f'{home_away_pitcher}_pitchers'][pitcher_id])):
                    pitch_data.update({
                        x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['play_id']:{
                            'game_date':x['gameDate'],
                            'game_status_code':x['game_status_code'],
                            'gamedayType':x['gamedayType'],
                            'game_id':game_id,
                            'venue_id':x['venue_id'],
                            'park_name':x['home_team_data']['venue']['name'],
                            'home_team':x['home_team_data']['abbreviation'],
                            'away_team':x['away_team_data']['abbreviation'],
                            'pitcher_id':pitcher_id,
                            'pitcher_name':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitcher_name'],
                            'p_throws':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['p_throws'],
                            'stand':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['stand'],
                            'inning':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['inning'],
                            'outs':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['outs'],
                            'balls':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['balls'],
                            'strikes':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['strikes'],
                            'pitch_type':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitch_type'] if 'pitch_type' in x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch].keys() else 'UN',
                            'batter':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['batter'],
                            'batter_name':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['batter_name'],
                            'description':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['description'],
                            'events':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['events'],
                            'hit_speed':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['hit_speed'] if 'hit_speed' in x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch].keys() else None,
                            'hit_angle':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['hit_angle'] if 'hit_angle' in x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch].keys() else None
                            }
                    })
                    try:
                        pitch_data[x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['play_id']].update({
                            'velo':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['start_speed'],
                            'ivb':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['inducedBreakZ'],
                            'extension':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['extension'] if 'extension' in x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch].keys() else None,
                            'spin_rate':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['spin_rate'],
                            'x0':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['x0'],
                            'z0':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['z0'],
                            'vx0':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['vx0'],
                            'vy0':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['vy0'],
                            'vz0':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['vz0'],
                            'ax':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['ax'],
                            'ay':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['ay'],
                            'az':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['az'],
                            'px':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['px'],
                            'pz':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pz'],
                            'sz_top':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['sz_top'],
                            'sz_bot':x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['sz_bot']
                        })
                    except KeyError:
                        None
    pitch_df = pd.DataFrame.from_dict({i: pitch_data[i] for i in pitch_data.keys()},
                                       orient='index')
    # pitch_df['Starter'] = np.where(pitch_df['inning'].groupby(pitch_df['pitcher_id']).transform('min')==1,1,0)
    group_map = {
        'FF':'Fastball',
        'SI':'Fastball',
        'FC':'Fastball',
        'FA':'Fastball',
        'CH':'Offspeed',
        'FS':'Offspeed',
        'FO':'Offspeed',
        'SC':'Offspeed',
        'KN':'Offspeed',
        'EP':'Offspeed',
        'CU':'Breaking Ball',
        'KC':'Breaking Ball',
        'CS':'Breaking Ball',
        'SL':'Breaking Ball',
        'ST':'Breaking Ball',
        'SV':'Breaking Ball',
        'IN':'Other',
        'PO':'Other'
    }
    pitch_df['pitch_type_bucket'] = pitch_df['pitch_type'].map(group_map)
    pitch_df[['IHB','IVB','total_IB']] = spin_calcs(pitch_df)
    pitch_df['IHB'] = np.where(pitch_df['p_throws']=='R',pitch_df['IHB'].mul(-1),pitch_df['IHB'])
    pitch_df[['VAA','HAVAA']] = adjusted_vaa(pitch_df)
  
    pitch_df['fastball_only'] = np.where(pitch_df['pitch_type_bucket']=='Fastball',pitch_df['pitch_type'],None)
    pitch_df['fastball_type'] = pitch_df.groupby(['pitcher_id','game_date','stand'])['fastball_only'].transform(get_mode).fillna(pitch_df['pitch_type'])
    
    # Add comparison stats to fastball
    for stat in ['IHB','IVB','velo']:
        pitch_df[stat+'_diff'] = fastball_differences(pitch_df,stat)
    pitch_df['total_IB_diff'] = (pitch_df['IHB_diff'].astype('float')**2+pitch_df['IVB_diff'].astype('float')**2)**0.5
    dummy_cols = list(pd.get_dummies(pitch_df[['p_throws','stand','balls','strikes']].astype('str')).columns.values)
    pitch_df[dummy_cols] = pd.get_dummies(pitch_df[['p_throws','stand','balls','strikes']].astype('str'))
  
    return pitch_df.reset_index().rename(columns={'index':'pitch_id'})

chart_df = scrape_pitch_data(date,level_code)
# st.write(chart_df['balls'].unique())

if chart_df.shape[0]==0:
    st.write('No fastballs thrown')
    st.stop()

category_feats = ['p_throws_L', 'stand_L',
                  'balls_1','balls_2','balls_3',
                  'strikes_1','strikes_2'
                 ]
stuff_feats = ['velo','velo_diff',
               'extension',
               'spin_rate',
               'adj_vaa',
               'x0','z0',
               'IHB','IVB','total_IB',
               'IHB_diff','IVB_diff','total_IB_diff', # Induced Stuff
              ]

pitch_types = ['Fastball','Breaking Ball','Offspeed']
model_feats = stuff_feats+category_feats

def stuff_preds(df):
    cols = list(df.columns.values)
    df[['stuff_reg','swinging_strike_raw_temp','contact_raw_temp','contact_input_temp',
                'foul_strike_raw_temp','in_play_raw_temp','10deg_raw_temp','10-20deg_raw_temp',
                '20-30deg_raw_temp','30-40deg_raw_temp','40-50deg_raw_temp','50+deg_raw_temp',
                'swinging_strike_pred','foul_strike_pred','in_play_input','50+deg_pred',
                'out_pred', 'single_pred', 'double_pred', 'triple_pred', 'home_run_pred']] = None
    
    for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
        df[[launch_angle+'_input_temp',launch_angle+': <90mph_raw_temp',
                 launch_angle+': 90-95mph_raw_temp',launch_angle+': 95-100mph_raw_temp',
                 launch_angle+': 100-105mph_raw_temp',launch_angle+': 105+mph_raw_temp',
                 launch_angle+': <90mph_pred',launch_angle+': 90-95mph_pred',
                 launch_angle+': 95-100mph_pred',launch_angle+': 100-105mph_pred',
                 launch_angle+': 105+mph_pred']] = None
        
    for pitch_type in ['Fastball','Breaking Ball','Offspeed']:
         # Regression Model
        with open(f'live_stuff/models/live_stuff_rv_model_{pitch_type}.pkl', 'rb') as f:
            stuff_model = pickle.load(f)
    
        df.loc[df['pitch_type_bucket']==pitch_type,'stuff_reg'] = stuff_model.predict(df.loc[df['pitch_type_bucket']==pitch_type,stuff_model.feature_names_in_])
      
        # Swing Result        
        with open(f'live_stuff/models/live_stuff_contact_model_{pitch_type}.pkl', 'rb') as f:
            swing_result_model = pickle.load(f)
    
        df.loc[df['pitch_type_bucket']==pitch_type,['swinging_strike_pred','contact_input_temp']] = swing_result_model.predict_proba(df.loc[df['pitch_type_bucket']==pitch_type,swing_result_model.feature_names_in_])
        # print(pitch_type+' Swing Result model done')
    
        # Contact Result
        with open(f'live_stuff/models/live_stuff_in_play_model_{pitch_type}.pkl', 'rb') as f:
            contact_model = pickle.load(f)
    
        df.loc[df['pitch_type_bucket']==pitch_type,['foul_strike_raw_temp','in_play_raw_temp']] = contact_model.predict_proba(df.loc[df['pitch_type_bucket']==pitch_type,contact_model.feature_names_in_])
        df.loc[df['pitch_type_bucket']==pitch_type,'foul_strike_pred'] = df.loc[df['pitch_type_bucket']==pitch_type,'foul_strike_raw_temp'].mul(df.loc[df['pitch_type_bucket']==pitch_type,'contact_input_temp'])
        df.loc[df['pitch_type_bucket']==pitch_type,'in_play_input'] = df.loc[df['pitch_type_bucket']==pitch_type,'in_play_raw_temp'].mul(df.loc[df['pitch_type_bucket']==pitch_type,'contact_input_temp'])
        # print(pitch_type+' Contact model done')
    
        # Launch Angle Result
        with open(f'live_stuff/models/live_stuff_launch_angle_model_{pitch_type}.pkl', 'rb') as f:
            launch_angle_model = pickle.load(f)
    
        df.loc[df['pitch_type_bucket']==pitch_type,['10deg_raw_temp','10-20deg_raw_temp','20-30deg_raw_temp','30-40deg_raw_temp','40-50deg_raw_temp','50+deg_raw_temp']] = launch_angle_model.predict_proba(df.loc[df['pitch_type_bucket']==pitch_type,launch_angle_model.feature_names_in_])
        for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
            df.loc[df['pitch_type_bucket']==pitch_type,launch_angle+'_input_temp'] = df.loc[df['pitch_type_bucket']==pitch_type,launch_angle+'_raw_temp'].mul(df.loc[df['pitch_type_bucket']==pitch_type,'in_play_input'])
        df.loc[df['pitch_type_bucket']==pitch_type,'50+deg_pred'] = df.loc[df['pitch_type_bucket']==pitch_type,'50+deg_raw_temp'].mul(df.loc[df['pitch_type_bucket']==pitch_type,'in_play_input'])
        # print(pitch_type+' Launch Angle model done')
    
        # Launch Velo Result
        for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
            with open(f'live_stuff/models/live_stuff_{launch_angle}_model_{pitch_type}.pkl', 'rb') as f:
                launch_velo_model = pickle.load(f)
    
            df.loc[df['pitch_type_bucket']==pitch_type,[launch_angle+': <90mph_raw_temp',launch_angle+': 90-95mph_raw_temp',launch_angle+': 95-100mph_raw_temp',launch_angle+': 100-105mph_raw_temp',launch_angle+': 105+mph_raw_temp']] = launch_velo_model.predict_proba(df.loc[df['pitch_type_bucket']==pitch_type,launch_velo_model.feature_names_in_])
            for bucket in [launch_angle+': '+x for x in ['<90mph','90-95mph','95-100mph','100-105mph','105+mph']]:
                df.loc[df['pitch_type_bucket']==pitch_type,bucket+'_pred'] = df.loc[df['pitch_type_bucket']==pitch_type,bucket+'_raw_temp'].mul(df.loc[df['pitch_type_bucket']==pitch_type,launch_angle+'_input_temp'])

    run_expectancies = pd.read_csv('live_stuff/re_12_vals.csv').set_index(['cleaned_description','count']).to_dict()['delta_re']
    df['count'] = df['balls'].astype('str')+'_'+df['strikes'].astype('str')
    
    df['delta_re'] = 0
    df['delta_re_str'] = 0
    df['delta_re_bbe'] = 0
    
    # Apply averages to each predicted grouping
    for outcome in ['out', 'single', 'double', 'triple', 'home_run']:
        # Start with 50+ degrees (popups)
        df[outcome+'_pred'] = df['50+deg_pred']*bip_result_dict['50+deg'][outcome]
        
        for launch_angle in ['10deg','10-20deg','20-30deg','30-40deg','40-50deg']:
            for bucket in [launch_angle+': '+x for x in ['<90mph','90-95mph','95-100mph','100-105mph','105+mph']]:
                df[outcome+'_pred'] += df[bucket+'_pred']*bip_result_dict[bucket][outcome]
    
    for stat in ['swinging_strike','foul_strike','out','single','double','triple','home_run']:
        df[stat+'_re'] = stat
        df[stat+'_re'] = df[[stat+'_re','count']].apply(tuple,axis=1).map(run_expectancies)
        df['delta_re'] = df['delta_re'].add(df[stat+'_pred'].mul(df[stat+'_re']))
        if stat in ['swinging_strike','foul_strike']:
            df['delta_re_str'] = df['delta_re_str'].add(df[stat+'_pred'].mul(df[stat+'_re']))
        else:
            df['delta_re_bbe'] = df['delta_re_bbe'].add(df[stat+'_pred'].mul(df[stat+'_re']))
    
    df['wOBAcon_pred'] = df[['single_pred', 'double_pred', 'triple_pred', 'home_run_pred']].mul([0.9,1.25,1.6,2]).sum(axis=1).div(df['in_play_input'].astype('float'))
    df['str_rv'] = df['delta_re_str'].sub(-0.047944)
    df['bbe_rv'] = df['delta_re_bbe'].sub(0.014494)
    df['stuff_rv'] = df['delta_re'].sub(-0.03345)
    df['stuff_class'] = df['stuff_rv'].div(0.041137).mul(-50).add(100)
    df['stuff_reg'] = df['stuff_reg'].sub(-0.03030).div(0.04699).mul(-50).add(100)

    df['plvStuff+'] = df['stuff_class']#df[['stuff_class','stuff_reg']].mean(axis=1)
    return df[cols+['stuff_class','stuff_reg','plvStuff+']].copy()

chart_df = stuff_preds(chart_df)

model_df = (chart_df
            .rename(columns={
              'pitcher_name':'Pitcher',
              'pitch_id':'#',
              'extension':'Ext',
              'velo':'Velo',
              'spin_rate':'Spin'
            })
            .groupby(['Pitcher'#,'pitch_type'
                     ])
            [['#','plvStuff+',
              'Velo','Ext','IVB','HAVAA','IHB','Spin','VAA','x0','z0'
             ]]
            .agg({
                '#':'count',
                'plvStuff+':'mean',
                'Velo':'mean',
                'Ext':'mean',
                'IVB':'mean',
                'HAVAA':'mean',
                'IHB':'mean',
                'Spin':'mean',
                'VAA':'mean',
                'x0':'mean',
                'z0':'mean',
              })
            .sort_values('plvStuff+',ascending=False)
           )

st.dataframe(model_df
             .style
             .format(precision=1, thousands=',')
             .background_gradient(axis=0, vmin=85, vmax=115, cmap="vlag", subset=['plvStuff+']),
            column_config={"Pitcher": st.column_config.Column(width="medium")}
)

players = list(chart_df
               .groupby('pitcher_name')
               ['plvStuff+']
               .mean()
               .reset_index()
               .sort_values('plvStuff+', ascending=False)
               ['pitcher_name']
              )

pitcher = st.selectbox('Choose a pitcher:', players)

player_df = (chart_df
             .loc[chart_df['pitcher_name']==pitcher]
             .rename(columns={
              'pitcher_name':'Pitcher',
              'pitch_id':'#',
              'extension':'Ext',
              'velo':'Velo',
              'spin_rate':'Spin'
            })
            .groupby('pitch_type')
            [['#','plvStuff+','Velo','Ext','IVB','HAVAA','IHB','Spin','VAA','x0','z0']]
            .agg({
                '#':'count',
                'plvStuff+':'mean',
                'Velo':'mean',
                'Ext':'mean',
                'IVB':'mean',
                'HAVAA':'mean',
                'IHB':'mean',
                'Spin':'mean',
                'VAA':'mean',
                'x0':'mean',
                'z0':'mean',
              })
            .sort_values('#',ascending=False)
            )
st.write(f"{pitcher}'s {date} Repertoire")
st.dataframe(player_df
             .style
             .format(precision=1, thousands=',')
             .background_gradient(axis=0, vmin=85, vmax=115, cmap="vlag", subset=['plvStuff+']),
            column_config={"Pitcher": st.column_config.Column(width="medium")}
)
