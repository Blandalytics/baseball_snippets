import streamlit as st
import datetime
from datetime import timedelta
import matplotlib as mpl
import seaborn as sns
import requests
import numpy as np
import pandas as pd

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
    needed_cols = ['Velo','px','pz','vx0','vy0',
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

today = (datetime.datetime.today() - timedelta(hours=6)).date()
st.header(f"Today's 4-Seam Fastballs by Starters ({today.strftime('%#m/%#d/%Y')})")

def load_savant(date=today):
    r = requests.get(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}')
    x = r.json()
    
    game_list = []
    for game in range(len(x['dates'][0]['games'])):
        game_list += [x['dates'][0]['games'][game]['gamePk']]
    
    game_date = []
    pitcher_id_list = []
    pitcher_name = []
    throws = []
    pitch_id = []
    pitch_type = []
    inning = []
    out = []
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
    for game_id in game_list:
        r = requests.get(f'https://baseballsavant.mlb.com/gf?game_pk={game_id}')
        x = r.json()
        if x['game_status_code'] in ['P','S']:
            continue
        for home_away_pitcher in ['home','away']:
            for pitcher_id in list(x[f'{home_away_pitcher}_pitchers'].keys()):
                for pitch in range(len(x[f'{home_away_pitcher}_pitchers'][pitcher_id])):
                    game_date += [x['gameDate']]
                    pitcher_id_list += [pitcher_id]
                    pitcher_name += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitcher_name']]
                    throws += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['p_throws']]
                    inning += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['inning']]
                    out += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['outs']]
                    pitch_id += [pitch]
                    pitch_type += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['pitch_type']]
                    velo += [x[f'{home_away_pitcher}_pitchers'][pitcher_id][pitch]['start_speed']]
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
    
    player_df = pd.DataFrame()
    player_df['game_date'] = game_date
    player_df['MLBAMID'] = pitcher_id_list
    player_df['Pitcher'] = pitcher_name
    player_df['P Hand'] = throws
    player_df['Inning'] = inning
    player_df['Out'] = out
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
    player_df['px'] = px
    player_df['pz'] = pz
    player_df[['IHB','IVB']] = spin_calcs(player_df)
    player_df['IHB'] = np.where(player_df['P Hand']=='R',player_df['IHB'].mul(-1),player_df['IHB'])
    player_df[['raw_vaa','HAVAA']] = adjusted_vaa(player_df)
  
    return (player_df
            .loc[(player_df['pitch_type']=='FF') &
                    (player_df['Inning'].groupby(player_df['MLBAMID']).transform('min')==1) & 
                    (player_df['Out'].groupby(player_df['MLBAMID']).transform('min')==0)]
            .groupby(['MLBAMID','Pitcher','P Hand'])
            [['#','Velo','Ext','IVB','HAVAA','IHB']]
            .agg({
              '#':'count',
              'Velo':'mean',
              'Ext':'mean',
              'IVB':'mean',
              'HAVAA':'mean',
              'IHB':'mean',
              })
            .sort_values('#',ascending=False)
           )

st.dataframe(load_savant()
             .style
             .format(precision=1, thousands=',')
             .background_gradient(axis=0, vmin=91.4, vmax=96.6, cmap="vlag", subset=['Velo'])
             .background_gradient(axis=0, vmin=5.95, vmax=6.95, cmap="vlag", subset=['Ext'])
             .background_gradient(axis=0, vmin=12.4, vmax=18.2, cmap="vlag", subset=['IVB'])
             .background_gradient(axis=0, vmin=0.4, vmax=1.5, cmap="vlag", subset=['HAVAA']))
