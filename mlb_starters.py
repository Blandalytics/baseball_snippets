import datetime
import os
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from pathlib import Path

API_KEY = '' # Enter API Key here

# Load IDs from database
dotenv_path = Path('pitcherlist_datascience.env')
load_dotenv(dotenv_path=dotenv_path)

conn = psycopg2.connect(f"dbname='{os.environ.get('PL_DB_DATABASE')}' user='{os.environ.get('PL_DB_USER')}' host='{os.environ.get('PL_DB_HOST')}' password='{os.environ.get('PL_DB_PASSWORD')}'")
cursor = conn.cursor()

cursor.execute("Select * FROM players")
colnames = [desc[0] for desc in cursor.description]
data = cursor.fetchall()

id_data = pd.DataFrame(data).copy()
id_data.columns = colnames

cursor.close()
conn.close()

id_map = id_data[['mlb_player_id','sportradar_player_id']].set_index('sportradar_player_id').to_dict()['mlb_player_id']
hand_map = id_data[['batting_hand','sportradar_player_id']].set_index('sportradar_player_id').to_dict()['batting_hand']
name_map = id_data[['full_name','sportradar_player_id']].set_index('sportradar_player_id').to_dict()['full_name']

team_map = {
    'ANA':'LAA', 'ATL':'ATL', 'AZ':'ARI', 'BAL':'BAL', 'BOS':'BOS', 
    'CHI-A':'CHW', 'CHI-N':'CHC', 'CIN':'CIN', 'CLE':'CLE','COL':'COL', 
    'DET':'DET', 'HOU':'HOU', 'KC':'KCR', 'LA':'LAD', 'MIA':'MIA', 
    'MIL':'MIL', 'MIN':'MIN', 'NY-A':'NYY','NY-N':'NYM', 'OAK':'OAK', 
    'PHI':'PHI', 'PIT':'PIT', 'SD':'SDP', 'SEA':'SEA', 'SF':'SFG', 
    'STL':'STL', 'TB':'TBR', 'TEX':'TEX','TOR':'TOR', 'WSH':'WSN'
}

def extract_pitcher_values(dictionary):
    if len(dictionary.keys())==0:
        game_id = None
        home_team = None
        home_starter_id = None
        home_starter_name = None
        home_starter_designation = None
        away_team = None
        away_starter_id = None
        away_starter_name = None
        away_starter_designation = None
    else:
        game_id = dictionary['Id']
        if dictionary['Teams'][0]['IsHome']==1:
            home_team = dictionary['Teams'][0]['Code']
            home_starter_id = dictionary['Teams'][0]['StartingPitcher']['SportsDataId']
            home_starter_name = dictionary['Teams'][0]['StartingPitcher']['FirstName'] + ' ' + dictionary['Teams'][0]['StartingPitcher']['LastName']
            home_starter_designation = dictionary['Teams'][0]['StartingPitcher']['Designation']
            if len(dictionary['Teams']) >1:
                away_team = dictionary['Teams'][1]['Code']
                away_starter_id = dictionary['Teams'][1]['StartingPitcher']['SportsDataId']
                away_starter_name = dictionary['Teams'][1]['StartingPitcher']['FirstName'] + ' ' + dictionary['Teams'][1]['StartingPitcher']['LastName']
                away_starter_designation = dictionary['Teams'][1]['StartingPitcher']['Designation']
            else:
                away_team = None
                away_starter_id = None
                away_starter_name = None
                away_starter_designation = None
        else:
            away_team = dictionary['Teams'][0]['Code']
            away_starter_id = dictionary['Teams'][0]['StartingPitcher']['SportsDataId']
            away_starter_name = dictionary['Teams'][0]['StartingPitcher']['FirstName'] + ' ' + dictionary['Teams'][0]['StartingPitcher']['LastName']
            away_starter_designation = dictionary['Teams'][0]['StartingPitcher']['Designation']
            if len(dictionary['Teams']) >1:
                home_team = dictionary['Teams'][1]['Code']
                home_starter_id = dictionary['Teams'][1]['StartingPitcher']['SportsDataId']
                home_starter_name = dictionary['Teams'][1]['StartingPitcher']['FirstName'] + ' ' + dictionary['Teams'][1]['StartingPitcher']['LastName']
                home_starter_designation = dictionary['Teams'][1]['StartingPitcher']['Designation']
            else:
                home_team = None
                home_starter_id = None
                home_starter_name = None
                home_starter_designation = None
    
    return game_id, home_team, home_starter_id, home_starter_name, home_starter_designation, away_team, away_starter_id, away_starter_name, away_starter_designation

def sp_schedule(start_date=datetime.date.today(),n_days=21):
    week_df = pd.DataFrame()
    for day in pd.date_range(start_date, periods=n_days):
        test_starters = pd.read_json(f'https://api.rotowire.com/Baseball/MLB/ProjectedStarters.php?key={API_KEY}&format=json&date='+day.strftime('%m%d%Y'))
        test_starters[['game_id', 'home_team', 'home_starter_id', 'home_starter_name', 'home_starter_designation', 
                        'away_team', 'away_starter_id', 'away_starter_name', 'away_starter_designation']] = None
        if test_starters.shape[0]>0:
            test_starters[['game_id', 'home_team', 'home_starter_id', 'home_starter_name', 'home_starter_designation', 
                        'away_team', 'away_starter_id', 'away_starter_name', 'away_starter_designation']] = test_starters['Games'].apply(lambda x: pd.Series(extract_pitcher_values(x)))
        week_df = pd.concat([week_df,test_starters], ignore_index=True)
    week_df['home_team'] = week_df['home_team'].map(team_map)
    week_df['away_team'] = week_df['away_team'].map(team_map)
    week_df['home_mlbamid'] = week_df['home_starter_id'].map(id_map)
    week_df['away_mlbamid'] = week_df['away_starter_id'].map(id_map)
    week_df['home_starter_name'] = np.where(~week_df['home_starter_id'].isna(),week_df['home_starter_id'].map(name_map),week_df['home_starter_name'])
    week_df['away_starter_name'] = np.where(~week_df['away_starter_id'].isna(),week_df['away_starter_id'].map(name_map),week_df['away_starter_name'])
    
    return pd.concat([
        week_df[['Date','game_id','home_mlbamid','home_starter_name','home_team','away_team','home_starter_designation']].rename(columns={
            'Date':'Game Date',
            'home_mlbamid':'MLBAMID',
            'home_starter_name':'Pitcher Name',
            'home_team':'Team',
            'away_team':'Opp',
            'home_starter_designation':'Designation'
        }).assign(Park = lambda x: x['Team'],h_a = 'home'),
        week_df[['Date','game_id','away_mlbamid','away_starter_name','away_team','home_team','away_starter_designation']].rename(columns={
            'Date':'Game Date',
            'away_mlbamid':'MLBAMID',
            'away_starter_name':'Pitcher Name',
            'away_team':'Team',
            'home_team':'Opp',
            'away_starter_designation':'Designation'
        }).assign(Park = lambda x: x['Opp'],h_a = 'away')],
        ignore_index=True).sort_values('Game Date')

sp_schedule()[['Game Date','MLBAMID','Pitcher Name','Team','Opp','Park','h_a','game_id','Designation']]
