### Install pybaseball package to import data
### Not necessary if you already have the data
#!pip install pybaseball -q 
from pybaseball import statcast, cache
cache.enable() # Optional

import datetime
import numpy as np
import pandas as pd

### Season Thresholds (roughly encapsulates the regular season for any given year)
this_year = 2023
season_start = datetime.datetime(this_year, 3, 1)
season_end = datetime.datetime(this_year, 11, 1)

### Use pybaseball to load data for season
pitch_data = statcast(start_dt=season_start.strftime('%Y-%m-%d'),end_dt=season_end.strftime('%Y-%m-%d'))

### Create a base_state field, containing all of 1B/2B/3B
for base in ['1b','2b','3b']:
  pitch_data.loc[pitch_data['on_'+base].notna(),'on_'+base] = int(base[0])
  pitch_data['on_'+base] = pitch_data['on_'+base].astype('str').replace('<NA>','_')
pitch_data['base_state'] = pitch_data['on_1b']+' '+pitch_data['on_2b']+' '+pitch_data['on_3b']

### Determine how many runs were scored for each inning
pitch_data['start_inning_score'] = pitch_data['bat_score'].groupby([pitch_data['game_pk'],pitch_data['inning'],pitch_data['inning_topbot']]).transform('min')
pitch_data['end_inning_score'] = pitch_data['bat_score'].groupby([pitch_data['game_pk'],pitch_data['inning'],pitch_data['inning_topbot']]).transform('max')
pitch_data['inning_runs'] = pitch_data['end_inning_score'].sub(pitch_data['start_inning_score']).astype('int')

### This chunk helps pandas sort the base states
### It's optional and doesn't affect the calculations
# from pandas.api.types import CategoricalDtype
# base_state_cats = CategoricalDtype(
#     ['_ _ _', 
#      '1 _ _', 
#      '_ 2 _', 
#      '_ _ 3', 
#      '1 2 _', 
#      '1 _ 3', 
#      '_ 2 3', 
#      '1 2 3'], 
#     ordered=True
# )
# pitch_data['base_state'] = pitch_data['base_state'].astype(base_state_cats)

### Generate a dataframe for the 24 base-out states
### 3 outs x 8 base states
re_24_df = (pitch_data
            .dropna(subset=['events'])
            .groupby(['game_year','base_state','outs_when_up'])
            ['inning_runs']
            .mean()
            .reset_index()
            .pivot(index=['game_year','base_state'],
                   columns='outs_when_up',
                   values='inning_runs')
            .copy()
)

### Commit year to csv
re_24_df.to_csv(f'RE24 Matrix - {}.csv')
