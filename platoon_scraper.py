"""
FanGraphs Roster Resource Platoon Scraper
Scrapes all 30 MLB team pages to identify platooned players
"""

import requests #2.32.3
from bs4 import BeautifulSoup #4.12.3
import pandas as pd #2.2.2
import time
from typing import List, Dict

# All 30 MLB teams with their FanGraphs roster resource URLs
TEAMS = {
    'orioles': 'BAL',
    'red-sox': 'BOS',
    'yankees': 'NYY',
    'rays': 'TBR',
    'blue-jays': 'TOR',
    'white-sox': 'CHW',
    'guardians': 'CLE',
    'tigers': 'DET',
    'royals': 'KCR',
    'twins': 'MIN',
    'astros': 'HOU',
    'angels': 'LAA',
    'athletics': 'OAK',
    'mariners': 'SEA',
    'rangers': 'TEX',
    'braves': 'ATL',
    'marlins': 'MIA',
    'mets': 'NYM',
    'phillies': 'PHI',
    'nationals': 'WSN',
    'cubs': 'CHC',
    'reds': 'CIN',
    'brewers': 'MIL',
    'pirates': 'PIT',
    'cardinals': 'STL',
    'diamondbacks': 'ARI',
    'rockies': 'COL',
    'dodgers': 'LAD',
    'padres': 'SDP',
    'giants': 'SFG'
}

BASE_URL = "https://www.fangraphs.com/roster-resource/depth-charts/"

def get_page_content(team_slug: str) -> BeautifulSoup:
    """
    Fetch and parse the HTML content for a team's roster page
    """
    url = f"{BASE_URL}{team_slug}"

    ### Commented out print statement
    # print(f"Fetching {url}...")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return BeautifulSoup(response.content, 'html.parser')


def extract_platooned_players(soup: BeautifulSoup, team_code: str) -> List[Dict[str, str]]:
    """
    Extract platooned players from the JSON data embedded in the page
    platoon:"vsL" = platooned vs LHP (plays against LHP)
    platoon:"vsR" = platooned vs RHP (plays against RHP)
    """
    import json
    
    platooned_players = []
    
    # Find the Next.js data script tag
    scripts = soup.find_all('script', id='__NEXT_DATA__')
    
    if not scripts:
        print(f"  Warning: Could not find __NEXT_DATA__ script for {team_code}")
        return platooned_players
    
    try:
        # Parse the JSON data
        data = json.loads(scripts[0].string)
        
        # Navigate to the roster data
        # The structure is: props -> pageProps -> dehydratedState -> queries -> [0] -> state -> data -> dataRoster
        queries = data.get('props', {}).get('pageProps', {}).get('dehydratedState', {}).get('queries', [])
        
        if not queries:
            print(f"  Warning: No queries found in data for {team_code}")
            return platooned_players
        
        # Get the roster data from the first query
        roster_data = queries[0].get('state', {}).get('data', {}).get('dataRoster', [])
        
        # Iterate through all players in the roster
        for player_data in roster_data:
            platoon_value = player_data.get('platoon', '')
            
            # Check if player is platooned
            # vsL = plays vs LHP (so marked as L in output)
            # vsR = plays vs RHP (so marked as R in output)
            if platoon_value == 'vsL':
                player_name = player_data.get('player', '')
                if player_name:
                    platooned_players.append({
                        'Name': player_name,
                        'Team': team_code,
                        'Platoon Start': 'L'
                    })
            elif platoon_value == 'vsR':
                player_name = player_data.get('player', '')
                if player_name:
                    platooned_players.append({
                        'Name': player_name,
                        'Team': team_code,
                        'Platoon Start': 'R'
                    })
        
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  Error parsing JSON data for {team_code}: {e}")
    
    return platooned_players


def scrape_all_teams() -> pd.DataFrame:
    """
    Scrape all 30 MLB teams and compile platoon data
    Resulting df columns are [Player Name,Team,Platoon (R/L)]
    """
    all_platooned_players = []
    
    for team_slug, team_code in TEAMS.items():
        try:
            soup = get_page_content(team_slug)
            platooned = extract_platooned_players(soup, team_code)

            ### Commented out print statement
            # print(f"  Found {len(platooned)} platooned players for {team_code}")
            all_platooned_players.extend(platooned)
            
            # Be respectful with rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"  Error processing {team_code}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_platooned_players)
    
    return df


def main():
    """
    Main execution function
    """
    ### Commented out print statement
    # print("Starting FanGraphs Roster Resource platoon scraper...")
    # print(f"Scraping {len(TEAMS)} teams...\n")
    
    # Scrape all teams
    df = scrape_all_teams()
    
    ### Could probably leave this and just return df
    # Save to CSV
    output_file = 'platooned_players.csv'
    df.to_csv(output_file, index=False)

    ### Commented out print statements
    # print(f"\n{'='*60}")
    # print(f"Scraping complete!")
    # print(f"Total platooned players found: {len(df)}")
    # print(f"Output saved to: {output_file}")
    # print(f"{'='*60}\n")
    
    # Display summary
    # if len(df) > 0:
    #     print("Summary by team:")
    #     print(df.groupby('team').size().sort_values(ascending=False))
    #     print("\nSummary by platoon type:")
    #     print(df.groupby('platooned').size())
    #     print("\nFirst 10 rows:")
    #     print(df.head(10))
    # else:
    #     print("No platooned players found. The page structure may have changed.")

if __name__ == "__main__":
    main()
