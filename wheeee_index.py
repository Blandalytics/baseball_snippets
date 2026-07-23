import streamlit as st
import datetime
import html
from datetime import timedelta
import numpy as np
import polars as pl
import seaborn as sns
import requests
import pytz
import matplotlib as mpl
import matplotlib.pyplot as plt
import urllib
import cairosvg
from PIL import Image

from matplotlib.path import Path
from matplotlib.patches import PathPatch

from concurrent.futures import ThreadPoolExecutor, as_completed


from pyfonts import load_google_font
import matplotlib.font_manager as fm


@st.cache_data(ttl=3600)
def load_logo():
    logo_loc = "https://res.cloudinary.com/dduabusaf/image/upload/v1772839288/PitcherList_Stats_watermark_with_logo_k9e3xa.webp"
    logo = Image.open(urllib.request.urlopen(logo_loc))
    return logo


logo = load_logo()


@st.cache_data(ttl=3600)
def letter_logo():
    logo_loc = "https://res.cloudinary.com/dduabusaf/image/upload/v1772839606/teal_letter_logo_owufaj.png"
    logo = Image.open(urllib.request.urlopen(logo_loc))
    return logo


letter_logo = letter_logo()


def load_team_logo(team_abbr):
    cairosvg.svg2png(url=logo_dict[team_abbr][0], write_to="team_logo.png")
    image = Image.open("team_logo.png")
    return image

base_font = "DM Sans"
for _weight in (400, 700):
    _f = load_google_font(base_font, weight=_weight)
    fm.fontManager.addfont(str(_f.get_file()))
for _weight in (600, 700, 800):
    _f = load_google_font("Work Sans", weight=_weight)
    fm.fontManager.addfont(str(_f.get_file()))

## Set Styling
# Plot Style
pl_white = "#FFFFFF"
pl_background = "#292C42"
pl_text = "#55e8ff"  #'#72CBFD'
pl_line_color = "#8D96B3"
pl_highlight = "#F1C647"
pl_highlight_gradient = ["#F1C647", "#F5A05E"]
pl_highlight_cmap = sns.color_palette(
    f"blend:{pl_highlight_gradient[0]},{pl_highlight_gradient[1]}", as_cmap=True
)

surface_1 = "#13152A"
surface_3 = "#21243A"
cyan = "#55e8ff"
cyan_header = "#73efff"
text_cell = "#f0f1f5"
text_team = "#B4BBD7"
border_table = "#768ABE"
red = "#f4707c"
gold = "#f5b950"

sns.set_theme(
    style={
        "axes.edgecolor": pl_line_color,
        "axes.facecolor": pl_white,
        "axes.labelcolor": pl_white,
        "xtick.color": pl_line_color,
        "ytick.color": pl_line_color,
        "figure.facecolor": pl_white,
        "grid.color": pl_background,
        "grid.linestyle": "-",
        "legend.facecolor": pl_background,
        "text.color": "k",
    },
    font=base_font,
)
mpl.rcParams.update({"font.weight": 700})

st.set_page_config(page_title="MLB Excitement Stats", page_icon=letter_logo)

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700;9..40,900&family=Work+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --surface-1:#13152A; --surface-2:#181A2C; --surface-3:#21243A;
  --cyan:#55e8ff; --cyan-header:#73efff; --amber:#ffc277; --amber-muted:rgba(255,194,119,0.82);
  --text-nav:#cbd7fd; --text-cell:#f0f1f5; --text-team:rgba(180,187,215,0.68);
  --border-tab:rgba(255,255,255,0.13); --border-table:rgba(118,138,190,0.22);
  --border-row:rgba(140,165,210,0.11); --border-thead:rgba(157,244,255,0.22);
  --row-a:rgba(48,58,97,0.46); --row-b:rgba(42,51,85,0.38);
  --gold:#f5b950;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"]{
  background-color:var(--surface-1);
  background-image:url("data:image/svg+xml,%3Csvg%20width%3D%222364%22%20height%3D%222589%22%20viewBox%3D%220%200%202364%202589%22%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%3E%0A%20%20%3Cdefs%3E%0A%20%20%20%20%3ClinearGradient%20id%3D%22base%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%222364%22%20y2%3D%222589%22%20gradientUnits%3D%22userSpaceOnUse%22%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%220%25%22%20stop-color%3D%22%23171A2B%22/%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%2248%25%22%20stop-color%3D%22%2321243A%22/%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%22100%25%22%20stop-color%3D%22%23181B2C%22/%3E%0A%20%20%20%20%3C/linearGradient%3E%0A%0A%20%20%20%20%3CradialGradient%20id%3D%22r1%22%20cx%3D%220%22%20cy%3D%220%22%20r%3D%221%22%20gradientUnits%3D%22userSpaceOnUse%22%0A%20%20%20%20%20%20gradientTransform%3D%22translate%28425.52%20207.12%29%20scale%281702.08%201190.94%29%22%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%220%25%22%20stop-color%3D%22%23BED7FF%22%20stop-opacity%3D%220.085%22/%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%2252%25%22%20stop-opacity%3D%220%22/%3E%0A%20%20%20%20%3C/radialGradient%3E%0A%0A%20%20%20%20%3CradialGradient%20id%3D%22r2%22%20cx%3D%220%22%20cy%3D%220%22%20r%3D%221%22%20gradientUnits%3D%22userSpaceOnUse%22%0A%20%20%20%20%20%20gradientTransform%3D%22translate%281796.64%20466.02%29%20scale%281371.12%20984.82%29%22%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%220%25%22%20stop-color%3D%22%2378A5EB%22%20stop-opacity%3D%220.055%22/%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%2254%25%22%20stop-opacity%3D%220%22/%3E%0A%20%20%20%20%3C/radialGradient%3E%0A%0A%20%20%20%20%3CradialGradient%20id%3D%22r3%22%20cx%3D%220%22%20cy%3D%220%22%20r%3D%221%22%20gradientUnits%3D%22userSpaceOnUse%22%0A%20%20%20%20%20%20gradientTransform%3D%22translate%28472.8%202278.32%29%20scale%281654.8%201346.28%29%22%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%220%25%22%20stop-color%3D%22%2391B9F5%22%20stop-opacity%3D%220.06%22/%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%2256%25%22%20stop-opacity%3D%220%22/%3E%0A%20%20%20%20%3C/radialGradient%3E%0A%0A%20%20%20%20%3CradialGradient%20id%3D%22r4%22%20cx%3D%220%22%20cy%3D%220%22%20r%3D%221%22%20gradientUnits%3D%22userSpaceOnUse%22%0A%20%20%20%20%20%20gradientTransform%3D%22translate%282080.32%202122.98%29%20scale%281087.44%201605.18%29%22%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%220%25%22%20stop-color%3D%22%23D2E4FF%22%20stop-opacity%3D%220.052%22/%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%2255%25%22%20stop-opacity%3D%220%22/%3E%0A%20%20%20%20%3C/radialGradient%3E%0A%0A%20%20%20%20%3ClinearGradient%20id%3D%22l1%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%222364%22%20y2%3D%222589%22%0A%20%20%20%20%20%20gradientUnits%3D%22userSpaceOnUse%22%20gradientTransform%3D%22rotate%28-17%201182%201294.5%29%22%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%220%25%22%20stop-color%3D%22%23FFFFFF%22%20stop-opacity%3D%220.045%22/%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%2218%25%22%20stop-opacity%3D%220%22/%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%2238%25%22%20stop-color%3D%22%2387AFF0%22%20stop-opacity%3D%220.052%22/%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%2261%25%22%20stop-opacity%3D%220%22/%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%22100%25%22%20stop-color%3D%22%23D2E4FF%22%20stop-opacity%3D%220.04%22/%3E%0A%20%20%20%20%3C/linearGradient%3E%0A%0A%20%20%20%20%3ClinearGradient%20id%3D%22l2%22%20x1%3D%220%22%20y1%3D%222589%22%20x2%3D%222364%22%20y2%3D%220%22%0A%20%20%20%20%20%20gradientUnits%3D%22userSpaceOnUse%22%20gradientTransform%3D%22rotate%289%201182%201294.5%29%22%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%220%25%22%20stop-opacity%3D%220%22/%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%2228%25%22%20stop-color%3D%22%236E91D7%22%20stop-opacity%3D%220.045%22/%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%2249%25%22%20stop-opacity%3D%220%22/%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%2273%25%22%20stop-color%3D%22%23E6F0FF%22%20stop-opacity%3D%220.032%22/%3E%0A%20%20%20%20%20%20%3Cstop%20offset%3D%22100%25%22%20stop-opacity%3D%220%22/%3E%0A%20%20%20%20%3C/linearGradient%3E%0A%20%20%3C/defs%3E%0A%0A%20%20%3Crect%20width%3D%222364%22%20height%3D%222589%22%20fill%3D%22url%28%23base%29%22/%3E%0A%20%20%3Crect%20width%3D%222364%22%20height%3D%222589%22%20fill%3D%22url%28%23l2%29%22/%3E%0A%20%20%3Crect%20width%3D%222364%22%20height%3D%222589%22%20fill%3D%22url%28%23l1%29%22/%3E%0A%20%20%3Crect%20width%3D%222364%22%20height%3D%222589%22%20fill%3D%22url%28%23r4%29%22/%3E%0A%20%20%3Crect%20width%3D%222364%22%20height%3D%222589%22%20fill%3D%22url%28%23r3%29%22/%3E%0A%20%20%3Crect%20width%3D%222364%22%20height%3D%222589%22%20fill%3D%22url%28%23r2%29%22/%3E%0A%20%20%3Crect%20width%3D%222364%22%20height%3D%222589%22%20fill%3D%22url%28%23r1%29%22/%3E%0A%3C/svg%3E%0A");
  background-size:cover; background-position:center top; background-repeat:no-repeat; background-attachment:fixed;
}
[data-testid="stHeader"]{background:transparent;}
html, body, [data-testid="stAppViewContainer"]{color:var(--text-cell); font-family:'DM Sans',sans-serif;}
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3{
  font-family:'DM Sans',sans-serif; font-weight:900; color:var(--cyan);
}
[data-testid="stMarkdownContainer"] a{color:var(--cyan-header);}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li{color:var(--text-nav);}
/* Date input — styled to match the Dropdown trigger for visual pairing with
   the selectbox next to it. st.date_input is still a BaseWeb input (not
   react-aria), so it's targeted via data-baseweb — verified against the live
   DOM, not guessed. */
[data-testid="stDateInput"] div[data-baseweb="input"]{
  position:relative; height:36px; border-radius:8px;
  background:linear-gradient(144.79deg,#245297 0%,#0a2e63 100%) !important;
  box-shadow:0 6px 20px rgba(0,0,0,0.36);
  border:none !important; overflow:hidden;
}
[data-testid="stDateInput"] div[data-baseweb="input"]::after{
  content:''; position:absolute; inset:0; border-radius:inherit; pointer-events:none;
  box-shadow:inset 0 1px 0 rgba(255,255,255,0.3);
}
[data-testid="stDateInput"] div[data-baseweb="input"]:focus-within{
  border-radius:8px 8px 0 0;
}
[data-testid="stDateInput"] div[data-baseweb="base-input"]{
  background:transparent !important; height:100%;
}
[data-testid="stDateInputField"]{
  background:transparent !important; border:none !important; box-shadow:none !important;
  color:var(--text-cell) !important; font:700 14px/1.2 'Work Sans',sans-serif !important;
  padding-left:13px !important; height:100% !important;
}
[data-testid="stDateInputField"]:focus-visible{outline:none;}
[data-baseweb="popover"]{
  background:linear-gradient(118.49deg,#245297 0%,#0a2e63 100%) !important;
  box-shadow:0 0 0 1px rgba(0,80,100,0.4), 0 6px 20px rgba(0,0,0,0.36) !important;
  border-radius:8px !important; border:none !important;
}
[data-baseweb="calendar"]{background:transparent !important;}
/* Dropdown — st.selectbox styled to the guide's "Dropdowns" component.
   Streamlit (1.60+) renders this as a react-aria ComboBox, not the older
   BaseWeb Select, so it's targeted by role/data-testid rather than
   data-baseweb — verified against the live DOM, not guessed. */
[data-testid="stSelectbox"] div[role="group"]{
  position:relative; height:36px; border-radius:8px;
  background:linear-gradient(144.79deg,#245297 0%,#0a2e63 100%) !important;
  box-shadow:0 6px 20px rgba(0,0,0,0.36);
  border:none !important; overflow:hidden;
}
[data-testid="stSelectbox"] div[role="group"]::after{
  content:''; position:absolute; inset:0; border-radius:inherit; pointer-events:none;
  box-shadow:inset 0 1px 0 rgba(255,255,255,0.3);
}
[data-testid="stSelectbox"] div[role="group"]:has(input[aria-expanded="true"]){
  border-radius:8px 8px 0 0;
}
[data-testid="stSelectbox"] input[role="combobox"]{
  background:transparent !important; border:none !important; box-shadow:none !important;
  color:var(--text-cell) !important; font:700 14px/1.2 'Work Sans',sans-serif !important;
  padding-left:13px !important;
}
[data-testid="stSelectbox"] input[role="combobox"]:focus-visible{outline:none;}
[data-testid="stSelectbox"] button[aria-haspopup="listbox"]{
  background:transparent !important; border:none !important;
}
[data-testid="stSelectbox"] button[aria-haspopup="listbox"] svg{
  color:#fff; fill:#fff; width:14px; height:14px;
}
[data-testid="stSelectboxVirtualDropdown"]{
  background:linear-gradient(118.49deg,#245297 0%,#0a2e63 100%) !important;
  box-shadow:0 0 0 1px rgba(0,80,100,0.4), 0 6px 20px rgba(0,0,0,0.36) !important;
  border-radius:0 0 8px 8px !important; border:none !important;
}
[data-testid="stSelectboxVirtualDropdown"]::after{
  content:''; position:absolute; inset:0; border-radius:inherit; pointer-events:none;
  box-shadow:inset 0 1px 0 rgba(255,255,255,0.3);
}
[data-testid="stSelectboxVirtualDropdown"] [role="listbox"]{
  padding-top:9px; padding-bottom:9px; background:transparent !important;
}
[data-testid="stSelectboxVirtualDropdown"] [role="option"] [data-item-hl]{
  font:400 12px/1.2 'Work Sans',sans-serif; color:var(--text-cell) !important;
  padding-left:14px; height:100%; display:flex; align-items:center;
  background:transparent !important;
}
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused="true"]{
  background:linear-gradient(154.81deg,#4071ba 0%,#204b8c 100%) !important;
}
*:focus-visible{outline:2px solid var(--cyan); outline-offset:2px;}
.plpd-table-wrap{
  border:1px solid var(--border-table); border-radius:10px; overflow:hidden; position:relative;
  box-shadow:0 0 0 1px rgba(7,9,18,0.3), 0 18px 30px rgba(0,0,0,0.28);
}
table.plpd-data{width:100%; border-collapse:collapse; font-family:'Work Sans',sans-serif;}
table.plpd-data th{
  background:var(--surface-3, #2E3658); background-color:#2E3658;
  font:600 12px 'Work Sans',sans-serif; color:var(--cyan-header);
  height:34.5px; text-align:right; padding:0 14px; white-space:nowrap;
  border-bottom:1px solid var(--border-thead);
}
table.plpd-data th.l{text-align:left;}
table.plpd-data td{
  font:400 14px 'Work Sans',sans-serif; color:var(--text-cell);
  text-align:right; padding:0 14px; height:44px;
  border-bottom:1px solid var(--border-row); font-variant-numeric:tabular-nums;
}
table.plpd-data td.l{text-align:left;}
table.plpd-data tr:nth-child(odd) td{background:var(--row-a);}
table.plpd-data tr:nth-child(even) td{background:var(--row-b);}
table.plpd-data td.gold{color:var(--gold); font-weight:700;}
</style>
""",
    unsafe_allow_html=True,
)

new_title = '<p style="color:#55e8ff; font-weight: bold; font-size: 42px; text-align:center;">MLB Excitement Index</p>'
st.markdown(new_title, unsafe_allow_html=True)

color_df = pl.read_csv(
    "https://github.com/Blandalytics/PLV_viz/blob/main/mlb_team_colors.csv?raw=true"
)
color_dict = color_df[["Short Code", "Color 1"]].rows_by_key(
    key=["Short Code"], unique=True
)
logo_dict = color_df[["Short Code", "Logo"]].rows_by_key(
    key=["Short Code"], unique=True
)

col1, col2 = st.columns(2)
with col1:
    today = (datetime.datetime.now(pytz.utc) - timedelta(hours=16)).date()
    date = st.date_input(
        "Select a game date:",
        today,
        min_value=datetime.date(2020, 3, 28),
        max_value=today,
    )


def fetch_game_ids(date):
    r = requests.get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}")
    x = r.json()
    if x["totalGames"] == 0:
        date_list = []
    else:
        date_list = []
        for game in x["dates"][0]["games"]:
            if "rescheduleGameDate" in list(game.keys()):
                if game["rescheduleGameDate"] != date:
                    continue
            if game["status"]["detailedState"] in ["Pre-Game", "Warmup"]:
                continue
            date_list += [game["gamePk"]]
    return date_list


def fetch_pitches(game_pk):
    df_list = [
        pl.DataFrame(
            {"game_pk": game_pk, "ab_number": 0, "game_total_pitches": 0, "outs": 0}
        )
    ]
    r = requests.get(f"https://baseballsavant.mlb.com/gf?game_pk={game_pk}")
    x = r.json()

    for team in ["home", "away"]:
        if f"{team}_pitchers" not in x.keys():
            continue
        if "scoreboard" not in x.keys():
            continue
        home_code = x["scoreboard"]["teams"]["home"]["teamName"]
        home_abbrev = x["home_team_data"]["abbreviation"]
        home_score = x["boxscore"]["teams"]["home"]["teamStats"]["batting"]["runs"]
        away_code = x["scoreboard"]["teams"]["away"]["teamName"]
        away_abbrev = x["away_team_data"]["abbreviation"]
        away_score = x["boxscore"]["teams"]["away"]["teamStats"]["batting"]["runs"]
        for pitcher_id in list(x[f"{team}_pitchers"].keys()):
            df_list.append(
                pl.DataFrame(
                    x[f"{team}_pitchers"][pitcher_id], strict=False
                ).with_columns(
                    pl.lit(home_code).alias("home_team"),
                    pl.lit(away_code).alias("away_team"),
                    pl.lit(home_abbrev).alias("home_abbrev"),
                    pl.lit(away_abbrev).alias("away_abbrev"),
                    pl.lit(home_score).alias("home_score"),
                    pl.lit(away_score).alias("away_score"),
                    pl.lit(date.year).alias("year_played"),
                )
            )
    if not df_list:
        pitches = pl.DataFrame(df_list)
    else:
        pitches = pl.concat(df_list, how="diagonal_relaxed").with_columns(
            pl.col("game_pk").cast(pl.Int32), pl.col("ab_number").cast(pl.Int32)
        )
    return pitches


def fetch_win_prob(game_pk):
    wp_list = [
        pl.DataFrame(
            {
                "game_pk": game_pk,
                "ab_number": 0,
                "homeTeamWinProbability": 50,
                "awayTeamWinProbability": 50,
                "hwp": 0,
                "awp": 0,
                "homeTeamWinProbabilityAdded": 0,
            }
        )
    ]
    r = requests.get(f"https://baseballsavant.mlb.com/gf?game_pk={game_pk}")
    x = r.json()

    wp_list.append(
        pl.DataFrame(x["scoreboard"]["stats"]["wpa"]["gameWpa"])
        .with_row_index("ab_number", offset=1)
        .with_columns(pl.lit(game_pk).alias("game_pk"))
    )
    if not wp_list:
        win_probs = pl.DataFrame(wp_list)
    else:
        win_probs = pl.concat(wp_list, how="diagonal_relaxed").with_columns(
            pl.col("game_pk").cast(pl.Int32), pl.col("ab_number").cast(pl.Int32)
        )
    return win_probs


def merge_dfs(game_pk):
    pitch_df = fetch_pitches(game_pk)
    wpa_df = fetch_win_prob(game_pk)
    if pitch_df.shape[0] == 0:
        combined_df = pl.DataFrame()
    else:
        combined_df = (
            pitch_df.join(wpa_df, how="inner", on=["game_pk", "ab_number"])
            .sort("game_total_pitches")
            .group_by(["game_pk", "ab_number"])
            .agg(
                pl.last(
                    [
                        "home_team",
                        "home_abbrev",
                        "home_score",
                        "away_team",
                        "away_abbrev",
                        "away_score",
                        "inning",
                        "outs",
                        "homeTeamWinProbabilityAdded",
                        "homeTeamWinProbability",
                        "awayTeamWinProbability",
                        "events",
                    ]
                )
            )
            .sort(["game_pk", "ab_number"])
            .with_columns(
                pl.concat_str(
                    [
                        pl.col("away_team"),
                        pl.lit("@"),
                        pl.col("home_team"),
                        pl.lit("-"),
                        pl.col("game_pk"),
                    ],
                    separator=" ",
                ).alias("game_name"),
                (
                    pl.when(pl.col("outs").shift(1) < 3)
                    .then(pl.col("outs") - pl.col("outs").shift(1, fill_value=0))
                    .otherwise(pl.col("outs"))
                    .alias("outs_made")
                ),
                (
                    pl.when(pl.col("events").str.contains("Double Play|GIDP"))
                    .then(pl.lit(2))
                    .when(pl.col("events").str.contains("Triple Play"))
                    .then(pl.lit(3))
                    .when(pl.col("events").str.contains("Out|out|Caught|Sac|Pickoff"))
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .alias("ab_outs")
                ),
            )
            .with_columns(
                pl.col("ab_outs").cum_sum().over("game_name").alias("game_outs"),
                (
                    -(
                        pl.max_horizontal(
                            "homeTeamWinProbability", "awayTeamWinProbability"
                        )
                        / 100
                        * np.log2(
                            pl.max_horizontal(
                                "homeTeamWinProbability", "awayTeamWinProbability"
                            )
                            / 100
                        ).replace(float("-inf"), 0)
                    )
                    - (
                        pl.min_horizontal(
                            "homeTeamWinProbability", "awayTeamWinProbability"
                        )
                        / 100
                        * np.log2(
                            pl.min_horizontal(
                                "homeTeamWinProbability", "awayTeamWinProbability"
                            )
                            / 100
                        ).replace(float("-inf"), 0)
                    )
                ).alias("tension"),
                (
                    (
                        pl.col("homeTeamWinProbability")
                        / 100
                        * np.log2(
                            pl.col("homeTeamWinProbability")
                            / pl.col("homeTeamWinProbability").shift(1)
                        ).replace(float("-inf"), 0)
                        + pl.col("awayTeamWinProbability")
                        / 100
                        * np.log2(
                            pl.col("awayTeamWinProbability")
                            / pl.col("awayTeamWinProbability").shift(1)
                        ).replace(float("-inf"), 0)
                    )
                ).alias("k_l_excite"),
            )
        )
    return combined_df


def threaded_data(game_list_input):
    games_data = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(merge_dfs, game_pk): game_pk for game_pk in game_list_input
        }
        for future in as_completed(futures):
            games_data.append(future.result())
    combined_df = pl.concat(games_data, how="diagonal_relaxed")
    return combined_df.filter(pl.col("game_name").is_not_null())


def game_table(win_prob_df):
    play_df = (
        win_prob_df.group_by(
            [
                "game_name",
                "game_pk",
                "home_team",
                "away_team",
                "home_abbrev",
                "away_abbrev",
                "inning",
                "ab_number",
                "events",
            ]
        )
        .agg(
            pl.col("k_l_excite").last(),
            pl.col("tension").last(),
            pl.col("homeTeamWinProbabilityAdded").sum(),
            pl.col("homeTeamWinProbability").last().alias("home_win_prob"),
            pl.col("home_score").last(),
            pl.col("away_score").last(),
            pl.col("game_outs").last(),
        )
        .sort(["game_pk", "ab_number"])
    )

    swing_df = (
        win_prob_df.group_by(
            [
                "game_name",
                "game_pk",
                "home_team",
                "away_team",
                "home_abbrev",
                "away_abbrev",
                "game_outs",
            ]
        )
        .agg(pl.col("homeTeamWinProbability").last().alias("home_win_prob"))
        .sort(["game_pk", "game_outs"])
        .rolling(
            index_column="game_outs",
            period="6i",
            group_by=[
                "game_name",
                "home_team",
                "away_team",
                "home_abbrev",
                "away_abbrev",
            ],
        )
        .agg(
            pl.max("home_win_prob").alias("rolling_max_prob"),
            pl.min("home_win_prob").alias("rolling_min_prob"),
        )
        .with_columns(
            (pl.col("rolling_max_prob") - pl.col("rolling_min_prob")).alias("win_swing")
        )
    )

    agg_df = (
        play_df.group_by(["game_name", "game_pk"])
        .agg(
            pl.col("game_outs").max(),
            pl.col("k_l_excite").abs().sum(),
            pl.col("tension").mean(),
        )
        .with_columns(
            (pl.col("tension") * 100).alias("tension_adj"),
            (2 ** pl.col("k_l_excite")).alias("k_l_excite_adj"),
        )
    )
    agg_df = (
        pl.concat(
            [
                agg_df,
                swing_df.group_by("game_name").agg(
                    pl.col("win_swing").max().alias("win_swing")
                ),
            ],
            how="align_inner",
        )
        .with_columns(
            ((pl.col("k_l_excite") - 0.25) / 2).alias("excite_scale"),
            ((pl.col("tension") - 0.35) / 0.575).alias("tension_scale"),
            ((pl.col("win_swing") - 21) / 60).alias("swing_scale"),
        )
        .with_columns(
            (
                (
                    (
                        pl.col("excite_scale").clip(0, 2.5)
                        + pl.col("tension_scale").clip(0, 2.5)
                        + pl.col("swing_scale").clip(0, 2.5)
                    )
                    / 3
                )
                ** (0.75)
                * 10
            ).alias("watch_scale")
        )
        .with_columns(pl.col("watch_scale").clip(0, 10).alias("watch_score"))
    )

    return play_df, agg_df.sort("watch_scale", descending=True)


days_games = fetch_game_ids(date)
game_df, table_df = game_table(threaded_data(days_games))

if table_df.shape[0] == 0:
    st.write("No games played")
    st.stop()

with col2:
    game_list = table_df["game_name"].to_list()
    game_choice = st.selectbox("Choose a game:", game_list)
    game_choice_id = table_df.row(by_predicate=(pl.col("game_name") == game_choice))[1]


def game_chart(game_choice_id):
    r = requests.get(f"https://baseballsavant.mlb.com/gf?game_pk={game_choice_id}")
    x = r.json()
    game_date = x[
        "gameDate"
    ]  # datetime.datetime.strptime(x['gameDate'],'%Y-%m-%d').strftime('%-m/%-d/%y')
    hline_y = 0.5
    single_game_df = game_df.filter(pl.col("game_pk") == game_choice_id)
    home_name = single_game_df["home_team"][0]
    away_name = single_game_df["away_team"][0]
    home_abbr = single_game_df["home_abbrev"][0]
    away_abbr = single_game_df["away_abbrev"][0]
    home_score = single_game_df["home_score"][0]
    away_score = single_game_df["away_score"][0]
    home_color = (
        "k" if color_dict[home_abbr][0] == "#FFFFFF" else color_dict[home_abbr][0]
    )
    away_color = (
        "k" if color_dict[away_abbr][0] == "#FFFFFF" else color_dict[away_abbr][0]
    )

    # Add start row for 50%
    append_row = single_game_df[0]
    append_row = append_row.with_columns(
        pl.lit(-1).alias("ab_number"),
        pl.lit(50).alias("home_win_prob"),
        pl.lit(0).alias("homeTeamWinProbabilityAdded"),
    )
    single_game_df = pl.concat([append_row, single_game_df], how="vertical_relaxed")
    x = single_game_df.select(pl.col("ab_number")).to_numpy().ravel()
    y = single_game_df.select(pl.col("home_win_prob")).to_numpy().ravel() / 100

    excite_index = (
        table_df.filter(pl.col("game_pk") == game_choice_id)["excite_scale"].clip(0, 1)[
            0
        ]
        * 10
    )
    tension_index = (
        table_df.filter(pl.col("game_pk") == game_choice_id)["tension_scale"].clip(
            0, 1
        )[0]
        * 10
    )
    biggest_swing = table_df.filter(pl.col("game_pk") == game_choice_id)["win_swing"][0]
    win_swing_index = (
        table_df.filter(pl.col("game_pk") == game_choice_id)["swing_scale"].clip(0, 1)[
            0
        ]
        * 10
    )
    watch_index = table_df.filter(pl.col("game_pk") == game_choice_id)[
        "watch_score"
    ].clip(0, 10)[0]

    game_abs = max(x)
    # chart_outs = 54 if game_outs <51 else game_outs

    # Create a figure and plot the line on it
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(
        1.005,
        color=home_color,
        alpha=1,
        xmin=(game_abs / 5) / (game_abs + 1.25),
        xmax=(game_abs + 1) / (game_abs + 1.25),
    )
    ax.axhline(
        -0.005,
        color=away_color,
        alpha=1,
        xmin=(game_abs / 5) / (game_abs + 1.25),
        xmax=(game_abs + 1) / (game_abs + 1.25),
    )
    inning_text_dict = (
        game_df.filter(pl.col("game_pk") == game_choice_id)
        .group_by("inning")
        .agg(pl.median("ab_number"), pl.max("ab_number").alias("ab_max"))
        .sort("inning")
        .to_dict(as_series=False)
    )
    for inning in inning_text_dict["inning"]:
        ab_number = inning_text_dict["ab_number"][
            inning_text_dict["inning"].index(inning)
        ]
        ab_max = inning_text_dict["ab_max"][inning_text_dict["inning"].index(inning)]
        ax.text(
            ab_number - 0.5,
            0.5,
            inning,
            ha="center",
            va="center",
            color=pl_background,
            bbox=dict(
                boxstyle="round", facecolor="w", alpha=0.75, edgecolor=pl_background
            ),
        )
        if inning != inning_text_dict["inning"][-1]:
            ax.axvline(
                ab_max,
                linestyle="--",
                alpha=0.25,
                ymin=(0.25 + 0.1) / 1.5,
                ymax=(0.75 + 0.1) / 1.5,
                color=pl_background,
            )

    ax.axhline(0.5, color=pl_background, alpha=0.5)

    sns.lineplot(x=x, y=y, color=pl_background)
    verts = np.column_stack([x, y]).tolist()
    verts += [[x[-1], hline_y], [x[0], hline_y]]
    clip_path = Path(verts + [verts[0]])
    clip_patch = PathPatch(
        clip_path, transform=ax.transData, facecolor="none", edgecolor="none"
    )
    ax.add_patch(clip_patch)

    # Gradient image spanning the fill's vertical range, colored by y-value
    # (i.e. distance from the reference line), then clipped to that polygon.
    y_min = 0
    y_max = 1
    gradient = np.linspace(y_min, y_max, 256).reshape(-1, 1)

    im = ax.imshow(
        gradient,
        extent=[x[0], x[-1], y_min, y_max],
        origin="lower",
        aspect="auto",
        cmap=sns.blend_palette([away_color, "w", home_color], as_cmap=True),
        vmin=y_min,
        vmax=y_max,
        zorder=1,
    )
    im.set_clip_path(clip_patch)

    ax.set(xlim=(-1, game_abs + 0.25), ylim=(1.1, -0.4))
    ax.axis("off")

    excite_ax = fig.add_axes([0.82, 0.8, 0.1, 0.1], anchor="NE", zorder=1)
    excite_ax.text(0, 0.9, "Excitement\nIndex", ha="center", va="center", fontsize=14)
    if round(watch_index, 1) >= 10:
        excite_ax.text(
            0,
            -0.4,
            f"{watch_index:.0f}",
            ha="center",
            va="center",
            size=18,
            color="w",
            bbox=dict(
                boxstyle="circle",
                pad=0.3,
                fc=sns.color_palette("vlag", n_colors=1001)[-1],
                ec=pl_background,
            ),
        )
    else:
        excite_ax.text(
            0,
            -0.4,
            f"{watch_index:.1f}",
            ha="center",
            va="center",
            size=18,
            color="k" if abs(watch_index - 5) < 2.5 else "w",
            bbox=dict(
                boxstyle="circle",
                pad=0.3,
                fc=sns.color_palette("vlag", n_colors=1001)[int(watch_index * 100)],
                ec=pl_background,
            ),
        )
    excite_ax.axis("off")

    home_team_ax = fig.add_axes([0.12, 0.115, 0.1, 0.12], anchor="NW", zorder=1)
    image = load_team_logo(home_abbr)
    home_team_ax.imshow(image, aspect="equal")
    home_team_ax.axis("off")
    ax.text(
        (game_abs / 7),
        1,
        f" {home_score:.0f} ",
        color=pl_highlight if home_score > away_score else "k",
        fontsize=30,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round",
            pad=0,
            fc="w",
            linewidth=2,
            ec=home_color if home_score > away_score else "w",
        ),
    )

    away_team_ax = fig.add_axes([0.12, 0.625, 0.1, 0.12], anchor="NW", zorder=1)
    image = load_team_logo(away_abbr)
    away_team_ax.imshow(image, aspect="equal")
    away_team_ax.axis("off")
    ax.text(
        (game_abs / 7),
        0,
        f" {away_score:.0f} ",
        color=pl_highlight if away_score > home_score else "k",
        fontsize=30,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round",
            pad=0,
            fc="w",
            linewidth=2,
            ec=away_color if away_score > home_score else "w",
        ),
    )

    fig.suptitle(f"{away_name} @ {home_name}", fontsize=25, x=0.415, y=0.92)
    fig.text(
        0.415,
        0.77,
        f"{game_date} - Game ID: {game_choice_id:.0f}",
        fontsize=12,
        ha="center",
    )
    fig.text(0.375, 0.1, "Volatility", ha="center", fontsize=16)
    fig.text(
        0.375,
        0.015,
        (
            f"{excite_index:.0f}"
            if round(excite_index, 1) >= 10
            else f"{excite_index:.1f}"
        ),
        ha="center",
        fontsize=16,
        color="k" if abs(excite_index - 5) < 2.5 else "w",
        bbox=dict(
            boxstyle="round",
            pad=0.25,
            fc=sns.color_palette("vlag", n_colors=1001)[
                int(np.clip(excite_index * 100, 0, 1000))
            ],
            ec="k",
        ),
    )

    fig.text(0.55, 0.1, "Tension", ha="center", fontsize=16)
    fig.text(
        0.55,
        0.015,
        (
            f"{tension_index:.0f}"
            if round(tension_index, 1) >= 10
            else f"{tension_index:.1f}"
        ),
        ha="center",
        fontsize=16,
        color="k" if abs(tension_index - 5) < 2.5 else "w",
        bbox=dict(
            boxstyle="round",
            pad=0.25,
            fc=sns.color_palette("vlag", n_colors=1001)[
                int(np.clip(tension_index * 100, 0, 1000))
            ],
            ec="k",
        ),
    )

    fig.text(0.775, 0.1, "Biggest Swing", ha="center", fontsize=16)
    fig.text(
        0.775,
        0.015,
        f"{biggest_swing:.0f}%",
        ha="center",
        fontsize=16,
        color="k" if abs(win_swing_index - 5) < 2.5 else "w",
        bbox=dict(
            boxstyle="round",
            pad=0.25,
            fc=sns.color_palette("vlag", n_colors=1001)[
                int(np.clip(win_swing_index * 100, 0, 1000))
            ],
            ec="k",
        ),
    )
    fig.text(0.55, 0.69, "Game Win Probability", ha="center", fontsize=12)
    fig.text(0.14, 0.02, "Data: MLB", ha="left", fontsize=10)

    # Add PL logo
    # pl_ax = fig.add_axes([0.675,0.04,0.2,0.1], anchor='NE', zorder=1)
    # pl_ax.set_facecolor(pl_background)
    # # width, height = logo.size
    # # pl_ax.imshow(logo.crop((0, 0, width, height-150)))
    # pl_ax.imshow(logo)
    # pl_ax.axis('off')

    sns.despine()
    st.pyplot(fig)


game_chart(game_choice_id)

st.header(f"Excitement stats for {date:%-m/%-d/%y} games:")

results_df = (
    table_df.with_columns(
        (pl.col("excite_scale").clip(0, 1).round(2) * 10).alias("Volatility"),
        (pl.col("tension_scale").clip(0, 1).round(2) * 10).alias("Tension"),
        (pl.col("watch_score").round(1)).alias("Excitement Index"),
    )
    .rename({"game_name": "Game", "win_swing": "Biggest Swing"})[
        ["Game", "Volatility", "Tension", "Biggest Swing", "Excitement Index"]
    ]
    .sort("Excitement Index", descending=True)
)

# Rendered as the style guide's table.data pattern (right-aligned tabular numerics,
# translucent zebra rows) rather than st.dataframe, whose canvas-drawn grid can't
# be restyled to match.
table_rows = "".join(f"""<tr>
        <td class="l">{html.escape(str(row['Game']))}</td>
        <td>{row['Volatility']:.1f}</td>
        <td>{row['Tension']:.1f}</td>
        <td>{row['Biggest Swing']:.0f}%</td>
        <td class="gold">{row['Excitement Index']:.1f}</td>
    </tr>""" for row in results_df.iter_rows(named=True))
st.markdown(
    f"""
<div class="plpd-table-wrap">
  <table class="plpd-data">
    <thead>
      <tr>
        <th class="l">Game</th><th>Volatility</th><th>Tension</th><th>Biggest Swing</th><th>Excitement Index</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>
""",
    unsafe_allow_html=True,
)

# st.dataframe(
#     table_df.with_columns(
#         (pl.col("excite_scale").clip(0, 1).round(2) * 10).alias("Volatility"),
#         (pl.col("tension_scale").clip(0, 1).round(2) * 10).alias("Tension"),
#         (pl.col("watch_score").round(1)).alias("Excitement Index"),
#     )
#     .rename({"game_name": "Game", "win_swing": "Biggest Swing"})[
#         ["Game", "Volatility", "Tension", "Biggest Swing", "Excitement Index"]
#     ]
#     .sort("Excitement Index", descending=True)
# )

st.header("Glossary")
st.write(
    "Excitement Index quantifies how exciting a baseball game was, on a 0-10 scale, based on three components:"
)
st.write(
    "- [Volatility](https://inpredictable.substack.com/p/quantifying-excitement): how much did the win probability change throughout the game (0-10)"
)
st.write(
    "- [Tension](https://www.inpredictable.com/2020/04/an-update-to-tension-index-with-assist_11.html): how uncertain the outcome of the game was, on average (0-10)"
)
st.write("- Biggest Swing: largest 6-out swing in win probability")
