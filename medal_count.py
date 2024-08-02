import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import colors

st.set_page_config(page_title='Olympic Medal Counts', page_icon='🥇')
st.title('Olympic Medal Counts')

@st.cache_data(ttl=10*60,show_spinner="Loading medal data")
def load_data():
  tables=pd.read_html("https://en.wikipedia.org/wiki/2024_Summer_Olympics_medal_table")
  return tables[3].iloc[:-1].copy()

medal_df = load_data().rename(columns={'NOC':'Country','Total':'Medal Count'})

col1, col2 = st.columns(2)

with col1:
    gold_vs_silver_weight = st.slider(
      "How much more valuable is a Gold Medal vs a Silver Medal? (🥇 = x * 🥈)",
      min_value=1,
      max_value=25,
      value=2,
    )
with col2:
    silver_vs_bronze_weight =  st.slider(
      "How much more valuable is a Silver Medal vs a Bronze Medal? (🥈 = x * 🥉)",
      min_value=1,
      max_value=25,
      value=2
    )

gold_vs_bronze_weight = gold_vs_silver_weight * silver_vs_bronze_weight

medal_df['weighted_Silver'] = medal_df['Silver'].div(gold_vs_silver_weight)
medal_df['weighted_Bronze'] = medal_df['Bronze'].div(gold_vs_bronze_weight)
medal_df['Weighted Count'] = medal_df[['Gold','weighted_Silver','weighted_Bronze']].astype('float').sum(axis=1)
count_adjust = medal_df['Medal Count'].sum() / medal_df['Weighted Count'].sum()
for medal_count in ['weighted_Bronze','weighted_Silver','Gold','Weighted Count']:
  medal_df[medal_count] = medal_df[medal_count].mul(count_adjust)


cmap = colors.LinearSegmentedColormap.from_list('medal_colors', ['#977547','#d6d6d6','#fcb434'], N=3)

fig, ax = plt.subplots(figsize=(6,5))
medal_df.sort_values('Weighted Count',ascending=False).set_index('Country')[['weighted_Bronze','weighted_Silver','Gold']].head(5).round(1).plot(kind='bar', 
                                                                           cmap=cmap, 
                                                                            stacked=True,
                                                                                                                              edgecolor='w',
                                                                           ax=ax, linewidth=2,
                                                                           width=0.7,
                                                                                                                             legend=False)
for container in ax.containers[2::3]:
    ax.bar_label(container)
  
new_patches = []
for patch in reversed(ax.patches):
    bb = patch.get_bbox()
    color=patch.get_facecolor()
    p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                        abs(bb.width), abs(bb.height),
                        boxstyle="round,pad=-0.015,rounding_size=0.075",
                        ec="none", fc=color,
                        mutation_aspect=4
                        )
    patch.remove()
    new_patches.append(p_bbox)
for patch in new_patches:
    ax.add_patch(patch)

ax.set(xlabel='',
      ylabel='Weighted Medals')
fig.suptitle('Olympic Medal Leaders',fontsize=18)
fig.text(0.5,0.925,f'(Weights: Gold = {gold_vs_silver_weight}x Silver, Silver = {silver_vs_bronze_weight}x Bronze)',ha='center',va='top')
plt.xticks(rotation=0)
sns.despine(left=True,bottom=True)
st.pyplot(fig)

st.dataframe(medal_df[['Country','Gold','Silver','Bronze','Medal Count','Weighted Count']].sort_values('Weighted Count',ascending=False).round({'Weighted Count':1}),
             hide_index=True,
             height=(medal_df.shape[0] + 1) * 35 + 3)
