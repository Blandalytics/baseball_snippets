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

medal_df = load_data().rename(columns={'NOC':'Country','Total':'Medals'})

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

medal_df['weighted_Gold'] = medal_df['Gold'].copy()
medal_df['weighted_Silver'] = medal_df['Silver'].div(gold_vs_silver_weight)
medal_df['weighted_Bronze'] = medal_df['Bronze'].div(gold_vs_bronze_weight)
medal_df['Weighted Count'] = medal_df[['weighted_Gold','weighted_Silver','weighted_Bronze']].astype('float').sum(axis=1)
count_adjust = medal_df['Medals'].sum() / medal_df['Weighted Count'].sum()
for medal_count in ['weighted_Bronze','weighted_Silver','weighted_Gold','Weighted Count']:
  medal_df[medal_count] = medal_df[medal_count].mul(count_adjust)


cmap = colors.LinearSegmentedColormap.from_list('medal_colors', ['#977547','#d6d6d6','#fcb434'], N=3)

fig, ax = plt.subplots(figsize=(6,5))
medal_df.sort_values('Weighted Count',ascending=False).set_index('Country')[['weighted_Bronze','weighted_Silver','weighted_Gold']].head(5).round(2).plot(kind='bar', 
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
  
plt.xticks(rotation=0)

medal_dict = {
    'Gold':[fig.add_axes([0.165,.81,0.2,0.2], anchor='SW', zorder=2), 1, '#fcb434'],
    'Silver':[fig.add_axes([0.375,.81,0.2,0.2], anchor='SW', zorder=2), gold_vs_silver_weight, '#d6d6d6'],
    'Bronze':[fig.add_axes([0.6,.81,0.2,0.2], anchor='SW', zorder=2), gold_vs_silver_weight*silver_vs_bronze_weight, '#977547']
}
for color in medal_dict.keys():
    medal_ax = medal_dict[color][0]
    medals = medal_dict[color][1]
    row_medals = (int((medals-1)**0.5)+1)
    rows = (medals - medals%row_medals) / row_medals
    medal_x = [x for x in range(row_medals)] * int(rows) + [x+(row_medals-(medals%row_medals))/2 for x in range(medals%row_medals)]
    medal_y = [rows-int(y/row_medals) for y in range(len(medal_x))]
    sns.scatterplot(x=medal_x,
                    y=medal_y,
                   color=medal_dict[color][2],
                    linewidth=1/3,
                    edgecolor='#666666',
                   s=1000/(row_medals**2),
                   ax=medal_ax)
    medal_ax.set(xlim=(medal_ax.get_xlim()[0]-1.5,medal_ax.get_xlim()[1]+1.5),
                 ylim=(medal_ax.get_ylim()[0]-1.5,medal_ax.get_ylim()[1]+1.5),)
    medal_ax.axis('off')
    sns.despine(left=True, bottom=True)
ax.set(xlabel='',
       ylim=(0,ax.get_ylim()[1]*1.05),
      ylabel='Weighted Medals')
fig.suptitle('Weighted Olympic Medal Leaders',fontsize=18,y=1.05)
fig.text(0.325,0.95,'=       =',ha='left',va='top',fontsize=32)
fig.text(0.04,0.015,'Data: Wikipedia',ha='left',va='center',fontsize=8)
fig.text(0.5,0.015,'@Blandalytics\nolympic-medal-counts.streamlit.app',ha='center',va='center',fontsize=8)
fig.text(0.9,0.015,'* Host Nation',ha='right',va='center',fontsize=8)

sns.despine(left=True,bottom=True)
st.pyplot(fig)

st.dataframe(medal_df[['Country','Weighted Count','Gold','Silver','Bronze','Medals']].sort_values('Weighted Count',ascending=False).round({'Weighted Count':2}),
             hide_index=True,
             height=(medal_df.shape[0] + 1) * 35 + 3)   
