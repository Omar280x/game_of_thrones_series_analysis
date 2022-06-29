#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import os
import pandas as pd
import numpy as np
import plotly
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from dash import Dash, html, dcc
from dash.dependencies import Input,Output,State
import dash_bootstrap_components as dbc

# ## Series Ratings EDA

# In[ ]:


df_rating = pd.read_csv('dataset/GOT_episodes_v5.csv')
df_rating


# In[ ]:


df_rating["Episode"]= df_rating["Episode"].astype(str)
df_rating.info()


# In[ ]:


df_rating.loc[9, 'Episode'] = '1.10'
df_rating.loc[19, 'Episode'] = '2.10'
df_rating.loc[29, 'Episode'] = '3.10'
df_rating.loc[39, 'Episode'] = '4.10'
df_rating.loc[49, 'Episode'] = '5.10'
df_rating.loc[59, 'Episode'] = '6.10'


# In[ ]:


df_rating["Season"]= df_rating["Season"].astype(str)
fig1 = px.bar(df_rating, x='Episode', y='Rating', color='Season',  color_discrete_sequence=px.colors.qualitative.Dark24, template='plotly_dark')
fig1.update_layout(title_text='Series Episodes Ratings')
fig1.show()


# In[ ]:


df_rating["Season"]= df_rating["Season"].astype(int)


# In[ ]:


df_rating["Season"]= df_rating["Season"].astype(str)
fig2 = px.scatter(df_rating, x='Episode', y='US_Viewers', size="Rating", color="Season", size_max=30, template='plotly_dark')
fig2.update_layout(title_text='Series Views in USA')
fig2.show()


# In[ ]:


df_rating["Season"]= df_rating["Season"].astype(int)


# In[ ]:


df_rating["Season"]= df_rating["Season"].astype(str)
fig3 = px.bar(df_rating, x='Rating', y='Director', orientation='h', color='Season', template='plotly_dark')
fig3.update_layout(title_text='Episdoes Directors Ratings', barmode='stack', yaxis={'categoryorder':'total ascending'})
#fig3.update_traces(marker_color='rgb(250, 218, 5)',
#                  marker_line_width=1.5, opacity=1)
fig3.show()
df_rating["Season"]= df_rating["Season"].astype(int)


# ## Battles EDA

# In[ ]:


df_battle = pd.read_csv('dataset/battles.csv')
df_battle


# In[ ]:


df_battle.drop(['attacker_1','attacker_2','attacker_3','attacker_4','defender_1','defender_2','defender_3','defender_4','note'],axis=1).head()


# In[ ]:


df_battle['attacker_outcome_flag'] = df_battle['attacker_outcome'].map({'win': 1, 'loss': 0})
# Fill NaN with zero
df_battle['attacker_outcome_flag'] = df_battle['attacker_outcome_flag'].fillna(0)
df_battle['major_death'] = df_battle['major_death'].fillna(0)
df_battle['major_capture'] = df_battle['major_capture'].fillna(0)
df_battle['summer'] = df_battle['summer'].fillna(0)
df_battle['attacker_size'] = df_battle['attacker_size'].fillna(0)
df_battle['defender_size'] = df_battle['defender_size'].fillna(0)

df_battle['attack_houses'] = df_battle[['attacker_1','attacker_2','attacker_3','attacker_4']].notnull().sum(axis=1)
df_battle['attack_houses'] = pd.to_numeric(df_battle.attack_houses)
df_battle['defender_houses'] = df_battle[['defender_1','defender_2','defender_3','defender_4']].notnull().sum(axis=1)
df_battle['defender_houses'] = pd.to_numeric(df_battle.defender_houses)
# Count attacker_commander
df_battle['attacker_commander_count'] = df_battle['attacker_commander'].str.split(',', expand=True).notnull().sum(axis=1)
# Count defender_commander
df_battle['defender_commander_count'] = df_battle['defender_commander'].str.split(',', expand=True).notnull().sum(axis=1)
# Drop columns with missing data
df_battle = df_battle.drop(columns = ['battle_number','attacker_2','attacker_3','attacker_4','defender_2','defender_3','defender_4','note'])
# Create battle_size columns
df_battle['battle_size'] = df_battle['attacker_size'] + df_battle['defender_size']
df_battle['battle_size'] = df_battle['battle_size'].fillna(0)
df_battle.loc[37, 'attacker_outcome']='win'
df_battle = df_battle.drop([22, 29])


# In[ ]:


df_battle


# In[ ]:


fig4 = px.bar(df_battle, y='attacker_king', orientation='h', template='plotly_dark')
fig4.update_layout(title_text='Kings Battles Counts', title_x=0.5, yaxis=dict(autorange="reversed"))
fig4.update_traces(marker_color='rgb(250, 218, 5)',
                  marker_line_width=1.5, opacity=1)
fig4.show()


# In[ ]:


df_grouped = df_battle.groupby(by=['attacker_king']).agg(
    attacker_outcome_flag_count = ('attacker_outcome_flag','count'),
    attacker_outcome_wins = ('attacker_outcome_flag','sum'),
    attacker_size_mean = ('attacker_size', 'mean'),
    attacker_size_std = ('attacker_size', 'std'),
    defender_size_mean = ('defender_size','mean'),
    defender_size_std = ('defender_size','std')).reset_index().sort_values(by = 'attacker_outcome_flag_count', ascending = False)

df_grouped['attacker_outcome_loss'] = df_grouped['attacker_outcome_flag_count'] - df_grouped['attacker_outcome_wins']
df_grouped['attacker_outcome_wins_pct'] = (df_grouped['attacker_outcome_wins']/df_grouped['attacker_outcome_flag_count']) * 100
df_grouped['attacker_outcome_loss_pct'] = 100 - df_grouped['attacker_outcome_wins_pct']
df_grouped


# In[ ]:


df_grouped[['attacker_king','attacker_outcome_wins','attacker_outcome_loss','attacker_outcome_wins_pct',
            'attacker_outcome_loss_pct','attacker_size_mean','defender_size_mean']].sort_values(
            by='attacker_outcome_wins_pct',ascending=False).round(1).rename(
            columns={"attacker_king": "Attacker King", "attacker_outcome_wins": "Wins", 
                     "attacker_outcome_loss": "Loss", "attacker_outcome_wins_pct":"Win Percentage",
                     "attacker_outcome_loss_pct" : "Loss Percentage",
                     "attacker_size_mean":"Attacker Size Mean",
                     "defender_size_mean":"Defender Size Mean"})


# In[ ]:


fig5 = go.Figure(data=[go.Bar(name='win', x=df_grouped['attacker_king'], y=df_grouped['attacker_outcome_wins_pct']),
                       go.Bar(name='loss', x=df_grouped['attacker_king'], y=df_grouped['attacker_outcome_loss_pct'])])
fig5.update_layout(barmode='group', title_text='Each King Wins and Losses', title_x=0.5, template='plotly_dark')
#update_traces(marker_color='rgb(250, 218, 5)',
#                  marker_line_width=1.5, opacity=1)


# In[ ]:


fig6_1 = px.scatter(x=df_battle['attacker_king'], y=df_battle['attacker_size'], color=df_battle['attacker_outcome'],
                  template='plotly_dark')

fig6_1.update_layout(title_text='How army size affected wins and losses', title_x=0.5,
                   xaxis_title='attacker_king',  yaxis_title='attacker_size')
fig6_1.update_traces(marker_size=10)


# In[ ]:


fig6_2 = px.bar(x=df_battle['attacker_king'], y=df_battle['attacker_size'], color=df_battle['attacker_outcome'],
                  template='plotly_dark')

fig6_2.update_layout(title_text='How army size affected wins and losses', title_x=0.5,
                   xaxis_title='attacker_king',  yaxis_title='attacker_size')


# In[ ]:


fig7 = px.bar(df_battle, x='battle_type', template='plotly_dark')
fig7.update_layout(title_text='Battles Types', title_x=0.5)
fig7.update_xaxes(type='category')
fig7.update_traces(marker_color='rgb(250, 218, 5)',
                  marker_line_width=1.5, opacity=1)


# In[ ]:


fig8 = px.bar(df_battle, x='region', template='plotly_dark')
fig8.update_layout(title_text='Regions with most battles', title_x=0.5)
fig8.update_xaxes(type='category')
fig8.update_xaxes(categoryorder='total descending')
fig8.update_traces(marker_color='rgb(250, 218, 5)',
                  marker_line_width=1.5, opacity=1)


# In[ ]:


df_battle_attacker_king = df_battle.groupby(by=['attacker_king','battle_type']).agg(
    battles_count = ('name','count'),
    attacker_outcome_flag_count = ('attacker_outcome_flag','count'),
    attacker_outcome_wins = ('attacker_outcome_flag','sum'),
    attacker_size_mean = ('attacker_size', 'mean'),
    defender_size_mean = ('defender_size','mean')).reset_index().sort_values(by = 'battles_count', ascending = False)

df_battle_attacker_king['attacker_outcome_loss'] = df_battle_attacker_king['attacker_outcome_flag_count'] - df_battle_attacker_king['attacker_outcome_wins']
df_battle_attacker_king['attacker_outcome_wins_pct'] = (df_battle_attacker_king['attacker_outcome_wins']/df_battle_attacker_king['attacker_outcome_flag_count']) * 100
df_battle_attacker_king['attacker_outcome_loss_pct'] = 100 - df_battle_attacker_king['attacker_outcome_wins_pct']
df_battle_attacker_king.sort_values('attacker_king', ascending = True)


# In[ ]:


fig9 = px.bar(df_battle_attacker_king, x="attacker_king", y="attacker_outcome_wins_pct",
              color="battle_type", barmode="group", facet_col="battle_type", template='plotly_dark')
fig9.update_traces(width=0.5)
fig9.update_layout(title_text='Kings outcome in each battle type', title_x=0.5)


# ## Deaths EDA

# In[ ]:


data_path = "dataset/got_characters_s1_to_s7_v2.csv"
character_df = pd.read_csv(data_path,quotechar='"',na_values='',encoding = "ISO-8859-1")


# In[ ]:


character_df['total_screen_time'] = character_df.apply(lambda x: sum([x['s'+str(i)+'_screenTime'] for i in range(1,8)]), axis=1)
character_df['num_of_episodes_appeared'] = character_df.apply(lambda x: sum([x['s'+str(i)+'_episodes'] for i in range(1,8)]), axis=1)
character_df['num_of_people_killed'] = character_df.apply(lambda x: sum([x['s'+str(i)+'_numKilled'] for i in range(1,8)]), axis=1)


# In[ ]:


character_df


# In[ ]:


seasons = [1,2,3,4,5,6,7] 
x_data = []
y_labels =['death count']
title = 'Character Death Count (Season1-7) <br><span style="font-size:x-small;width:50%;">NOTE: only characters with considerable screen time included in death counts, but not the army people</span>'

for x_season in seasons:
    count = character_df[character_df.dead_in_season=='s'+str(x_season)]['character_name'].count()
    x_data.append([count])
#print (x_data)

for ylabel_idx,y_label in enumerate(y_labels):
    ylabel_data_points = [x_data[x_idx][ylabel_idx] for x_idx in range(len(x_data))]
#print (ylabel_data_points)



# Use the hovertext kw argument for hover text
fig10 = go.Figure(data=[go.Bar(x=seasons, y=ylabel_data_points,
            hovertext=['total number of deathes : 25', 'total number of deathes :25', 'total number of deathes :17',
                       'total number of deathes :20', 'total number of deathes :18',
                       'total number of deathes :50', 'total number of deathes :10'])])
# Customize aspect
fig10.update_traces(marker_color='rgb(250, 218, 5)',
                  marker_line_width=1.5, opacity=1)
fig10.update_layout(title=title,yaxis=dict(title='death count'),template="plotly_dark",)
fig10.show()


# In[ ]:


df2 = character_df[character_df.num_of_people_killed>1][['character_name','num_of_people_killed']].sort_values(by=['num_of_people_killed'],ascending=False).head(50)
y=list(df2['num_of_people_killed'].values)   
title='Number of People Killed by Characters in GoT (Season1-7)'

# Use the hovertext kw argument for hover text
fig11 = go.Figure(data=[go.Bar(x=list(df2['character_name'].values),
                             y=list(df2['num_of_people_killed'].values),
                             hovertext= y)])
# Customize aspect
fig11.update_traces(marker_color='rgb(250, 218, 5)',
                  marker_line_width=1.5, opacity=1)
fig11.update_layout(title=title,yaxis=dict(title='Number of Kills'),template="plotly_dark")
fig11.show()


# In[ ]:


# Use the hovertext kw argument for hover text
fig12 = px.histogram(x=character_df['gender'], y=character_df['is_dead'])
 
fig12.update_traces(marker_color='rgb(0, 204, 204)',
                  marker_line_width=1.5, opacity=1)
title='Death Presentage by Gender'
fig12.update_layout(xaxis=dict(title='Gender'),title=title,template="plotly_dark", title_x=0.5)
fig12.update_yaxes(visible = False)
fig12.show()


# In[ ]:


character_df['royal'] =character_df['royal'].map({0: 'Not Royal', 1: 'Royal'})


# In[ ]:


fig13 = px.histogram(x=character_df['royal'] ,y=character_df['is_dead'])
# Customize aspect
fig13.update_traces(marker_color='rgb(229, 204, 255)',
                  marker_line_width=1.5, opacity=1)
title='Death Presentage by Being Royal Member'
fig13.update_layout(xaxis=dict(title='Being Royal'),title=title,template="plotly_dark", title_x=0.5)
fig13.update_xaxes(type='category')
fig13.update_yaxes(visible = False)
fig13.show()


# In[ ]:


fig14 = px.histogram(x=character_df['kingsguard'] ,y=character_df['is_dead'])
# Customize aspect
fig14.update_traces(marker_color='rgb(250, 218, 5)',
                  marker_line_width=1.5, opacity=1)
title='Death Presentage by Being Kingguard '
fig14.update_layout(xaxis=dict(title='Being Kingsguard'),title=title,template="plotly_dark")
fig14.update_xaxes(type='category')
fig14.update_yaxes(visible = False)
fig14.show()


# In[ ]:


fig15 = px.histogram(x=character_df['house'] ,y=character_df['is_dead'])
# Customize aspect
fig15.update_traces(marker_color='rgb(250, 218, 5)',
                  marker_line_width=1.5, opacity=1)
title='Death Presentage by House'
fig15.update_layout(title=title,template="plotly_dark")
fig15.update_yaxes(visible = False)
fig15.update_xaxes(title = 'House')
#fig.update_yaxes(tick0=0, dtick=50)
fig15.show()


# In[ ]:


## chart of overall screen time by characters

df2 = character_df[['character_name','total_screen_time','s1_screenTime','s2_screenTime','s3_screenTime','s4_screenTime','s5_screenTime','s6_screenTime','s7_screenTime']].sort_values(by=['total_screen_time'], ascending=False).head(30)

traces = []
for i in range(1,8):
    s_prefix = 's' + str(i) + '_'
    traces.append(
        go.Bar(
            x = list(df2['character_name'].values),
            y = list(df2[s_prefix + 'screenTime'].values),
            name = 'Season ' + str(i)
        )
    )

data = [traces[i] for i in range(len(traces))]
title = 'Character Overall Screen Times (Season1-7) ' + '<br><span style="font-size:x-small;width:50%;">NOTE: click on individual legend items to the right to selectively enable or disable a color group</span>'
layout = go.Layout(barmode='stack', yaxis=dict(title='screen time (mins)'))

fig16 = go.Figure(data=data, layout=layout)
fig16.update_layout(title=title,template="plotly_dark")
fig16.show()

# ## Dash App

# In[ ]:


app = Dash(name=__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

tabs_styles = {
    'height': '44px',
    'align-items': 'center'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'border-radius': '15px',
    'background-color': '#F2F2F2',
    'box-shadow': '2px 2px 2px 2px rgba(149, 165, 166)',
    'margin-top': '35px'

}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
    'border-radius': '15px',
    'margin-top': '35px'
}

colors = {
    'background': '#111111',
    'head': '#ffffff',
    'text': '#E3BE00'
}

app.layout = html.Div(style={'backgroundColor': colors['background'], 'margin-top': '0', 'height': '100%', 'width':'100%'},
                      children=[
                                html.H1('GOT Series Analysis' ,style={'textAlign': 'center', 'margin-top': '25px','width':'100%',
                                                                      'height': '100%',
                                                               'color': '#F2F2F2','font-family':'Game of Thrones',
                                                               'font-size': 40, 'display': 'inline-block'}),  
    
html.Div([
    html.Div([
        dcc.Tabs(id = "tabs-styled-with-inline", value = 'tab-1', children = [
            dcc.Tab(label = 'Series Ratings', value = 'tab-1', style = tab_style, selected_style = tab_selected_style),
            dcc.Tab(label = 'Battles', value = 'tab-2', style = tab_style, selected_style = tab_selected_style),
            dcc.Tab(label = 'Deaths', value = 'tab-3', style = tab_style, selected_style = tab_selected_style)
        ], style = tabs_styles)]),
           html.Br(),
        html.Div(id = 'tabs-content-inline')])])

        


@app.callback(Output('tabs-content-inline', 'children'),
              Input('tabs-styled-with-inline', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div(children=[
                        
                        html.Div(children=[html.P('Select a Season', 
                        style = {'color': '#F2F2F2', 'font-size': 20, 'margin-top': '70px', 'textAlign': 'center'}),
                                
                        dcc.Dropdown(id='select_season', clearable=False, value=None, multi=True,
                                     placeholder='Choose a Season .....', style = {'width': '60%',
                                    'margin-top': '1em', 'margin-bottom': '1em', 'margin-left': '240px'},
                                     options=[{'label': season, 'value': season} for season in df_rating.Season.unique()]),
                                    
                        dcc.Graph(id='episodes_rating')]),
                
                html.Hr(style={'color': '#F2F2F2'}),
    
                html.Div(children=[dcc.Graph(id='views-graph', figure=fig2)]),
           
                html.Hr(style={'color': '#F2F2F2'}),
            
                html.Div(children=[dcc.Graph(id='charchter_screen_time', figure=fig16)]),
                
                html.Hr(style={'color': '#F2F2F2'}),
            
                html.Div(children=[dcc.Graph(id='directors-graph', figure=fig3)])])

                        
    elif tab == 'tab-2':
        return  html.Div(children=[ 
                                
                                html.Div(style = {'margin-top':'20px'},
                                         children=[dcc.Graph(id='Battle types', figure=fig9)]),
                                
                                html.Hr(style={'color': '#F2F2F2'}),
                                
                                html.Div(style = {'display': 'inline-block', 'width': '50%', 'margin-top':'30px'},
                                         children=[dcc.Graph(id='Each King Wins and Losses', figure=fig5)]),
                                
                                html.Div(style = {'display': 'inline-block', 'width': '50%', 'margin-top': '30px'},
                                         children=[dcc.Graph(id='How army size affected wins and losses(scatter)', figure=fig6_1)]),
                             
                                #html.Div(style = {'display': 'inline-block', 'width': '50%', 'margin-top': '30px'},
                                 #        children=[dcc.Graph(id='How army size affected wins and losses(bar)', figure=fig6_2)]),
                               
                                #html.Div(children=[dcc.Graph(id='Battles Types', figure=fig7)]),
                                html.Hr(style={'color': '#F2F2F2'}),
                                html.Div(style = {'display': 'inline-block', 'width': '50%'},
                                         children=[dcc.Graph(id='Kings Battles Counts', figure=fig4)]),
            
                                html.Div(style = {'display': 'inline-block', 'width': '50%'},
                                children=[dcc.Graph(id='Regions with most battles', figure=fig8)])])
    

    elif tab == 'tab-3':
        return  html.Div(children=[ 
                                
                                html.Div(children=[dcc.Graph(id='x', figure=fig10)]),
            
                                html.Hr(style={'color': '#F2F2F2'}),
                                
                                html.Div(children=[dcc.Graph(id='y', figure=fig11)]),
            
                                html.Hr(style={'color': '#F2F2F2'}),
            
                                html.Div(children=[dcc.Graph(id='a', figure=fig15)]),
            
                                html.Hr(style={'color': '#F2F2F2'}),
                                
                                html.Div(style = {'display': 'inline-block', 'width': '50%'},
                                         children=[dcc.Graph(id='z', figure=fig12)]),
                               
                                html.Div(style = {'display': 'inline-block', 'width': '50%'},
                                         children=[dcc.Graph(id='w', figure=fig13)])
                               
                                #html.Div(children=[dcc.Graph(id='f', figure=fig14)]),
                               
                                ])
     


@app.callback(
    Output('episodes_rating', 'figure'),
    [Input("select_season", "value")])

def update_episodes_rating(dropdownvalue):
    if dropdownvalue==None:
        df_rating["Season"]= df_rating["Season"].astype(str)
        fig = px.bar(df_rating, x='Episode', y='Rating', color='Season',
              color_discrete_sequence=px.colors.qualitative.Dark24
              ,title='Episodes Ratings', template='plotly_dark')
        fig.update_layout(title_text='Episodes Ratings')
        
        return fig

    
    else:
        df_rating["Season"]= df_rating["Season"].astype(int)
        filtered_df = df_rating[df_rating['Season'].isin(dropdownvalue)]
        filtered_df["Season"]= filtered_df["Season"].astype(str)
                          
        fig = px.bar(filtered_df, x='Episode', y='Rating', color='Season',
              color_discrete_sequence=px.colors.qualitative.Dark24, title='Series Episodes Ratings', template='plotly_dark')
        fig.update_layout(title_text='Episodes Ratings')
        
        return fig


# In[ ]:


if __name__ == '__main__':
    app.run_server()


# In[ ]:




