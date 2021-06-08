import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc

import dash_html_components as html
import plotly.graph_objects as go
#import swifter
import plotly.graph_objs as go
import datetime as dt
import plotly.express as px
import pandas as pd
import re


from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

import spacy
s = SIA()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, prevent_initial_callbacks=True, external_stylesheets=external_stylesheets)


def clean(text):
    # removing paragraph numbers
    text = re.sub('[0-9]+.\t','',str(text))
    # removing new line characters
    text = re.sub('\n ','',str(text))
    text = re.sub('\n',' ',str(text))
    # removing apostrophes
    text = re.sub("'s",'',str(text))
    # removing hyphens
    text = re.sub("-",' ',str(text))
    text = re.sub("— ",'',str(text))
    # removing quotation marks
    text = re.sub('\"','',str(text))
    text = re.sub('  \t','',str(text))
    text = re.sub(' \t','', str(text))
    text = re.sub('\t','',str(text))
    # removing salutations
    text = re.sub("Mr\.",'Mr',str(text))
    text = re.sub("Mrs\.",'Mrs',str(text))
    # removing any reference to outside text
    text = re.sub("[\(\[].*?[\)\]]", "", str(text))
    return text


def sentiment_calculation(df, body, compound_score,pos_score,neg_score,neu_score):
    df[body] = df[body].apply(clean)
    df[compound_score] = df[body].apply(lambda x : s.polarity_scores(str(x))['compound'])
    df[pos_score] = df[body].apply(lambda x : s.polarity_scores(str(x))['pos'])
    df[neg_score] = df[body].apply(lambda x : s.polarity_scores(str(x))['neg'])
    df[neu_score] = df[body].apply(lambda x : s.polarity_scores(str(x))['neu'])
    
    return df
    
    
colors = {
    'text': '#33C3F0'
}
app.layout = html.Div(children=[
    html.H1(
        children="Wizikey",
        style={
            "textAlign": "center",
            'backgroundColor': colors['text']
        }
    ),html.Div(dcc.Input(id='input-on-submit', type='text',placeholder ='Enter-Customer-Name',style={
            'textAlign': 'center'
        } )),
    html.Button('Submit', id='submit-val', n_clicks=0, style={
            'textAlign': 'center',
            'color': '#D90416'
        }),
    html.Div(id='container-button-basic',
             children='Enter a value and press submit'),


    dcc.DatePickerRange(
        id="date-picker-range",
        start_date=dt.datetime(2020, 6, 1),
        end_date=dt.datetime(2020, 6, 30),
        end_date_placeholder_text="Select a date"
    )
    

])

@app.callback(
    
    Output('container-button-basic', 'children'),
    [Input('submit-val', 'n_clicks')],
    [State("date-picker-range", "start_date"),
    State("date-picker-range", "end_date"),
    State('input-on-submit', 'value')]
    
  
    
)

def update_graph(n_clicks,start_date, end_date, value):
    data = pd.read_excel('\\Users\\HP\\data_wysh.xlsx')    
    data['Date_Local_Time'] = pd.to_datetime(data['publishedAt'], format='%Y/%m/%d', utc = True).dt.date # String to datetime
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    data_before = data[data['Date_Local_Time'] < end_date]
    data_after = data[data['Date_Local_Time'] >= end_date]
    #data[(data['Date_Local_Time'] > start_date) & (data['Date_Local_Time'] < end_date)]
    String_1 = str(value)

    data_before = data_before[data_before['body'].str.contains(r"\b{}\b".format(String_1), case = True,na = False)]
    data_after = data_after[data_after['body'].str.contains(r"\b{}\b".format(String_1), case = True,na = False)]
    
    
    #before processing
    data_before = sentiment_calculation(data_before,'body','compound_score','pos_before','neg_before','neu_before')
    data_before['type'] = data_before['compound_score'].apply(lambda x:'Positive' if x>= 0.5 else ('Neutral' if (x > -0.5) and (x < 0.5) else 'Negative'))    
    if len(data_before['body'])!=0:       
        pos_sentiment_score_before=sum(data_before.pos_before)/len(data_before['body'])
        neg_sentiment_score_before=sum(data_before.neg_before)/len(data_before['body'])
        neu_sentiment_score_before=sum(data_before.neu_before)/len(data_before['body'])
    else:
        pos_sentiment_score_before=sum(data_before.pos_before)
        neg_sentiment_score_before=sum(data_before.neg_before)
        neu_sentiment_score_before=sum(data_before.neu_before)
    
    x = list(data_before.type.value_counts().index)
    y = list(data_before.type.value_counts())
    d={}
    for a,b in zip(x,y):
        d[a] = b
    d_1= {}
    for x in ['Positive','Negative','Neutral']:
        if x in d:
            d_1[x] = d[x]
        else:
            d_1[x] =0
    
    #after processing
    data_after = sentiment_calculation(data_after,'body','compound_score','pos_after','neg_after','neu_after')
    data_after['type'] = data_after['compound_score'].apply(lambda x:'Positive' if x>= 0.5 else ('Neutral' if (x > -0.5) and (x < 0.5) else 'Negative'))    
    if len(data_after['body'])!=0:       
        pos_sentiment_score_after=sum(data_after.pos_after)/len(data_after['body'])
        neg_sentiment_score_after=sum(data_after.neg_after)/len(data_after['body'])
        neu_sentiment_score_after=sum(data_after.neu_after)/len(data_after['body'])
    else:
        pos_sentiment_score_after=sum(data_after.pos_after)
        neg_sentiment_score_after=sum(data_after.neg_after)
        neu_sentiment_score_after=sum(data_after.neu_after)
        
        
    X = list(data_after.type.value_counts().index)
    Y = list(data_after.type.value_counts())
    D={}
    for a,b in zip(X,Y):
        D[a] = b
    D_1= {}
    for x in ['Positive','Negative','Neutral']:
        if x in D:
            D_1[x] = D[x]
        else:
            D_1[x] =0
    


    return dcc.Tabs([
        dcc.Tab(label='Before Joining Wizikey '+'till '+str(end_date), children=[
            dcc.Graph(
        id = "pieGraph",
        figure = {
          "data": [
            {
              "values": list([d_1['Positive'],d_1['Negative'],d_1['Neutral']]),
              "labels": [
                "Positive",
                "Negative",
                "Neutral"
              ],
              "textinfo":"label+percent",
              "insidetextorientation":"radial",
              "name": "Sentiment",
              "hoverinfo":"label+name+percent",
              "hole": 0,
              "type": "pie",
              "marker" : dict(colors=['#05C7F2','#D90416','#D9CB04'])}],
          "layout": {
                "title" : dict(text ="Type of News",
                               font =dict(
                               size=20,
                               color = 'white')),
                "paper_bgcolor":"#111111",
                "width": "2000",
                "annotations": [
                    {
                        "font": {
                            "size": 20
                        },
                        "showarrow": False,
                        "text": "",
                        "x": 0.2,
                        "y": 0.2
                    }
                ],
                "showlegend": True
              }
        }
    
       
        ),dcc.Graph(
        id = "pieGrapha",
        figure = {
          "data": [
            {
              "values": [pos_sentiment_score_before, neg_sentiment_score_before, neu_sentiment_score_before],
              "labels": [
                "Positive",
                "Negative",
                "Neutral"
              ],
              "name": "Sentiment",
              "hoverinfo":"label+name+percent",
              "hole": .7,
              "type": "pie",
              "marker" : dict(colors=['#05C7F2','#D90416','#D9CB04'])}],
          "layout": {
                "title" : dict(text ="Sentiment Analysis",
                               font =dict(
                               size=20,
                               color = 'white')),
                "paper_bgcolor":"#111111",
                "width": "20",
                "annotations": [
                    {'text':str(len(data_before))+'articles',
                        "font": {
                            "size": 1,
                            'color':'white'
                        },
                        "showarrow": False,
                        "text": "",
                        "x": 10,
                        "y": 10
                    }
                ],
                "showlegend": True
              }
        }
    
       
        ), 
       
        
    ]),
    dcc.Tab(label='After Joining Wizikey '+'after '+str(end_date), children=[
            dcc.Graph(
        id = "pieGraph1",
        figure = {
          "data": [
            {
              "values": list([D_1['Positive'],D_1['Negative'],D_1['Neutral']]),
              "labels": [
                "Positive",
                "Negative",
                "Neutral"
              ],
              "textinfo":"label+percent",
              "insidetextorientation":"radial",
              "name": "Sentiment",
              "hoverinfo":"label+name+percent",
              "hole": 0,
              "type": "pie",
              "marker" : dict(colors=['#05C7F2','#D90416','#D9CB04'])}],
          "layout": {
                "title" : dict(text ="Type of News",
                               font =dict(
                               size=20,
                               color = 'white')),
                "paper_bgcolor":"#111111",
                "width": "2000",
                "annotations": [
                    {
                        "font": {
                            "size": 20
                        },
                        "showarrow": False,
                        "text": "",
                        "x": 0.2,
                        "y": 0.2
                    }
                ],
                "showlegend": True
              }
        }
    
       
        ),
        dcc.Graph(
        id = "pieGraph1a",
        figure = {
          "data": [
            {
              "values": [pos_sentiment_score_after, neg_sentiment_score_after, neu_sentiment_score_after],
              "labels": [
                "Positive",
                "Negative",
                "Neutral"
              ],
              "name": "Sentiment",
              "hoverinfo":"label+name+percent",
              "hole": .7,
              "type": "pie",
              "marker" : dict(colors=['#05C7F2','#D90416','#D9CB04'])}],
          "layout": {
                "title" : dict(text ="Sentiment Analysis",
                               font =dict(
                               size=20,
                               color = 'white')),
                "paper_bgcolor":"#111111",
                "width": "20",
                "annotations": [
                    {'text' : str(len(data_after))+'articles',
                        "font": {
                            "size": 1
                        },
                        "showarrow": False,
                        "text": "",
                        "x": 10,
                        "y": 10
                    }
                ],
                "showlegend": True
              }
        }
    
       
        )
    ])

    ])
if __name__ == '__main__':
    app.run_server(debug=True)