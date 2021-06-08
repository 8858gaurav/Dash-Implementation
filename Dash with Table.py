import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc

import dash_html_components as html
import plotly.graph_objs as go
import datetime as dt
import swifter
import pandas as pd
#import swifter
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import spacy
s = SIA()
import re
import numpy as np


import pickle
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
tqdm.pandas(desc="progress bar!")

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
    
def calculation(df,headline,body,headline_body):
    df[headline_body] = df[headline] + " " + df[body]
    return df

def remove_short_words(text):
    words = [w for w in text if len(w) > 2]
    return words

new_words = ["div", "style", "nbsp", "font", "http", "bodytext", "class", "href", "rdquo", "ldquo", "an", "rsquo", "news", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]
def remove_stopwords(text):
    words = [w for w in text if w not in new_words]
    return words

def word_lemmatizer(text):
    lem_text = " ".join([lemmatizer.lemmatize(w) for w in text])
    return lem_text
    
def get_n_largest_ind(a,n):
    ind = np.argpartition(a, -n)[-n:]
    return ind[0]



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, prevent_initial_callbacks=True,external_stylesheets=external_stylesheets)


colors = {
    'text': '#33C3F0'
}
app.layout = html.Div(children=[
    html.H1(
        children="Google",
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
        start_date=dt.datetime(2020, 5, 1),
        end_date=dt.datetime(2020, 5, 31),
        min_date_allowed=dt.datetime(2020, 5, 1),
        max_date_allowed=dt.datetime(2020, 5, 31),
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
    
    data = pd.read_excel('/Users/HP/Downloads/Industry_May.xlsx', index_col = False).reset_index(drop = True)
    data  = data[['_source.publishedAt','_source.domain','_source.url','_source.body','_source.headline']]
    data.rename(columns={'_source.publishedAt': 'publishedAt','_source.domain':'domain','_source.url':'url','_source.body':'body','_source.headline':'headline'}, inplace=True)
    data['Date_Local_Time'] = pd.to_datetime(data['publishedAt'], format='%Y/%m/%d', utc = True).dt.date # String to datetime
    data = calculation(data, 'headline','body','headline_body')
    data['headline_body'] = data['headline_body'].apply(lambda x: clean(x))
    data['headline_body'] = data['headline_body'].str.split()
    data['word_length'] = data['headline_body'].str.len()
    data['headline_body'] = data['headline_body'].apply(lambda x: remove_short_words(x))
    data['headline_body'] = data['headline_body'].apply(lambda x: remove_stopwords(x))
    data['headline_body']  = data['headline_body'].apply(lambda x: word_lemmatizer(x))
    
    
    

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    String_1 = str(value)


    data_2 = data[(data['Date_Local_Time'] >= start_date) & (data['Date_Local_Time'] <= end_date)]
    data_1 = data_2[data_2['body'].str.contains(r"\b{}\b".format(String_1), case = True,na = False)] # '(?:\s|^|[,;])'+String_1+'(?:\s|$|[,;])'
    
    data_1 = sentiment_calculation(data_1,'body','compound_score','pos_score','neg_score','neu_score')
    data_1['type'] = data_1['compound_score'].apply(lambda x:'Positive' if x>= 0.5 else ('Neutral' if (x > -0.5) and (x < 0.5) else 'Negative')) 
    
   
    if len(data_1['body'])!=0:       
        pos_sentiment_score=sum(data_1.pos_score)/len(data_1['body'])
        neg_sentiment_score=sum(data_1.neg_score)/len(data_1['body'])
        neu_sentiment_score=sum(data_1.neu_score)/len(data_1['body'])
    else:
        pos_sentiment_score=sum(data_1.pos_score)
        neg_sentiment_score=sum(data_1.neg_score)
        neu_sentiment_score=sum(data_1.neu_score)
    
    X = list(data_1.type.value_counts().index)
    Y = list(data_1.type.value_counts())
    D={}
    for a,b in zip(X,Y):
        D[a] = b
    D_1= {}
    for x in ['Positive','Negative','Neutral']:
        if x in D:
            D_1[x] = D[x]
        else:
            D_1[x] =0
            
            
    hist = dcc.Graph(
        id = "pieGraph",
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
    
       
        )
    
    figure = dcc.Graph(
        id = "pieGraph",
        figure = {
          "data": [
            {
              "values": [pos_sentiment_score, neg_sentiment_score, neu_sentiment_score],
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
    
       
        )
        
    bar = dcc.Graph(
        id = "3",
        figure ={
                  "data": [
                  {
                          'x':['positive', 'negative', 'neutral'],
                          'y':[pos_sentiment_score,neg_sentiment_score,neu_sentiment_score],
                          'name':'SF Zoo',
                          'type':'bar',
                          'marker' :dict(color=['#05C7F2','#D90416','#D9CB04']),
                  }],
                "layout": {
                      "title" : dict(text ="Overall Sentiments",
                                     font =dict(
                                     size=20,
                                     color = 'white')),
                      "xaxis" : dict(tickfont=dict(
                          color='white')),
                      "yaxis" : dict(tickfont=dict(
                          color='white')),
                      "paper_bgcolor":"#111111",
                      "plot_bgcolor":"#111111",
                      "width": "2000",
                      "grid": {"rows": 1, "columns": 2},
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
        )
    
    return  'Total number of an articles related with '+ String_1+ ' is {} '.format(len(data_1)), ',Positive News Count- '+str(D_1['Positive']),',Negative News Count - '+str(D_1['Negative']),',Neutral News Count- '+str(D_1['Neutral']),hist, bar, figure, html.Table(
        # Header
        [html.Tr([html.Th(col) for col in data_1[['Date_Local_Time','pos_score','neg_score','type']]]) ] +
        # Body
        [html.Tr([
            html.Td(data_1.iloc[i][col]) for col in data_1[['Date_Local_Time','pos_score','neg_score','type']]
        ]) for i in range(min(len(data_1), 15))]
    )

if __name__ == "__main__":
    app.run_server(debug=True)