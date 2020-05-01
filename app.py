import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px

#sk
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('./books.csv')

def read():
    return pd.read_csv('./books.csv')

def reset(df,flag=True):
    if 'similarity' in df.columns:
        df.drop(['similarity'], axis=1, inplace=True)

    df = df.dropna()
    return df

def generate_table(dataframe,value, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

def plot_bar(df):
    fig = px.bar(df, y='similarity', x='book_name', text='similarity')
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    return fig


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Plotter',style={  'text-align':'center','padding':'30px'}),
    html.Div(children=[
      dcc.Input(id='input',placeholder='enter plot to search',type='text',style={'text-align':'center','padding':'10px'}),
      html.Div(html.Button('Click',id='button',n_clicks=0))],
    style={
        'text-align':'center',
        'padding-top' : '10px',
        'margin' :'auto',
        'block' : 'in-line'

    }),


    html.Div(id='body',children=[
    dcc.Graph(id='bar'),
    dcc.Graph(id='pie')
        ],style={'visibility':'hidden'}),
    html.Div(id='Output'),
],style={
    # 'position': 'absolute',
    # 'top': '50 %',
    # 'left':' 50 %',
    # 'margin - top':' -50px',
    # 'margin - left':' -50px',
    # 'width':' 100px',
    # 'height': '100px'
})


def plot_pie(df):
    fig = px.pie(df, values='similarity', names='genre', title='Users Preffered Genre')
    return fig


@app.callback(
    [dash.dependencies.Output('body', 'style'),
     dash.dependencies.Output('Output', 'children'),
     dash.dependencies.Output('bar', 'figure'),
     dash.dependencies.Output('pie', 'figure')],
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input', 'value')])
def update_output(n_clicks=0, value=' '):
    if n_clicks > 0:
        n_clicks =0
        df = read()
        df = reset(df)
        df['corpus_tf'] = (pd.Series(df[['description']]
                                     .fillna('')
                                     .values.tolist()
                                     ).str.join(' '))

        tf_corpus = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
        tfidf_matrix_corpus = tf_corpus.fit_transform(df['corpus_tf'].values.astype('U'))
        i = value
        i = [i]
        i = tf_corpus.transform(i)
        df['similarity'] = cosine_similarity(i, tfidf_matrix_corpus).T
        df = df.groupby(['book_name', 'description','genre'], as_index=False).agg({'similarity': pd.Series.mean}).sort_values(
            by='similarity', ascending=False).head(10)

        return {'visibility':'visible'},generate_table(df,value),plot_bar(df),plot_pie(df)



if __name__ == '__main__':
    app.run_server(debug=True)