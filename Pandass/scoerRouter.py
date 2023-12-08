from io import BytesIO
from flask import render_template, Blueprint, send_file, request
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic' #맑은 고딕
matplotlib.rcParams['font.size'] = 15 #글자 크기
matplotlib.rcParams['axes.unicode_minus'] = False #한글 폰트 사용 시 마이너스 글자가 깨지는 현상을 해결
score=Blueprint('score', __name__)

@score.route('/page1')
def score_page1():
    df=pd.read_csv('score.csv' ,index_col='지원번호')
    df2=df[['이름', '학교', '키', 'SW특기']]
    df3=df[['이름', '국어', '영어', '수학']]

    return render_template('page1.html', 
                           table2=df2.to_html(classes='table table-striped', table_id='tbl2'),
                           table3=df3.to_html(classes='table table-striped', table_id='tbl3')
                           )
@score.route('/graph')
def score_graph():
    return render_template('graph.html')

@score.route('/chart/<subject>')
def score_chart(subject):
    plt.figure(figsize=(10, 5), dpi=50)
    df=pd.read_csv('score.csv' ,index_col='지원번호')
    x=df['이름'].values
    y=df[subject].values
    plt.bar(x, y)
    #plt.show()
    img = BytesIO()
    plt.savefig(img, format='png', dpi=50)
    img.seek(0)
    return send_file(img, mimetype='image/png')

@score.route('/list.json')
def score_json():
    query = request.args['query']
    df=pd.read_csv('score.csv' ,index_col='지원번호')
    fliter = df['이름'].str.contains(query)
    df=df[fliter]
    json = df.to_json(orient = 'records')
    return json

@score.route('/page3')
def score_page3():
    return render_template('page3.html')