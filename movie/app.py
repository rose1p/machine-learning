from flask import Flask, render_template, request
import pickle
from tmdbv3api import Movie, TMDb

app = Flask(__name__)

movie = Movie()
tmdb = TMDb()
tmdb.api_key = 'c668cda4cf75bf267ef2aeffa2da0341'
tmdb.language = 'ko-KR'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/page1')
def page1():
    return render_template('page1.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/page3')
def page3():
    return render_template('page3.html')

# 인기영화 순으로 나열
@app.route('/movies1.json')
def movie1():
    movies = pickle.load(open('./data/movies1.pickle', 'rb'))
    json = []
    for i in range(10):
        title = movies['title'].iloc[i]
        id = movies['id'].iloc[i]
        details = movie.details(id)
        image = details['poster_path']
        title = details['title']
        # etitle = details['etitle']
        if image:
            image = 'https://image.tmdb.org/t/p/w500' + image
        else:
            image = 'http://via.placeholder.com/100x120'
            
        score = round(movies['score'].iloc[i], 2)
        data = {'title' : title, 'id': str(id), 'score' : str(score), 'image' : image}
        json.append(data)
    return json

# 전체영화 제목
@app.route('/movies.json')
def movies():
     movies = pickle.load(open('./data/movies.pickle', 'rb'))
     json = movies.to_json(orient = 'records')
     return json
 
 # 줄거리 추천 영화
@app.route('/movie2.json/<title>')
def movises2(title):
    sim = pickle.load(open('./data/sim.pickle', 'rb'))
    movies = pickle.load(open('./data/movies.pickle', 'rb'))
    
    # 1. 제목에 대한 인덱스값 구한다.
    filter = movies['title'] == title
    idx = movies[filter].index[0]
    
    # 2. 유사도 테이블에서 인덱스번째 유사도 데이터를 구한다
    sim_scores = sim[idx]
    
    # 3. 인덱스번호, 유사도 값을 배열로 생성
    sim_scores = list(enumerate(sim_scores))
    
    # 4. 비교했을때 유사도가 가장 높은순으로 정렬
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
 
    # 5. 상위 10개의 값을 구한다.
    sim_scores = sim_scores[1:13]
 
    # 6. 인덱스 배열 생성
    sim_movies = [i[0] for i in sim_scores]

    json = []
    for i in range(12):
        etitle = movies['title'].iloc[sim_movies[i]]
        id = movies['id'].iloc[sim_movies[i]]
        details = movie.details(id)
        image = details['poster_path']
        title = details['title']
        # etitle = details['etitle']
        if image:
            image = 'https://image.tmdb.org/t/p/w500' + image
        else:
            image = 'http://via.placeholder.com/100x120'
            
        data = {'etitle' : etitle, 'title': title, 'image': image}
        json.append(data)
        
    return json
    
# 배우/감독 추천 영화
@app.route('/movie3.json/<title>')
def movises3(title):
    sim = pickle.load(open('./data/sim2.pickle', 'rb'))
    movies = pickle.load(open('./data/movies.pickle', 'rb'))
    
    # 1. 제목에 대한 인덱스값 구한다.
    filter = movies['title'] == title
    idx = movies[filter].index[0]
    
    # 2. 유사도 테이블에서 인덱스번째 유사도 데이터를 구한다
    sim_scores = sim[idx]
    
    # 3. 인덱스번호, 유사도 값을 배열로 생성
    sim_scores = list(enumerate(sim_scores))
    
    # 4. 비교했을때 유사도가 가장 높은순으로 정렬
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
 
    # 5. 상위 10개의 값을 구한다.
    sim_scores = sim_scores[1:13]
 
    # 6. 인덱스 배열 생성
    sim_movies = [i[0] for i in sim_scores]

    json = []
    for i in range(12):
        etitle = movies['title'].iloc[sim_movies[i]]
        id = movies['id'].iloc[sim_movies[i]]
        details = movie.details(id)
        image = details['poster_path']
        title = details['title']
        # etitle = details['etitle']
        if image:
            image = 'https://image.tmdb.org/t/p/w500' + image
        else:
            image = 'http://via.placeholder.com/100x120'
            
        data = {'etitle' : etitle, 'title': title, 'image': image}
        json.append(data)
    return json
    
if __name__== '__main__':
    app.run(port=5000, debug=True)