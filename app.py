from flask import Flask, render_template, request
from main import get_recs

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Please enter /movie in the url to reach the website!'

@app.get('/movie')
def get_movie():
    return render_template('recs.html')
@app.post('/movie')
def movie_post():
    movie_search = request.form.get('title')
    result = get_recs(movie_search)

    return render_template('result.html', top_titles_df)



if __name__ == '__main__':
    app.run(debug=True)