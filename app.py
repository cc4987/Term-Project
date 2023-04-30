from flask import Flask, render_template, request
from main import get_recs

app = Flask(__name__)

@app.route('/')
def hello():
    return 'READ.md'

@app.get('/movie')
def get_movie():
    return render_template('index.html')
@app.post('/movie')
def movie_post():
    movie_search = request.form.get('movie_name')



if __name__ == '__main__':
    app.run(debug=True)