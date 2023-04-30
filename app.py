from flask import Flask, render_template, request
from main import get_recs

app = Flask(__name__)

@app.route('/')