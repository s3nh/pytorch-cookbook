import flask
from flask import Flask, request, render_template

from skimage import io 
import numpy as np 
import json 
import zipfile 




app =  Flask(__name__)

@app.route("/")
@app.route("/index")

def index():
    return flask.render_template('index.html')
    


