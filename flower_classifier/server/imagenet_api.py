from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin

from src.model import FlowerNet
from PIL import Image 
from src.prediction import ModelLoader

app = Flask(__name__)
cors = CORS(app)

@app.route("/", methods=['GET'])
def index():
	return send_from_directory('.', 'index.html')

@app.route("/api/v1/classify", methods=['POST'])
@cross_origin(origin='*')
def classify():
    ml = ModelLoader()
    image = request.files['image'].read()
    image = ml.image_prepare(image)
    prediction = ml._predict(image)


    return jsonify(
        prediction=prediction
    )

if __name__ == '__main__':
    app.run(debug=True)
