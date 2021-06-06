# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import src

# Load the Random Forest CLassifier model

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        str1 = str(request.form['pregnancies'])
        
        my_prediction = src.give_me_output(str1)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)