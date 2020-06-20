import numpy as np
from flask import Flask, request, jsonify, render_template,url_for
from joblib import load

app = Flask(__name__,template_folder="template")
model = load('HeartDisease.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output=prediction*10

    return render_template('home.html', prediction_text='Probability of Heart Disease is {} %'.format(int(output)))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)