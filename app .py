from flask import Flask,jsonify,request,render_template
import pickle
import numpy as np

app = Flask(__name__)
lr=pickle.load(open('model_2.pkl','rb'))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    float_features=[float(x) for x in request.form.values()]
    final=[np.array(float_features)]
    prediction=lr.predict(final)
    return render_template('index.html',prediction_text ='predicted class{}'.format(float(prediction)))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = lr.predict([int(np.array(list(data.values())))])

    # output = prediction[0]
    return jsonify(prediction)
if __name__ == "__main__":
    app.run(debug=True)