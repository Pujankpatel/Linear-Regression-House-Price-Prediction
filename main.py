import pickle
from flask import Flask,render_template,request,redirect,url_for,app,jsonify
import numpy as np
import pandas as pd

app=Flask(__name__)
#load the model
houseregmod=pickle.load(open('Model/modelForPrediction.pkl','rb'))
scalar=pickle.load(open('Model/scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']#request apsu predict_api na a data json form ma jase
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=houseregmod.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=houseregmod.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))
if __name__=="__main__":
    app.run(debug=True)


