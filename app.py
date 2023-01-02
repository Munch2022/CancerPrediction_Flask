import os, sys
from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
import numpy as np

# loading the trained model; copy the path where u saved downloaded trained model
loaded_model= pickle.load(open('C:/Users\Manjula\Desktop\Project_CancerPrediction_Flask/svc_model.pkl', 'rb')) 

# creating function to predict new data
    
def cancer_prediction(new_data):
    input_data_arr= np.asarray(new_data)
    inputdata_reshaped= input_data_arr.reshape(1, -1)               # coz it sontains sample(series of values)

    prediction = loaded_model.predict(inputdata_reshaped)              # here instead of clf2, im giving the loaded_model as teh model is loaded in this variable
    print(prediction)

    if (prediction[0] == 2):
        return 'Its a benign cancer. The patient is not diagnosed with Breast Cancer'
    else:
        return 'Its a malignant cancer, Consult doctor immediately. The patient is diagnosed with Breast Cancer'
    


# creating flask app
app = Flask(__name__)

@app.route('/')
def loadPage():
	return render_template('home.html', query="")



@app.route("/predict", methods=['POST'])

def predict():
    if request.method == 'POST':
        inputdata1 = request.form['query1']
        inputdata2 = request.form['query2']
        inputdata3 = request.form['query3']
        inputdata4 = request.form['query4']
        inputdata5 = request.form['query5']
        inputdata6 = request.form['query6']
        inputdata7 = request.form['query7']
        inputdata8 = request.form['query8']
        inputdata9 = request.form['query9'] 

        input_data= [[inputdata1, inputdata2, inputdata3, inputdata4, inputdata5, inputdata6, inputdata7, inputdata8, inputdata9]]
        pred= cancer_prediction(input_data)

    return render_template('home.html', output1= pred, query1 = request.form['query1'], query2 = request.form['query2'],query3 = request.form['query3'],query4 = request.form['query4'],query5 = request.form['query5'], query6 = request.form['query6'], query7 = request.form['query7'], query8 = request.form['query8'], query9 = request.form['query9'])  




if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=9100)

    
# if __name__ == '__main__':
#    app.run(debug=True)


    

    
