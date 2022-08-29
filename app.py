from flask import Flask,request,url_for,redirect,render_template

import joblib
import pandas as pd

app = Flask(__name__)


model = joblib.load("model.pkl")
scale = joblib.load("scale.pkl")


@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():

    pregnancies=request.form['1']
    glucose=request.form['2']
    bloodPressure=request.form['3']
    skinThickness=request.form['4']
    insulin=request.form['5']
    bmi=request.form['6']
    dpf=request.form['7']
    age=request.form['8']

    rowDF= pd.DataFrame([pd.Series([pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,dpf,age])])
    rowDF_new= pd.DataFrame(scale.transform(rowDF))

    print(rowDF_new)

    # model prediction 
    prediction= model.predict_proba(rowDF_new) 
    print(prediction)
    print(f"The Predicted values is :{prediction[0][1]}") 
    if prediction[0][1] >= 0.5: 
        valPred = round(prediction[0][1],3) 
        print(f"The Round val {valPred*100}%") 
        return render_template('result.html',pred=f'You have a chance of having diabetes.\n\nProbability of you being a diabetic is {valPred*100:.2f}%.\n\nAdvice : Exercise Regularly') 
    else: 
        valPred = round(prediction[0][0],3) 
        print(f"The Round val {valPred*100}%") 
        return render_template('result.html',pred=f'Congratulations!\n\n You are in a Safe Zone.\n\n Probability of you being a non-diabetic is {valPred*100:.2f}%.\n\n Advice : Exercise Regularly and maintain like this..!') 

    

if __name__ == "__main__":
    app.run(debug=True)
