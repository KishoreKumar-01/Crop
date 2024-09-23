from flask import Flask, flash,render_template,request
import numpy as np
import csv
import joblib
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import timedelta
from datetime import datetime

app = Flask(__name__)

classes = ['Paddy', 'Ragi','Cotton', 'Sugarcane','Chilli', 'Pigeon Pea', 
        'Coconut', 'Onion', 'Banana','Mangoes', 'Turmeric', 'Groundnut', 
        'Maize', 'Brinjal', 'Carrot', 'Beans']

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result',methods=['POST','GET'])
def result():

    values = []
    if request.method == 'POST':
        values.append(float(request.form.get('nitrogen')))
        values.append(float(request.form.get('phosphorous')))
        values.append(float(request.form.get('potassium')))
        values.append(float(request.form.get('temperature')))
        values.append(float(request.form.get('humidity')))
        values.append(float(request.form.get('ph')))
        values.append(float(request.form.get('rainfall')))
        
        model = joblib.load('E:\Crop Recommendation Price\predict1.pkl')
        predict_pro = model.predict_proba([values])
        list_proba = []
        for i in [-1, -2, -3, -4, -5]:
            list_proba.append(classes[np.argsort(np.max(predict_pro, axis=0))[i]])
        return render_template('result.html',probab = list_proba)

@app.route('/price')
def price():
    return render_template('price.html', data1 = classes)

@app.route('/crop_name',methods=['POST','GET'])
def crop_name():
    if request.method == 'POST':
        crop = request.form.get('crop')
        return render_template('price.html',data1 = [crop])

@app.route('/price_predict',methods=['POST','GET'])
def price_predict():
    if request.method == 'POST':
        # price = joblib.load('E:\Crop Recommendation Price\templates\price.pkl')
        price = joblib.load('E:\Crop Recommendation Price\price.pkl')
        pr = {'Banana':1, 'Beans':2, 'Brinjal':3, 'Carrot':4, 'Coconut':5, 'Cotton':6, 
        'Chilli':7, 'Groundnut':8, 'Maize':9, 'Mangoes':10, 'Onion':11, 'Paddy':12, 
        'Pigeon Pea':13, 'Ragi':14, 'Sugarcane':15, 'Turmeric':16}
        
        loc = {'Banana': 'Andhra Pradesh', 'Beans':'Assam', 'Brinjal':'Andhra Pradesh', 'Carrot':'Andhra Pradesh', 
        'Coconut':'Karnataka', 'Cotton':'Gujarat', 'Chilli':'Chattisgarh', 'Groundnut':'Madhya Pradesh', 
        'Maize':'Chattisgarh', 'Mangoes':'Haryana', 'Onion':'Goa', 'Paddy':'Bihar', 'Pigeon Pea':'Gujarat', 
        'Ragi':'Gujarat', 'Sugarcane':'Chattisgarh', 'Turmeric':'Karnataka'}
        d = request.form.get('date')
        a = request.form.get('area')
        dif = request.form.get('dif')
        Begindate = datetime.strptime(d, "%Y-%m-%d")
        if dif == '1w':
            Enddate = Begindate + timedelta(days=7)
        elif dif == '1m':
            Enddate = Begindate + timedelta(days=30)
        elif dif == '3m':
            Enddate = Begindate + timedelta(days=90)

        print(Begindate)
        print(Enddate)
        y = int(str(Enddate)[0:4])
        m = int(str(Enddate)[5:7])
        da = int(str(Enddate)[8:10])
        lo = loc[a]
        n = pr[a]
        result = price.predict([[n, y, m, da]])
        num = int(result)
        return render_template('price.html', data = num, l = lo ,view = 'style=display:block')

@app.route('/real')
def real():
    return render_template('real.html')

@app.route('/l1')
def l1():
    l1 = [[100.0,99,150,35,19,7,11]]
    return render_template('predict1.html',data=l1,msg='Land 1')

@app.route('/l2')
def l2():
    l2 = [[1,1,1,10,14,1,11]]
    return render_template('predict1.html',data=l2,msg='Land 2')

@app.route('/l3')
def l3():
    l3 = [[71.4,4,70.8,33,62,0,81]]
    return render_template('predict1.html',data=l3,msg='Land 3')

#@app.route('/l4')
#def l4():
    #l4 = [[62.2,4.2,66.6,35,59,0,110]]
    #return render_template('predict1.html',data=l4,msg='Land 4')

if __name__ == '__main__':
    app.run(debug=True)