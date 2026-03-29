import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,jsonify,render_template

application = Flask(__name__)
app=application

## import ridge and standard sclaer pickel
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))
        import pandas as pd

        data = pd.DataFrame([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]],
                    columns=['Temperature','RH','Ws','Rain','FFMC','DMC','ISI','Classes','Region'])

        new_data_scaled = standard_scaler.transform(data)
        result=ridge_model.predict(new_data_scaled)
        print("Prediction:", result)
        return render_template('home.html',results=result[0])  
    else:
        return render_template("home.html")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)