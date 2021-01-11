import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  
    if request.method == "POST": 
        
        Gender = request.form["Gender"]
        if Gender=="Male":
            Gender=1
        else:
            Gender=0
        
        Married = request.form["Marital_Status"]
        if Married=="Married":
            Married=1
        else:
            Married=0

        Dependents = request.form["Dependents"]
        if Dependents=="0":
            Dependents=0
        elif Dependents=="1":
            Dependents=1
        elif Dependents=="2":
            Dependents=2
        else:
            Dependents=3

        Education = request.form["Education"]
        if Education=="Graduate":
            Education=1
        else:
            Education=0

        Self_Employed = request.form["Self_Employed"]
        if Self_Employed=="Yes":
            Self_Employed=1
        else:
            Self_Employed=0

        Applicant_Income = int(request.form["Applicant_Income"])
        if Applicant_Income == 0:
            Applicant_Income=1
        Applicant_Income_Log = np.log(Applicant_Income)
        
        Coapplicant_Income = int(request.form["Coapplicant_Income"])
        if Coapplicant_Income == 0:
            Coapplicant_Income=1
        Coapplicant_Income_Log = np.log(Coapplicant_Income)

        Loan_Amount = int(request.form["Loan_Amount"])
        if Loan_Amount == 0:
            Loan_Amount=1
        Loan_Amount_Log = np.log(Loan_Amount)

        Loan_Tenure = int(request.form["Loan_Tenure"])
        if Loan_Tenure==0:
            Loan_Tenure=1
        Loan_Tenure_Log = np.log(Loan_Tenure)

        Credit_History = request.form["Credit_History"]
        if Credit_History=="Yes":
            Credit_History=1
        else:
            Credit_History=0

        Property_Area = request.form["Property_Area"]
        if Property_Area=="Urban":
            Property_Area=2
        elif Property_Area=="Semiurban":
            Property_Area=1
        else:
            Property_Area=0         
        
        Features = np.array([[Gender,Married,Dependents,Education,Self_Employed,Applicant_Income_Log,Coapplicant_Income_Log,Loan_Amount_Log,Loan_Tenure_Log,Credit_History,Property_Area]])

        Prediction = model.predict(Features)

        if Prediction==1:
            return render_template('index.html', prediction_text='Congratulations, your loan will be approved.')
        else:
            return render_template('index.html', prediction_text='Sorry, your loan will not be approved.')
   
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)