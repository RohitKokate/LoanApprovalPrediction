import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))


Gender = int(input("Gender: "))
Married = int(input("Married: "))
Dependents = int(input("Dependents: "))
Education = int(input("Education: "))
Self_Employed = int(input("Self_Employed: "))
Applicant_Income = np.log(int(input("Applicant_Income: ")))
Coapplicant_Income = np.log(int(input("Coapplicant_Income: ")))
Loan_Amount = np.log(int(input("Loan_Amount: ")))
Loan_Tenure = np.log(int(input("Loan_Tenure: ")))
Credit_History = int(input("Credit_History: "))
Property_Area = int(input("Property_Area: "))

Features = np.array([[Gender,Married,Dependents,Education,Self_Employed,Applicant_Income,Coapplicant_Income,
	Loan_Amount,Loan_Tenure,Credit_History,Property_Area]])

Prediction = model.predict(Features)

if Prediction==1:
	print("Loan Approved")
else:
	print("Loan Rejected")