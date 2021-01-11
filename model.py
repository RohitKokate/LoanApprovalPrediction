import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

df = pd.read_csv('Loan_Approval_1.csv')

df['Coapplicant_Income'].replace(0,1,inplace=True)
df['Loan_Amount'].replace(0, df['Loan_Amount'].median(), inplace=True)
df['Loan_Tenure'].fillna(df['Loan_Tenure'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

df['Applicant_Income']=np.log(df['Applicant_Income'])
df['Coapplicant_Income']=np.log(df['Coapplicant_Income'])
df['Loan_Amount']=np.log(df['Loan_Amount'])
df['Loan_Tenure']=np.log(df['Loan_Tenure'])

to_numeric = {'Male':1, 'Female':0,
               'Yes':1, 'No':0,
               'Graduate':1, 'Not Graduate':0,
               'Urban':2, 'Semiurban':1, 'Rural':0,
               '0':0,'1':1,'2':2, '3+':3}

df = df.applymap(lambda s: to_numeric.get(s) if s in to_numeric else s)

df.drop("Loan_ID", axis=1, inplace=True)

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# print(y_test)
# print(y_pred)

evaluation = f1_score(y_test, y_pred)
print(evaluation)

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))

# print(model.predict([[1,1,0,1,0,25000,30000,300000,360,1,1]]))

