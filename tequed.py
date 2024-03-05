import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


dataset=pd.read_csv('C:\\Users\\mrnis\\OneDrive\\Desktop\\INTERNSHIP\\Loan_Train.csv')
#print(dataset.head(10))

df=dataset.info()
#print(df)

dataset.drop("Gender" ,axis=1,inplace=True)
dataset.drop("Married" ,axis=1,inplace=True)
dataset.drop("CoapplicantIncome" ,axis=1,inplace=True)
dataset.drop("Loan_ID" ,axis=1,inplace=True)
sns.heatmap(dataset.isnull())
#plt.show()

X=dataset.iloc[:,: -1].values
Y=dataset.iloc[:,-1].values

#print(Y)
#dataset.info()

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

A=make_column_transformer((OneHotEncoder(categories="auto"),[2]),remainder="passthrough")
X=A.fit_transform(X)
#print(X)

B=make_column_transformer((OneHotEncoder(categories="auto"),[3]),remainder="passthrough")
X=B.fit_transform(X)
#print(X)


C=make_column_transformer((OneHotEncoder(categories="auto"),[9]),remainder="passthrough")
X=C.fit_transform(X)
#print(X)
'''

data=dataset.head(15)

A=data["ApplicantIncome"]
B=data["LoanAmount"]
plt.scatter(A,B,c='r',s=55)
plt.title("INCOME VS LOAN")
plt.xlabel("Income of applicant")
plt.ylabel("Looan Amount")
#plt.show()


a=data['ApplicantIncome'] 
b=data['LoanAmount'] 
c=data['Loan_Amount_Term'] 
d=data['Property_Area'] 
plt.subplot(221) 
#plt.hist(a) 

plt.title("ApplicantIncome") 
plt.subplot(222) 
#plt.hist(b) 

plt.title("LoanAmount") 
plt.subplot(223) 
#plt.hist(c) 

plt.title("Loan_Amount_Term") 
plt.subplot(224) 
#plt.hist(d) 

plt.title("Property_Area")
#plt.show()

from matplotlib import style
style.use("ggplot") 
x=data['Education'] 
y=data['Dependents']       
plt.plot(x,y)
plt.xlabel("Education") 
plt.ylabel("Depdendents") 
plt.title("Graduate VS Dependents")
#plt.show()


import plotly.express as px
fig = px.bar(data, x="Education", y="LoanAmount",
color='Loan_Status', barmode='group',height=400)
fig.update_layout(title_text='LoanAmount, Loan Status approved Educationwise')
#fig.show()
'''
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=6)
#print(X_train)
p#rint(Y_train)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,Y_train)
y_pred=logreg.predict(X_test)
#print(y_pred)

from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(Y_test,y_pred)
print(cnf_matrix)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,fmt='g')
plt.title("Confusion Matrix")
plt.xlabel("Actual Label")
plt.ylabel("Predicted Label")
plt.show()

print(metrics.accuracy_score(Y_test,y_pred))
print(metrics.precision_score(y_pred,Y_test))
print(metrics.recall_score(y_pred,Y_test))

'''
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,Y_train)
y_pred=logreg.predict([[0,0,1,1,0,0,1,2,4500,45,350,2]])
print(y_pred)


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,Y_train)
y_pred=logreg.predict([[1,0,0,1,1,0,0,5,10,15,700,0]])
print(y_pred)

'''


