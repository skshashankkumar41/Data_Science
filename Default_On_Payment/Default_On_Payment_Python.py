import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('Default_On_Payment.csv')
Columns_Name=list(df.columns)

#Meta Data of DataFrame
df.info()

#Summary of Age Column
Age_Summary_Before_Quantile=df['Age'].describe()

#Checking Quantile Range of Age Column
Age_Quantile=df['Age'].quantile(0.999)

for i in range(len(df['Age'])):
    if df['Age'][i]>75:
        df['Age'][i]=75

#Summary of Age column
Age_Summary_After_Quantile=df['Age'].describe()

#Checking How Many NULL Values in Housing Column
sum(df.Housing.isnull())

#Filling NULL Values of Housing Column with Highest Frequency Element
df.Housing.fillna(value='A152',inplace=True)

#Checking Missing Values in Num_Dependents
sum(df.Num_Dependents.isnull())

#Defautlers on Num_Dependents like table in R
pd.crosstab(index=df["Num_Dependents"], columns=df["Default_On_Payment"])

#Checking Missing Values in Job_Status
sum(df['Job_Status'].isnull())

#Defaulter on Job_Status
pd.crosstab(index=df["Job_Status"].fillna('Missing'), columns=df["Default_On_Payment"].fillna('Missing'))

#Filling the Missing Value with A174
df['Job_Status'].fillna(value='A174',inplace=True)

#Creating Dummy Variables for Job_Status THE HARD WAY

df['Dummy_Job_Status_A171']=0
df['Dummy_Job_Status_A172']=0
df['Dummy_Job_Status_A173']=0

for i in range(len(df['Dummy_Job_Status_A171'])):
    if df['Job_Status'][i]=='A171':
        df.iloc[i,17]=1
    else:
        df.iloc[i,17]=0

for i in range(len(df['Dummy_Job_Status_A172'])):
    if df['Job_Status'][i]=='A172':
        df.iloc[i,18]=1
    else:
        df.iloc[i,18]=0

for i in range(len(df['Dummy_Job_Status_A172'])):
    if df['Job_Status'][i]=='A173':
        df.iloc[i,19]=1
    else:
        df.iloc[i,19]=0

#Defaulter on Purpose Credit Taken
pd.crosstab(index=df["Purpose_Credit_Taken"].fillna('Missing'), columns=df["Default_On_Payment"].fillna('Missing'))

#Checking NULL in Purpose_Credit_Taken
sum(df.Purpose_Credit_Taken.isnull())

#Creating Dummy Varible for Purpose_Credit_Taken
df['Dummy_Purpose_Credit_Taken_Low']=0

for i in range(len(df['Dummy_Purpose_Credit_Taken_Low'])):
    if df['Purpose_Credit_Taken'][i]=='P41' or df['Purpose_Credit_Taken'][i]=='P43' or df['Purpose_Credit_Taken'][i]=='P48':
        df.iloc[i,20]=1
    else:
        df.iloc[i,20]=0

df['Dummy_Purpose_Credit_Taken_High']=0

for i in range(len(df['Dummy_Purpose_Credit_Taken_High'])):
    if df['Purpose_Credit_Taken'][i]=='P49' or df['Purpose_Credit_Taken'][i]=='P40' or df['Purpose_Credit_Taken'][i]=='P45' or df['Purpose_Credit_Taken'][i]=='P50' or df['Purpose_Credit_Taken'][i]=='P46' :
        df.iloc[i,21]=1
    else:
        df.iloc[i,21]=0

#Defaulter on Status_Checking_Accnt
pd.crosstab(index=df["Status_Checking_Accnt"].fillna('Missing'), columns=df["Default_On_Payment"].fillna('Missing'))

#Checking NULL in Purpose_Credit_Taken
sum(df.Status_Checking_Accnt.isnull())

#Creating Dummy Varible for Status_Checking_Accnt
df['Dummy_Status_Checking_Accnt_High']=0

for i in range(len(df['Dummy_Status_Checking_Accnt_High'])):
    if df['Status_Checking_Accnt'][i]=='S11':
        df.iloc[i,22]=1
    else:
        df.iloc[i,22]=0

df['Dummy_Status_Checking_Accnt_Medium']=0

for i in range(len(df['Dummy_Status_Checking_Accnt_Medium'])):
    if df['Status_Checking_Accnt'][i]=='S12':
        df.iloc[i,23]=1
    else:
        df.iloc[i,23]=0

#Defaulter on Credit_History
pd.crosstab(index=df["Credit_History"].fillna('Missing'), columns=df["Default_On_Payment"].fillna('Missing'))

#Checking NULL in Credit_History
sum(df.Credit_History.isnull())

#Creating Dummy Varible for Credit_History
df['Dummy_Credit_History_High']=0

for i in range(len(df['Dummy_Credit_History_High'])):
    if df['Credit_History'][i]=='A30' or df['Credit_History'][i]=='A31' :
        df.iloc[i,24]=1
    else:
        df.iloc[i,24]=0

df['Dummy_Credit_History_Low']=0

for i in range(len(df['Dummy_Credit_History_Low'])):
    if df['Credit_History'][i]=='A34':
        df.iloc[i,25]=1
    else:
        df.iloc[i,25]=0

#Defaulter on Years_At_Present_Employment
pd.crosstab(index=df["Years_At_Present_Employment"].fillna('Missing'), columns=df["Default_On_Payment"].fillna('Missing'))

#Checking NULL in Years_At_Present_Employment
sum(df.Years_At_Present_Employment.isnull())

#Creating Dummy Varible for Dummy_Years_At_Present_Employment_High
df['Dummy_Years_At_Present_Employment_High']=0

for i in range(len(df['Dummy_Years_At_Present_Employment_High'])):
    if df['Years_At_Present_Employment'][i]=='E71' or df['Years_At_Present_Employment'][i]=='E72' :
        df.iloc[i,26]=1
    else:
        df.iloc[i,26]=0

df['Dummy_Years_At_Present_Employment_Medium']=0

for i in range(len(df['Dummy_Years_At_Present_Employment_Medium'])):
    if df['Years_At_Present_Employment'][i]=='E73':
        df.iloc[i,27]=1
    else:
        df.iloc[i,27]=0

#Defaulter on Marital_Status_Gender
pd.crosstab(index=df["Marital_Status_Gender"].fillna('Missing'), columns=df["Default_On_Payment"].fillna('Missing'))

#Checking NULL in Marital_Status_Gender
sum(df.Marital_Status_Gender.isnull())

#Creating Dummy Varible for Marital_Status_Gender
df['Dummy_Marital_Status_Gender']=0

for i in range(len(df['Dummy_Marital_Status_Gender'])):
    if df['Marital_Status_Gender'][i]=='A91' or df['Marital_Status_Gender'][i]=='A92':
        df.iloc[i,28]=1
    else:
        df.iloc[i,28]=0

#Defaulter on Other_Debtors_Guarantors
pd.crosstab(index=df["Other_Debtors_Guarantors"].fillna('Missing'), columns=df["Default_On_Payment"].fillna('Missing'))

#Checking NULL in Other_Debtors_Guarantors
sum(df.Other_Debtors_Guarantors.isnull())

#Creating Dummy Varible for Other_Debtors_Guarantors
df['Dummy_Other_Debtors_Guarantors']=0

for i in range(len(df['Dummy_Other_Debtors_Guarantors'])):
    if df['Other_Debtors_Guarantors'][i]=='A103':
        df.iloc[i,29]=0
    else:
        df.iloc[i,29]=1

#Defaulter on Housing
pd.crosstab(index=df["Housing"].fillna('Missing'), columns=df["Default_On_Payment"].fillna('Missing'))

#Checking NULL in Housing
sum(df.Housing.isnull())

#Creating Dummy Varible for Housing
df['Dummy_Housing']=0

for i in range(len(df['Dummy_Housing'])):
    if df['Housing'][i]=='A152':
        df.iloc[i,30]=0
    else:
        df.iloc[i,30]=1

#Defaulter on Foreign_Worker
pd.crosstab(index=df["Foreign_Worker"].fillna('Missing'), columns=df["Default_On_Payment"].fillna('Missing'))

#Checking NULL in Foreign_Worker
sum(df.Foreign_Worker.isnull())

#Creating Dummy Varible for Foreign_Worker
df['Dummy_Foreign_Worker']=0

for i in range(len(df['Dummy_Foreign_Worker'])):
    if df['Foreign_Worker'][i]=='A201':
        df.iloc[i,31]=1
    else:
        df.iloc[i,31]=0

#Defaulter on Age
pd.crosstab(index=df["Age"].fillna('Missing'), columns=df["Default_On_Payment"].fillna('Missing'))

#Checking NULL in Age
sum(df.Foreign_Worker.isnull())

#Creating Dummy Varible for Age
df['Dummy_Age_Group']=0

for i in range(len(df['Dummy_Age_Group'])):
    if df['Age'][i]<30:
        df.iloc[i,32]=1
    else:
        df.iloc[i,32]=0

#Creating Dummy Varible for Credit_Amount
df['Dummy_Credit_Amount']=0

for i in range(len(df['Dummy_Credit_Amount'])):
    if df['Credit_Amount'][i]<5000:
        df.iloc[i,33]=0
    else:
        df.iloc[i,33]=1

#Creating Dummy Varible for Current_Address_Yrs
df['Dummy_Current_Address_Yrs']=0

for i in range(len(df['Dummy_Current_Address_Yrs'])):
    if df['Current_Address_Yrs'][i]==1:
        df.iloc[i,34]=0
    else:
        df.iloc[i,34]=1

#New DataFrame with Created Dummy Variables
new_df=df[['Dummy_Job_Status_A171','Dummy_Job_Status_A172','Dummy_Purpose_Credit_Taken_High','Dummy_Purpose_Credit_Taken_Low',
'Dummy_Status_Checking_Accnt_High', 'Dummy_Status_Checking_Accnt_Medium',
'Dummy_Credit_History_High','Dummy_Credit_History_Low','Dummy_Years_At_Present_Employment_High','Dummy_Years_At_Present_Employment_Medium',
'Dummy_Housing','Dummy_Credit_Amount','Dummy_Marital_Status_Gender','Dummy_Other_Debtors_Guarantors','Dummy_Foreign_Worker','Dummy_Current_Address_Yrs','Duration_in_Months',
'Dummy_Age_Group','Default_On_Payment']]

from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

features = "+".join(['Dummy_Job_Status_A171','Dummy_Job_Status_A172','Dummy_Purpose_Credit_Taken_High','Dummy_Purpose_Credit_Taken_Low',
'Dummy_Status_Checking_Accnt_High', 'Dummy_Status_Checking_Accnt_Medium',
'Dummy_Credit_History_High','Dummy_Credit_History_Low','Dummy_Years_At_Present_Employment_High','Dummy_Years_At_Present_Employment_Medium',
'Dummy_Housing','Dummy_Credit_Amount','Dummy_Marital_Status_Gender','Dummy_Other_Debtors_Guarantors','Dummy_Foreign_Worker','Dummy_Current_Address_Yrs','Duration_in_Months',
'Dummy_Age_Group'])

# get y and X dataframes based on this regression:
y, X = dmatrices('Default_On_Payment ~' + features, new_df, return_type='dataframe')

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

# Dummy_Purpose_Credit_Taken_High & Dummy_Purpose_Credit_Taken_Low have VIF > 1.5

y, X = dmatrices('Default_On_Payment ~' + 'Dummy_Purpose_Credit_Taken_Low', new_df, return_type='dataframe')

# For each X, calculate VIF and save in dataframe
vif_Dummy_Purpose_Credit_Taken_Low = pd.DataFrame()
vif_Dummy_Purpose_Credit_Taken_Low["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_Dummy_Purpose_Credit_Taken_Low["features"] = X.columns

y, X = dmatrices('Default_On_Payment ~' + 'Dummy_Purpose_Credit_Taken_High', new_df, return_type='dataframe')

# For each X, calculate VIF and save in dataframe
vif_Dummy_Purpose_Credit_Taken_High = pd.DataFrame()
vif_Dummy_Purpose_Credit_Taken_High["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_Dummy_Purpose_Credit_Taken_High["features"] = X.columns

#Remove Dummy_Purpose_Credit_Taken_High and check VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

features = "+".join(['Dummy_Job_Status_A171','Dummy_Job_Status_A172','Dummy_Purpose_Credit_Taken_Low',
'Dummy_Status_Checking_Accnt_High', 'Dummy_Status_Checking_Accnt_Medium',
'Dummy_Credit_History_High','Dummy_Credit_History_Low','Dummy_Years_At_Present_Employment_High','Dummy_Years_At_Present_Employment_Medium',
'Dummy_Housing','Dummy_Credit_Amount','Dummy_Marital_Status_Gender','Dummy_Other_Debtors_Guarantors','Dummy_Foreign_Worker','Dummy_Current_Address_Yrs','Duration_in_Months',
'Dummy_Age_Group'])

# get y and X dataframes based on this regression:
y, X = dmatrices('Default_On_Payment ~' + features, new_df, return_type='dataframe')

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

#Now VIF of All Column is <1.5
#Remove Dummy_Purpose_Credit_Taken_High from DataFrame
new_df=df[['Dummy_Job_Status_A171','Dummy_Job_Status_A172','Dummy_Purpose_Credit_Taken_Low',
'Dummy_Status_Checking_Accnt_High', 'Dummy_Status_Checking_Accnt_Medium',
'Dummy_Credit_History_High','Dummy_Credit_History_Low','Dummy_Years_At_Present_Employment_High','Dummy_Years_At_Present_Employment_Medium',
'Dummy_Housing','Dummy_Credit_Amount','Dummy_Marital_Status_Gender','Dummy_Other_Debtors_Guarantors','Dummy_Foreign_Worker','Dummy_Current_Address_Yrs','Duration_in_Months',
'Dummy_Age_Group','Default_On_Payment']]

#Now Sepereate Dependent and Independent Variable for Our Logistic Regression Model
X=new_df.iloc[:,:17].values
y=new_df.iloc[:,17:].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3)

#Training Data with Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)

#Predicting Data
pred=classifier.predict(X_test)

#Confusion Matrix and Accuracy of Model
from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix=confusion_matrix(y_test,pred)
accuracy=accuracy_score(y_test,pred)
