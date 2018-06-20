import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

new_data=train.append(test)

#Removing Unneccasry Columns

new_data.drop(['PassengerId','Ticket'],axis=1,inplace=True)

#Preprocessing Embarked 
new_data.Embarked.fillna(value='S',inplace=True)

#Preprocessing Name and Age Accordingly

new=[]
for i in new_data['Name']:
    k=i.split(',')
    m=k[1].split('.')
    new.append(m[0].strip())
    
new_data.iloc[:,4]=new

new_data.Name[new_data.Name=='Dona']='Mrs'
        
fig = plt.figure(figsize=(15,6))
i=1
for title in new_data['Name'].unique():
    fig.add_subplot(3, 6, i)
    plt.title('Title : {}'.format(title))
    new_data.Survived[new_data['Name'] == title].value_counts().plot(kind='pie')
    i += 1
    
replacement = {
    'Don': 0,
    'Rev': 0,
    'Jonkheer': 0,
    'Capt': 0,
    'Mr': 1,
    'Dr': 2,
    'Col': 3,
    'Major': 3,
    'Master': 4,
    'Miss': 5,
    'Mrs': 6,
    'Mme': 7,
    'Ms': 7,
    'Mlle': 7,
    'Sir': 7,
    'Lady': 7,
    'the Countess': 7
}


new_data.Age.fillna(value=-1,inplace=True)

median_age=dict()
for i in new:
    median=new_data.Age[(new_data['Age']!=-1) & (new_data['Name']==i)].median()
    median_age[i]=median
    
for i in range(1309):
    if new_data.iloc[i,0]==-1:
        new_data.iloc[i,0]=median_age[new_data.iloc[i,4]]        


new_data.Age=StandardScaler().fit_transform(new_data.Age.values.reshape(-1,1))
new_data['Name']=new_data['Name'].apply(lambda x:replacement.get(x))


new_data.Name=StandardScaler().fit_transform(new_data.Name.values.reshape(-1,1))

#Preprocessing Fare
new_data.Fare.fillna(-1,inplace=True)
median_dict={}

for i in new_data.Pclass.unique():
    median=new_data.Fare[(new_data.Fare!=-1) & (new_data.Pclass==i)].median()
    median_dict[i]=median
    
for i in range(1309):
    if new_data.iloc[i,4]==-1:
        new_data.iloc[i,4]=median_dict[new_data.iloc[i,6]]
        
new_data.Fare=StandardScaler().fit_transform(new_data.Fare.values.reshape(-1,1))
        
#Normalizing Pclass
        
new_data['Pclass'] = StandardScaler().fit_transform(new_data['Pclass'].values.reshape(-1, 1))

#Preprocessing Parch

#Plotting Cabin with Parch to see any Correlation
fig = plt.figure(figsize=(15,8))
i = 0
for parch in new_data['Parch'].unique():
    fig.add_subplot(2, 4, i+1)
    plt.title('Parents / Child : {}'.format(parch))
    new_data.Survived[new_data['Parch'] == parch].value_counts().plot(kind='pie')
    i += 1
    
replacement = {
    6: 0,
    4: 0,
    5: 1,
    0: 2,
    2: 3,
    1: 4,
    3: 5
}
new_data['Parch'] = new_data['Parch'].apply(lambda x: replacement.get(x))
new_data['Parch'] = StandardScaler().fit_transform(new_data['Parch'].values.reshape(-1, 1))
    
#Preprocessing Embarked

replacement = {
    'S': 0,
    'Q': 1,
    'C': 2
}

new_data['Embarked'] = new_data['Embarked'].apply(lambda x: replacement.get(x))
new_data['Embarked'] = StandardScaler().fit_transform(new_data['Embarked'].values.reshape(-1, 1))

#Preprocessing Sibsp

#Plotting Cabin with Sibsp to see any Correlation
fig = plt.figure(figsize=(15,8))
i = 1
for sibsp in new_data['SibSp'].unique():
    fig.add_subplot(2, 4, i)
    plt.title('SibSp : {}'.format(sibsp))
    new_data.Survived[new_data['SibSp'] == sibsp].value_counts().plot(kind='pie')
    i=i+1
replacement = {
    5: 0,
    8: 0,
    4: 1,
    3: 2,
    0: 3,
    2: 4,
    1: 5
}

new_data['SibSp'] = new_data['SibSp'].apply(lambda x: replacement.get(x))
new_data['SibSp'] = StandardScaler().fit_transform(new_data['SibSp'].values.reshape(-1, 1))

#Preprocessing Cabin

new_data.Cabin.fillna(value='U',inplace=True)

new_data.Cabin.unique()

#Subsetting only First Character of Cabin 
new_data.Cabin=new_data.Cabin.apply(lambda x:x[0])

#Plotting Cabin with Survived to see any Correlation
fig = plt.figure(figsize=(15,12))
i = 1
for cabin in new_data['Cabin'].unique():
    fig.add_subplot(3, 3, i)
    plt.title('Cabin : {}'.format(cabin))
    new_data.Survived[new_data['Cabin'] == cabin].value_counts().plot(kind='pie')
    i += 1
    

replacement  = {
    'T': 0,
    'U': 1,
    'A': 2,
    'G': 3,
    'C': 4,
    'F': 5,
    'B': 6,
    'E': 7,
    'D': 8
}

new_data['Cabin'] = new_data['Cabin'].apply(lambda x: replacement.get(x))
new_data['Cabin'] = StandardScaler().fit_transform(new_data['Cabin'].values.reshape(-1, 1))

#Encodeing Sex

from sklearn.preprocessing import LabelEncoder
new_data.Sex=LabelEncoder().fit_transform(new_data.Sex)


#Spliting Trianing and Test
X=new_data.iloc[:,:9].values
y=new_data.iloc[:,9:].values


X_train=X[:891,:]
X_test=X[891:,:]
y_train=y[:891,:]

#Tuning for Random Forest 
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
param_grid = {
    'n_estimators':[100,200,300,400,500,600],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth':[10,20,30,40,50,60]
}

clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv= 5)

y_train=y_train.reshape(891,)

clf.fit(X_train,y_train)

print(clf.best_estimator_)

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=600, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

classifier.fit(X_train,y_train)

pred=classifier.predict(X)

pred=pd.DataFrame(pred)

pred.to_csv('Best_Titanic.csv')
