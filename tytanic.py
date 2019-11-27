import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 50,max_depth = 25,min_samples_split = 10)
 
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train["Embarked"] = train["Embarked"].fillna("S")
train.Embarked = train.Embarked.replace('C',0)
train.Embarked = train.Embarked.replace('S',1)
train.Embarked = train.Embarked.replace('Q',2)
train.Sex = train.Sex.replace('female',1)
train.Sex = train.Sex.replace('male',0)
train["Family"] = train["SibSp"] + train["Parch"] + 1


train['Initial'] = train['Cabin'].map(lambda x: str(x)[0])
pd.set_option('display.max_rows', None)
#train

train2 = [train]
for train in train2:
        train['Family'] = train['Family'].replace(1,'Single')
        train['Family'] = train['Family'].replace([2,3,4],'Middle')
        train['Family'] = train['Family'].replace([5,6,7],'Large')
        #del train['Family']
Family_Dot = {"Single": 1, "Middle": 2, "Large": 3} 
for train in train2: 
        train['Family'] = train['Family'].map(Family_Dot) 
        train['Family'] = train['Family'].fillna(0)        


train3 = [train]

for train in train3: 
        train['Salutation'] = train.Name.str.extract(' ([A-Za-z]+).', expand=False) 
for train in train3: 
        train['Salutation'] = train['Salutation'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        train['Salutation'] = train['Salutation'].replace('Mlle', 'Miss')
        train['Salutation'] = train['Salutation'].replace('Ms', 'Miss')
        train['Salutation'] = train['Salutation'].replace('Mme', 'Mrs')
        del train['Name']
Middle_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 
for train in train3: 
        train['Salutation'] = train['Salutation'].map(Middle_mapping) 
        train['Salutation'] = train['Salutation'].fillna(0)

train["Embarked"] = train["Embarked"].fillna(train.Embarked.mean())
train["Age"] = train["Age"].fillna(30)
df = train.drop(['SibSp','Ticket','Parch','Cabin'],axis=1)
df_corr = df.corr()
print(df_corr)
sns.heatmap(df_corr, vmax=1, vmin=-1, center=0)

train_data = df.values
X = train_data[:, 2:] 
y = train_data[:, 1]

forest = forest.fit(X, y)

test["Age"] = test["Age"].fillna(30)
test["Embarked"] = test["Embarked"].fillna("S")
test.Embarked = test.Embarked.replace('C',0)
test.Embarked = test.Embarked.replace('S',1)
test.Embarked = test.Embarked.replace('Q',2)
test.Sex = test.Sex.replace('female',1)
test.Sex = test.Sex.replace('male', 0)
test["Family"] = test["SibSp"] + test["Parch"] + 1
test["Fare"] = test["Fare"].fillna(test.Fare.mean())
test2 = [test]
for test in test2:
        test['Family'] = test['Family'].replace(1,'Single')
        test['Family'] = test['Family'].replace([2,3,4],'Middle')
        test['Family'] = test['Family'].replace([5,6,7],'Large')
        #del test['Family']
Family_Dot = {"Single": 1, "Middle": 2, "Large": 3} 
for test in test2: 
        test['Family'] = test['Family'].map(Family_Dot) 
        test['Family'] = test['Family'].fillna(0)
        
test3 = [test]

for test in test3: 
        test['Mid_Name'] = test.Name.str.extract(' ([A-Za-z]+).', expand=False) 
for test in test3: 
        test['Mid_Name'] = test['Mid_Name'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        test['Mid_Name'] = test['Mid_Name'].replace('Mlle', 'Miss')
        test['Mid_Name'] = test['Mid_Name'].replace('Ms', 'Miss')
        test['Mid_Name'] = test['Mid_Name'].replace('Mme', 'Mrs')
        del test['Name']
Middle_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rarecharactor": 5} 
for test in test3: 
        test['Mid_Name'] = test['Mid_Name'].map(Middle_mapping) 
        test['Mid_Name'] = test['Mid_Name'].fillna(0)
df2 = test.drop(['SibSp','Ticket','Parch','Cabin'],axis=1)
#df2

test_data = df2.values
xs_test = test_data[:, 1:]
output = forest.predict(xs_test)

print(len(test_data[:,0]), len(output))
zip_data = zip(test_data[:,0].astype(int), output.astype(int))
predict_data = list(zip_data)

import csv
with open("Predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])

