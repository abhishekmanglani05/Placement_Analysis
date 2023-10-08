import pickle
import numpy as np
import pandas as pd

df = pd.read_csv('C:\\Users\\Dell\\complete web development\\shree ram\\collegePlace.data')



df.head()


df.shape

df.info()

df.describe()

# df.corr()['PlacedOrNot']

df.isnull().sum()

print(df.duplicated().sum())

df.drop_duplicates(inplace=True)
print(df.duplicated().sum())


#EDA

import plotly.express as px

fig = px.scatter(df, x="CGPA", y="Internships", color="PlacedOrNot",
                 hover_data=['CGPA'])
fig.show()

px.histogram(df, x='PlacedOrNot', color='PlacedOrNot', barmode='group')

print("Maximum age of placed person :",df[(df['Age'] == df['Age'].max()) & (df["PlacedOrNot"] == 1)]['Age'].values[0] )
print("Minimum age of placed person :",df[(df['Age'] == df['Age'].min()) & (df["PlacedOrNot"] == 1)]['Age'].values[0] )

print("Max Internships Done by the Placed Student: ",df[(df['Internships'] == df['Internships'].max()) & (df['PlacedOrNot']==1)]['Internships'].values[0])
print("No of students who did max Internships and are placed: ",df[(df['Internships'] == df['Internships'].max()) & (df['PlacedOrNot']==1)]['Internships'].value_counts().values[0])

print("Min Internships Done by the Placed Person: ",df[(df['Internships'] == df['Internships'].min()) & (df['PlacedOrNot']==1)]['Internships'].values[0])
print("No of students who did min Internships and are placed: ",df[(df['Internships'] == df['Internships'].min()) & (df['PlacedOrNot']==1)]['Internships'].value_counts().values[0])

print("Max CGPA of Placed Student: ",df[(df['CGPA'] == df['CGPA'].max()) & (df['PlacedOrNot']==1)]['CGPA'].values[0])
print("No of students has max CGPA and are placed: ",df[(df['CGPA'] == df['CGPA'].max()) & (df['PlacedOrNot']==1)]['CGPA'].value_counts().values[0])

print("Min CGPA of Placed Person: ",df[(df['CGPA'] == df['CGPA'].min()) & (df['PlacedOrNot']==1)]['CGPA'].values[0])
print("No of students has min CGPA and are placed: ",df[(df['CGPA'] == df['CGPA'].min()) & (df['PlacedOrNot']==1)]['CGPA'].value_counts().values[0])

fig = px.box(df, y=['Internships','CGPA', 'Age'])
fig.show()

#Convert categorial data to numeric

df['Gender'] = df['Gender'].map({
    'Male': 1,
    'Female': 0})

df['Stream'].unique()

df['Stream'] = df['Stream'].map({
    'Electronics And Communication': 1,
    'Computer Science': 2,
    'Information Technology': 3,
    'Mechanical': 4,
    'Electrical': 5,
    'Civil': 6})

df.head()

X = df.drop(columns=['PlacedOrNot'],axis=1)
y = df['PlacedOrNot']

X.head()

y.head()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

# Without Scaling
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without Scaling and CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Without Scaling and With CV: ",scores.mean())

# With Scaling
clf.fit(X_train_scale,y_train)
y_pred = clf.predict(X_test_scale)
print("With Scaling and Without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train_scale, y_train, cv=10)
print("With Scaling and With CV: ",scores.mean())


from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(max_iter=1000, tol=1e-3)

# Without Scaling
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without Scaling and CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Without Scaling and With CV: ",scores.mean())

# With Scaling
clf.fit(X_train_scale,y_train)
y_pred = clf.predict(X_test_scale)
print("With Scaling and Without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train_scale, y_train, cv=10)
print("With Scaling and With CV: ",scores.mean())


from sklearn.linear_model import Perceptron
# this is same as SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None)

clf = Perceptron(tol=1e-3, random_state=0)
# Without Scaling
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without Scaling and CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Without Scaling and With CV: ",scores.mean())

# With Scaling
clf.fit(X_train_scale,y_train)
y_pred = clf.predict(X_test_scale)
print("With Scaling and Without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train_scale, y_train, cv=10)
print("With Scaling and With CV: ",scores.mean())


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)

#without scaling
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without Scaling and CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Without Scaling and With CV: ",scores.mean())

# With Scaling
clf.fit(X_train_scale,y_train)
y_pred = clf.predict(X_test_scale)
print("With Scaling and Without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train_scale, y_train, cv=10)
print("With Scaling and With CV: ",scores.mean())


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=10, random_state=0)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without Scaling and CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Without Scaling and With CV: ",scores.mean())

# With Scaling
clf.fit(X_train_scale,y_train)
y_pred = clf.predict(X_test_scale)
print("With Scaling and Without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train_scale, y_train, cv=10)
print("With Scaling and With CV: ",scores.mean())


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#clf = SVC(gamma='auto')

svc = SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
clf = GridSearchCV(svc, parameters)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without Scaling and CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Without Scaling and With CV: ",scores.mean())

# With Scaling
clf.fit(X_train_scale,y_train)
y_pred = clf.predict(X_test_scale)
print("With Scaling and Without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train_scale, y_train, cv=10)
print("With Scaling and With CV: ",scores.mean())

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#y_pred = gnb.fit(X_train, y_train).predict(X_test)
#print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without Scaling and CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Without Scaling and With CV: ",scores.mean())

# With Scaling
clf.fit(X_train_scale,y_train)
y_pred = clf.predict(X_test_scale)
print("With Scaling and Without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train_scale, y_train, cv=10)
print("With Scaling and With CV: ",scores.mean())

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without Scaling and CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Without Scaling and With CV: ",scores.mean())

from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without Scaling and CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Without Scaling and With CV: ",scores.mean())

# With Scaling
clf.fit(X_train_scale,y_train)
y_pred = clf.predict(X_test_scale)
print("With Scaling and Without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train_scale, y_train, cv=10)
print("With Scaling and With CV: ",scores.mean())


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without Scaling and CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Without Scaling and With CV: ",scores.mean())


# With Scaling
clf.fit(X_train_scale,y_train)
y_pred = clf.predict(X_test_scale)
print("With Scaling and Without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train_scale, y_train, cv=10)
print("With Scaling and With CV: ",scores.mean())

from sklearn.metrics import precision_score, recall_score, f1_score
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("With CV: ",scores.mean())
print("Precision Score: ", precision_score(y_test, y_pred))
print("Recall Score: ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


param_grid = {
    'bootstrap': [False,True],
    'max_depth': [5,8,10, 20],
    'max_features': [3, 4, 5, None],
    'min_samples_split': [2, 10, 12],

    'n_estimators': [100, 200, 300]
}

rfc = RandomForestClassifier()

clf = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 1)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy: ",accuracy_score(y_test,y_pred))
print(clf.best_params_)
print(clf.best_estimator_)


clf = RandomForestClassifier(bootstrap=False, max_depth=5,max_features=None,
                             min_samples_split=2,
                             n_estimators=100, random_state=0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("With CV: ",scores.mean())
print("Precision Score: ", precision_score(y_test, y_pred))
print("Recall Score: ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))

import pickle

pickle.dump(clf,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

age = 22
gender = 1 # 1=Male, 0=Female
stream = 3  # Electronics And Communication': 1,
#              'Computer Science': 2,
#              'Information Technology': 3,
#              'Mechanical':4,
#              'Electrical':5,
#              'Civil':6
Internships = 2
CGPA = 6
Hostel = 1 # 1= stay in hostel, 0=not staying in hostel
HistoryOfBacklogs = 0 # 1 = had backlogs, 0=no backlogs

prediction = clf.predict([[age,gender,stream,Internships,CGPA,Hostel,HistoryOfBacklogs]])
prediction