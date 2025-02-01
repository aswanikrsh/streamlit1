import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('passenger_survival_dataset.csv')

print(df.head())

encoder = LabelEncoder()
df['Name'] = encoder.fit_transform(df['Name'])
df['Gender'] = encoder.fit_transform(df['Gender'])
df['Class'] = encoder.fit_transform(df['Class'])
df['Seat_Type'] = encoder.fit_transform(df['Seat_Type'])
print(df)

X = df[['Passenger_ID','Name','Age','Gender','Class','Seat_Type','Fare_Paid']]
y = df['Survival_Status']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

Classifier = DecisionTreeClassifier()
model = Classifier.fit(X_train,y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_pred,y_test)
report = classification_report(y_pred,y_test)
print(accuracy)
print(report)

pickle.dump(model,open('model.pkl','wb'))