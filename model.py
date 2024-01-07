import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score

df=pd.read_csv("mail_data.csv")
print(df.head())
data=df.where(pd.notnull(df),' ') #to replace the values in the DataFrame with empty spaces (' ') where the values are null (NaN).
data.info()
data.loc[data["Category"] == 'spam', 'Category'] = 0
data.loc[data["Category"] == 'ham', 'Category'] = 1
X=data['Message']
y=data['Category']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
vectorizer = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
x_train_features=vectorizer.fit_transform(x_train)
x_test_features=vectorizer.transform(x_test)
y_train=y_train.astype('int')
y_test=y_test.astype('int')

model=MultinomialNB()
model.fit(x_train_features,y_train)
y_pred=model.predict(x_test_features)
cf=confusion_matrix(y_test,y_pred)
print("Confusion matrix is:",cf)
acc_score= accuracy_score(y_test,y_pred)
print("Accuracy of the model is:",acc_score)


