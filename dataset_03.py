import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score


import nltk
nltk.download('stopwords')

from sklearn.metrics import classification_report, confusion_matrix
# printing the stopwords in english
print(stopwords.words('english'))

news_dataset = pd.read_excel('Datasets/fake_new_dataset.xlsx')
print(news_dataset.shape)
print(news_dataset.head())
# counting the number of missing values in the dataset
# print(news_dataset.isnull().sum())

# replacing the null values empty string
# news_dataset = news_dataset.fillna('')
# news_dataset['content'] = news_dataset['author']+ ' ' + news_dataset['title']
# print(news_dataset['content'])

# separating the data and label

X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']
print('X', X)
print('Y', Y)


prot_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [prot_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

print(news_dataset['text'])
news_dataset['text'] = news_dataset['text'].apply(stemming)
print('stemming facn: ',news_dataset['text'])
#
X = news_dataset['text'].values
Y = news_dataset['label'].values
print('X ', X)
print('Y ', Y)

# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
print('X vect')
print(X)

# Splitting the dataset to training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, stratify=Y, random_state=2)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
#
# Training model: PA
model = PassiveAggressiveClassifier()
model.fit(X_train, Y_train)
#
# Evaluation

# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('training data accuracy : ', training_data_accuracy)

# accuracy score on testing data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('testing data accuracy : ', testing_data_accuracy)

print('confusion_matrix: ',confusion_matrix(Y_test, X_test_prediction) )
print(classification_report(Y_test, X_test_prediction))
plt.figure()
sns.heatmap(pd.crosstab(Y_test, X_test_prediction), annot=True, fmt='d')
plt.xlabel('Target')
plt.ylabel('outcome')
plt.show()
#
#
# X_new = X_test[0]
# prediction = model.predict(X_new)
#
# if(prediction[0] == 0):
#     print('The news is Real')
# else:
#     print('The news is Fake')
#
