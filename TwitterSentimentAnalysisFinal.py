# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:06:22 2019

@author: achowdh2
"""
#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('TwitterTrainDataset.csv', error_bad_lines=False, encoding='latin-1',skiprows=0, nrows=10000)
dataset = dataset.drop(['ItemID'],axis=1)

#Number of words in each tweet
dataset['word_count'] = dataset['SentimentText'].apply(lambda x: len(str(x).split(" ")))
dataset[['SentimentText','word_count']].head()

#number of characters in each tweet
dataset['char_count'] = dataset['SentimentText'].str.len() ## this also includes spaces
dataset[['SentimentText','char_count']].head()

#function to calculate the average number of words in a tweet
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

#avg number of words in each tweet
dataset['avg_word'] = dataset['SentimentText'].apply(lambda x: avg_word(x))
dataset[['SentimentText','avg_word']].head()

#More libraries
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop = stopwords.words('english')

#Removing urls
Sentiment = []
for i in range(0, 10000):
    SentimentText = re.sub(r'http\S+', '', dataset['SentimentText'][i])
    Sentiment.append(SentimentText)
    
df = pd.DataFrame({'Text':Sentiment})
#concatenating the data frames
dataset = pd.concat([dataset, df], axis=1, sort=False)
#dataset = dataset.drop(['SentimentText'],axis=1)

#number of stop words
dataset['stopwords'] = dataset['Text'].apply(lambda x: len([x for x in x.split() if x in stop]))
dataset[['Text','stopwords']].head()

#number of hastags
dataset['hastags'] = dataset['Text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
dataset[['Text','hastags']].head()

#Number of Numerics
dataset['numerics'] = dataset['Text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
dataset[['Text','numerics']].head()

#number of uppercase letters
dataset['upper'] = dataset['Text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
dataset[['Text','upper']].head()

#converting to lowercase removing special characters and keeping only characters
dataset['Text'] = dataset['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
dataset['Text'].head()

dataset['Text'] = dataset['Text'].str.replace('[^\w\s]','')
dataset['Text'].head()

dataset['Text'] = dataset['Text'].str.replace('[^a-zA-Z]',' ')
dataset['Text'].head()


#removing stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
dataset['Text'] = dataset['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
dataset['Text'].head()

#checking for most frequent words and removing them
freq = pd.Series(' '.join(dataset['Text']).split()).value_counts()[:15]
freq

freq = list(freq.index)
dataset['Text'] = dataset['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
dataset['Text'].head()

#checking for least frequent words and removing them
freq = pd.Series(' '.join(dataset['Text']).split()).value_counts()[-15:]
freq

freq = list(freq.index)
dataset['Text'] = dataset['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
dataset['Text'].head()

#correcting spellings
from textblob import TextBlob
dataset['Text'] = dataset['Text'][:10000].apply(lambda x: str(TextBlob(x).correct()))

#tokenization and lemmatization
from nltk.stem import WordNetLemmatizer
corpus = []
for i in range(0, 10000):
    review = dataset['Text'][i]
    review = review.split()
    WNL = WordNetLemmatizer()
    review = [WNL.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 12300, lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,1))
X = tfidf.fit_transform(corpus).toarray()

#Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer(max_features = 12300, lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,1))
X = CV.fit_transform(corpus).toarray()

#general sentiment analysis
dataset['Label'] = dataset['Text'].apply(lambda x: TextBlob(x).sentiment[0] )
dataset[['Text','Label']].head()  

X1 = dataset.iloc[:, [2, 3, 4, 6, 10]].values

X2 = np.concatenate((X, X1), axis=1)

y = dataset.iloc[:, 0].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_NB = confusion_matrix(y_test, y_pred)
#63.25% accurate

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RFC = confusion_matrix(y_test, y_pred) #73.6% accurate #F1 score = 0.648

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_DT = confusion_matrix(y_test, y_pred) #65.55% accurate

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_KSVM = confusion_matrix(y_test, y_pred)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_LR = confusion_matrix(y_test, y_pred)

