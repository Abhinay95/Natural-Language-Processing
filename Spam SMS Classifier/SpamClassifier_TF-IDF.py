# Importing Dataset
import pandas as pd

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['lables','message'])

# Data cleaning and Preprocessing

import nltk
#nltk.download('stopwords')
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
corpus=[]

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]  
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['lables'])
y = y.iloc[:,1].values

#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=0)

# training model by using naive baye's classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect = MultinomialNB()
spam_detect.fit(x_train,y_train)

y_pred = spam_detect.predict(x_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)

# Accuracy Score
score = accuracy_score(y_test,y_pred)