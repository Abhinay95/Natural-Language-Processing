# Importing Dataset
import pandas as pd

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['lables','message'])

# Data cleaning and Preprocessing

import nltk
#nltk.download('stopwords')
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stemmer = PorterStemmer()
corpus=[]

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]  
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

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

#Predictions
def predict_spam(sample_message):
    sample_message = re.sub('[^a-zA-Z]', ' ', sample_message)
    sample_message = sample_message.lower()
    sample_message = sample_message.split()
    sample_message = [stemmer.stem(word) for word in sample_message if not word in set(stopwords.words('english'))]
    sample_message = ' '.join(sample_message)
    
    temp = cv.transform([sample_message]).toarray()
    return spam_detect.predict(temp)

sample_message = 'IMPORTANT - You could be entitled up to £3,160 in compensation from mis-sold PPI on a credit card or loan. Please reply PPI for info or STOP to opt out.'

if predict_spam(sample_message):
  print('This is a SPAM message!')
else:
  print('This is a normal message.')
  
sample_message = 'Came to think of it. I have never got a spam message before.'

if predict_spam(sample_message):
  print('This is a SPAM message!')
else:
  print('This is a normal message.')
  
sample_message = 'You have still not claimed the compensation you are due for the accident you had. To start the process please reply YES. To opt out text STOP'

if predict_spam(sample_message):
  print('This is a SPAM message!')
else:
  print('This is a normal message.')
  
sample_message = 'A [redacted] loan for £950 is approved for you if you receive this SMS. 1 min verification & cash in 1 hr at www.[redacted].co.uk to opt out reply stop'

if predict_spam(sample_message):
  print('This is a SPAM message!')
else:
  print('This is a normal message.')