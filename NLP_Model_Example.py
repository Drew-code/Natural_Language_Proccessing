import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


nltk.download_shell()

messages = [line.rstrip() for line in open(r'smsspamcollection/SMSSpamCollection')] # getting just the text lines out of the data

print(len(messages)) #printing how many messages her are

for mess_no,message in enumerate(messages[:10]): #creating a for look to print the message and the no of the message
    print(mess_no,message)
    print('\n')



messages = pd.read_csv('smsspamcollection/SMSSpamCollection',sep = '\t', #opening the data again as a csv for a data frame
                       names=['label','message'])

print(messages.groupby('label').describe()) # describing the stats for each label, label being ham or spam

messages['length'] = messages['message'].apply(len) # creating a new column in the message data frame for number of words, naming it lenth

print(messages.head()) #printing the df so far

messages['length'].plot.hist(bins=50) #plotting texts by number words
plt.show()

messages['length'].describe() #describing the row column of the message data frame

print(messages[messages['length']==910]['message'].iloc[0]) # we realized from our description that 910 was the largest text, we then printed that

messages.hist(column='length',by='label',bins=60,figsize=(12,4)) # printing two graphs one for ham one for spam, and printing them by word count
plt.show()

mess = 'Sample message! Notice: it has punctuation!' #same message used to show punc function

nopunc = [c for c in mess if c not in string.punctuation] #removing each character that is punctuation
stopwords.words('english') # import words that are very common in the english language,

nopunc = ''.join(nopunc) #joining the string back together after all of the punctuation has been removed

nopunc.split() #splitting the new sentance into words now

clean_mess = [word for word in nopunc.split()if word.lower()not in stopwords.words('english')] #filting for words that are too common in english languague

def text_process(mess): ##function to to the punctuation removal and removal of common words in the english langugue
   nopunc = [char for char in mess if char not in string.punctuation ]
   nopunc = ''.join(nopunc)

   return [word for word in nopunc.split() if word.lower() not in stopwords.words() ]

messages['message'].head(5).apply(text_process) #applying function to each of the top 5 messages

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message']) #creating a vector for bag of words in message column

print(len(bow_transformer.vocabulary_)) #printing all of the words in the bag of words

mess4 = messages['message'][3] #creating an example

print(mess4)

bow4 = bow_transformer.transform([mess4]) #transforming the example into a bow

bow_transformer.get_feature_names()[4068] # after checking the shape of bow 4 we then checked to see which words it that appeared twice

messages_bow = bow_transformer.transform(mess['message']) #transforming all of the data in the message column

print(messages_bow.shape) #checking the shape of the new bag of words

print(messages_bow.nnz) #amount that are non-zeros

sparsity = (100 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])) #checking the spparsity of the message columns
print('sparsity: {}'.format(round(sparsity)))

tfidf_transformer = TfidfTransformer().fit(messages_bow) # term frequeceny-inverse document frequencey
tfidf4 = tfidf_transformer.transform(bow4)

tfidf_transformer.idf_[bow_transformer.vocabulary_['university']] #checking the IDF of the word university

messages_tfidf = tfidf_transformer.transform(messages_bow) #transforming the rest of the data

spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label']) #creating a model to detect which messages are spam. Fitting it to Niave Beyes

spam_detect_model.fit(tfidf4)[0]


msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],messages['label'],test_size=.3) #train/test/split the data

pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)), #this call, does everything we just did but in a much faster way
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])

pipline.fit(msg_train,label_train) #training the new model using pipeline

predictions = pipeline.predict(msg_test) #predicintg the new model

print(classification_report(predictions,label_test)) #checking for acurancy