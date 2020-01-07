# Natural_Language_Proccessing
Natural Language Proccessing Example  
## Introduction  
Natural Language Proccessing, is used to classify writing and put it into groups. This example is using a dataset of text  
messages and determining if each message is spam or a legitimate message.
 
## Prerequisites
1. Python 3.7 or newer  
2. Scikit-Learn module for Python  
3. Pandas module for Python  
4. Numpy Modules for Python
  
## Walkthough  
Start by importing all modules you will need at the beginning of your script. These include: Pandas, Scikit-Learn,  
Numpy, nltk, string and seaborn.  

```
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
```  
To download the dataset for this example, run ```nltk.download_shell() ``` and wait for the menu to appear.  
Now press "d" for download. Next type in "stopwords".  This only needs to be done once. Comment out the line after  
the dataset is downloaded.  

Next you need to strip out the actual message using the line below. To check how many messages there are in the dataset,  
use ```len(messages)```  
```
messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')] # getting just the text lines out of the data

print(len(messages)) #printing how many messages her are
```  
Next use a for loop to look at the messages. This will show you the message number as well as whether or not this message  
is ham or spam.  
```
for mess_no,message in enumerate(messages[:10]): #creating a for look to print the message and the no of the message
    print(mess_no,message)
    print('\n')
```  
To get a cleaner look at the data, open the file as a .csv. Now you are able to get a statistical view of whether or not  
the message is spam or ham.  
```
messages = pd.read_csv('smsspamcollection/SMSSpamCollection',sep = '\t', #opening the data again as a csv for a data frame
                       names=['label','message'])

print(messages.groupby('label').describe()) # describing the stats for each label, label being ham or spam
```  
Create a new column in the dataframe you just made and name it "length", showing the length of each messege.  
After that, lets see what the new dataframe looks like by calling ```.head()``` on messages.  
```
messages['length'] = messages['message'].apply(len) # creating a new column in the message data frame for number of words, naming it lenth

print(messages.head()) #printing the df so far
```  
Now we can plot all texts by word count to take a look at the message distribution by message size.
```
messages['length'].plot.hist(bins=50) #plotting texts by number words
plt.show()
```
Next take a look at the distribution of ham and spam messages by word count to see if there is any correlation.  
```
messages.hist(column='length',by='label',bins=60,figsize=(12,4)) # printing two graphs one for ham one for spam, and printing them by word count
plt.show()
```  
Now we need to clean all of the messages in the dataset. This will be done in a few steps. First, by removing  
all characters that are punctuation. Second, by importing common words in the english language. This is done  
to help the model cut down on time. There is no reason to feed it words that do not indicate whether or not the message is spam.  
Next, the string has to be put back together without all of punctuation, then split the sentences into words that are then  
put into the "bag of words". All of these steps can be done with one function.  
```
def text_process(mess): ##function to to the punctuation removal and removal of common words in the english langugue
   nopunc = [char for char in mess if char not in string.punctuation ]
   nopunc = ''.join(nopunc)

   return [word for word in nopunc.split() if word.lower() not in stopwords.words() ]
```
To show an example of this function at work, this can be applied to the first 5 rows in the dataset as shown below.  
```
messages['message'].head(5).apply(text_process) #applying function to each of the top 5 messages
```
Next a vector will be created for the whole dataset. This will count the occurance of each word in every message.  
Then each message has to be transformed so it is uniform for the model to learn. Lastly a model needs to be put into  
the pipline. Pipline is used to streamline this proccess.
```
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)), #this call, does everything we just did but in a much faster way
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])
```
The dataset now is ready to be split into a test and a train dataset. This is done by calling train/test/split as seen  
below. Lastly, the model can be trained using the ```.fit()``` method and predictions are made using the  
```.predict()``` method of our pipline model.  
```
pipline.fit(msg_train,label_train) #training the new model using pipeline

predictions = pipeline.predict(msg_test) #predicintg the new model
```
As with most other models we are now able to use the classifcation report to analyze accuracy.
```
print(classification_report(predictions,label_test)) #checking for accuracy

```




