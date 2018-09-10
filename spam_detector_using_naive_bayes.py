#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 16:57:47 2018

@author: priyansu
"""
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
mails=pd.read_csv("spam.csv",encoding="latin-1")
mails.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],axis=1, inplace=True)
mails.rename(columns={"v1":"label", "v2":"messages"}, inplace = True)
mails["label"]=mails["label"].replace({"ham":0, "spam":1})
y=mails.iloc[:,0].values



spam_words=" ".join(list(mails[mails["label"]==1]["messages"]))
spam_wc=WordCloud(width=512, height=512).generate(spam_words)
plt.imshow(spam_wc)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

ham_words=" ".join(list(mails[mails["label"]==0]["messages"]))
ham_wc=WordCloud(width=512, height=512).generate(ham_words)
plt.imshow(ham_wc)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

vectorize_count = CountVectorizer(decode_error='ignore')
x = vectorize_count.fit_transform(mails['messages'])


from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("Accuracy is : ",end=" ")
accuracy=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print(accuracy)

#Bonus: Mails which are actually spam but predicted wrong by our classifier

mails['predictions'] = classifier.predict(x)
sneaky_spam = mails[(mails["predictions"]==0) & (mails["label"]==1)]['messages']
for msg in sneaky_spam:
  print(msg)






