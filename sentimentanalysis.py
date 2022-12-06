#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[3]:


company_name = ["GOOG", "AMZN", "MSFT", "AAPL"]

def sentiment_analysis(company): #gets company name
    data = pd.read_csv('./data/tweets/cleaned{}.csv'.format(company))
    data = data[data.columns[1:]]
    data['tweets'] = data[data.columns[2:]].apply(lambda x: '. '.join(x.dropna().astype(str)), axis=1)

    pos_list, neg_list = [], []
    sia = SentimentIntensityAnalyzer()
    for i in range(len(data["tweets"])):
        pos, neg = 0, 0
        for text in data["tweets"][i].split('.'):
            res = sia.polarity_scores(text)
            if res['compound'] >= 0.05:
                pos += 1
            elif res['compound'] <= - 0.05:
                neg += 1

        posneg = pos + neg if pos + neg != 0 else 1
        pos_list.append(pos / posneg)
        neg_list.append(neg / posneg)

    df = pd.DataFrame(data={'date': data.date, 'pos': pos_list, 'neg': neg_list})
    return df 


# In[4]:


import nltk
nltk.download('vader_lexicon')

google_rates = sentiment_analysis("GOOG")
google_rates.head(10)


# In[ ]:




