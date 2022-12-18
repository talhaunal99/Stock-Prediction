#!/usr/bin/env python
# coding: utf-8

# In[142]:




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from finta import TA

from datetime import date
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import datetime
import preprocessor as p


# In[17]:


from tcn import compiled_tcn
from wordcloud import WordCloud
import snscrape.modules.twitter as sntwitter


# In[126]:


def getNextDay(y, m, d):
    gDate = datetime.datetime(y, m, d)
    nextday = gDate + datetime.timedelta(days=1)
    nDate = '{:%Y-%m-%d}'.format(nextday)
    return nDate


# In[3]:


def updown(v):
    return 'Up' if v==1 else 'Down'


# In[4]:



def process_date(date):
    datearr = date.split('-')

    newdate = datearr[0]

    newdate += '-'
    if len(datearr[1]) == 1:
        newdate += '0'
    newdate += datearr[1]

    newdate += '-'
    if len(datearr[2]) == 1:
        newdate += '0'
    newdate += datearr[2]

    return newdate


# In[120]:




# In[132]:



def textclean(t):
    '''
    This function cleans the tweets.
    '''
    t = t.lower()  # convert to lowercase
    t = t.replace('\\', "+")
    t = re.sub('\\\\u[0-9A-Fa-f]{4}', '', t)  # remove NON- ASCII characters
    t = re.sub("[0-9]", "", t)  # remove numbers # re.sub("\d+", "", t)
    t = re.sub('[#!"$%&\'()*+,-./:@;<=>?[\\]^_`{|}~]', '', t)  # remove punctuations

    return t


# In[138]:



def remove_emoji(text):
    p.set_options(p.OPT.URL, p.OPT.EMOJI)
    return p.clean(text)


# In[143]:



def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)

    if sentiment_dict['compound'] >= 0.05:
        return 1

    elif sentiment_dict['compound'] <= - 0.05:
        return -1

    else:
        return 0


# In[121]:


def date_prediction(company_name, year, month, day):
    start_date = "{}-{}-{}".format(year, str(month).zfill(2), str(day).zfill(2))
    end_date = getNextDay(year, month, day)

    search_content = "{} + since:{} until:{} -filter:links -filter:replies".format(company_name, start_date, end_date)

    texts_list = []
    sentence_list = []
    pos = 0
    neg = 0
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_content).get_items()):
        texts_list.append(tweet.content)
        if i == 59:
            break
    # print(len(texts_list))
    for j in range(0, min(60, len(texts_list)), 1):
        sentence = texts_list[j]
        if sentence:
            sentence = textclean(sentence)
            if sentence:
                sentence = remove_emoji(sentence)
                sentence = textclean(sentence)
                sentence_list.append(sentence)

                if sentiment_scores(sentence) == 1:
                    pos += 1
                elif sentiment_scores(sentence) == -1:
                    neg += 1

    if pos == 0 and neg == 0:
        pos_rate = 0
        neg_rate = 0
    else:
        pos_rate = pos / (pos + neg)
        neg_rate = neg / (pos + neg)

    s = ' '.join(sentence_list)
    return s, pos_rate, neg_rate


# In[84]:


from sentimentanalysis import sentiment_analysis
def gather_data(company, start_time, end_time):

    # the time of tweets collected, change it when collecting more tweets to train the model
    start_time = "2018-01-01"
    end_time = "2022-05-07"

    sentiment = sentiment_analysis(company)
    
    
    stock = pd.read_csv("data/stock/{}.csv".format(company))
    
    
    
    data = sentiment.merge(stock, how='inner', left_on='date',right_on='Date')
    data.drop(data[(data.pos == 0) & (data.neg == 0)].index, inplace=True)
    data = data.reset_index(drop=True)
    

    # data = stock
    last_idx = data.shape[0] - 1

    hist_data = []
    for i in range(20, last_idx):
        hist_data.append([data['Open'][i], data['High'][i], data['Low'][i], data['Close'][i], data['Adj Close'][i],
                          data['Volume'][i]])

    # Technical Indicators
    stock = data
    ema = TA.EMA(stock, 50).round(decimals=8)
    adx = TA.ADX(stock).round(decimals=8)
    macd = TA.MACD(stock).round(decimals=8)
    rsi = TA.RSI(stock).round(decimals=8)
    sar = TA.SAR(stock).round(decimals=8)
    cci = TA.CCI(stock).round(decimals=8)
    stoch = TA.STOCH(stock).round(decimals=8)
    bop = TA.BOP(stock).round(decimals=8)
    do = TA.DO(stock).round(decimals=8)
    vwap = TA.VWAP(stock).round(decimals=8)

    indicators = []
    for i in range(20, last_idx):
        ind = [ema[i], adx[i], macd['MACD'][i], rsi[i], sar[i], cci[i], stoch[i], bop[i], do['MIDDLE'][i], vwap[i]]
        indicators.append(ind)

    # predicted needed data
    stock_data = np.concatenate((hist_data, indicators), 1)

    # add label
    label = []
    for i in range(20, last_idx):
        if data['Open'][i + 1] >= data['Open'][i]:
            label.append(1)
        else:
            label.append(0)
    label = np.array(label)

    print("Num of class=0: ", sum(i == 0 for i in label))
    print("Num of class=1: ", sum(i == 1 for i in label))

    # Min-Max normalization
    scaler = MinMaxScaler()
    scaler.fit(stock_data)

    stock_data = scaler.transform(stock_data)

    # add sentiment analysis
    pos = np.array(data[20:last_idx]['pos']).reshape((data[20:last_idx]['pos'].shape[0], 1))
    neg = np.array(data[20:last_idx]['neg']).reshape((data[20:last_idx]['neg'].shape[0], 1))
    stock_data = np.concatenate((stock_data, pos, neg), 1)

    print(stock_data)
    return stock_data, label    

gather_data("GOOG","2018-01-01","2022-05-07")  


# In[102]:





# In[ ]:





# In[ ]:





# In[85]:



def knn_model(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, knn


# In[86]:



def logisticReg_model(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(random_state = 0)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    return acc, lr


# In[87]:


def decisionTree_model(X_train, X_test, y_train, y_test):
    dt_entropy = RandomForestClassifier(max_depth=3, random_state=0)
    dt_entropy.fit(X_train, y_train)
    y_pred = dt_entropy.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    return acc, dt_entropy


# In[88]:



def svm_model(X_train, X_test, y_train, y_test):
    svm = sklearn.svm.SVC(kernel='rbf')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    return acc, svm


# In[89]:



def naiveBayes_model(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    return acc, gnb


# In[90]:


def LSTM_model(X_train, X_test, y_train, y_test, epochs):
    model = Sequential()
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)

    y_pred_prob = model.predict(X_test)
    y_pred = []
    for prob in y_pred_prob:
        if prob < 0.5:
            y_pred.append(0)
        else:
            y_pred.append(1)

    acc = accuracy_score(y_test, y_pred)

    return acc, model


# In[91]:




def tcn_model(X_train, X_test, y_train, y_test):
    tcn = compiled_tcn(
        return_sequences=False,
        num_feat=1,
        num_classes=2,
        nb_filters=30,
        kernel_size=2,
        dilations=[2 ** i for i in range(9)],
        nb_stacks=1,
        max_len=X_train.shape[1],
        use_skip_connections=True,
        use_weight_norm=True,
        dropout_rate=0.1)

    tcn.fit(X_train, y_train, epochs=30, verbose=0)

    y_pred_prob = tcn.predict(X_test)
    y_pred = []
    for prob in y_pred_prob:
        if prob[0] > prob[1]:
            y_pred.append(0)
        else:
            y_pred.append(1)
    acc = accuracy_score(y_test, y_pred)

    return acc, tcn


# In[ ]:





# In[198]:





def risk_prediction_ml(X_train, X_test, y_train, y_test, today):
    acc_list = {}
    pred = {}
    acc_list['knn'], knn = knn_model(X_train, X_test, y_train, y_test)
    acc_list['svm'], svm = svm_model(X_train, X_test, y_train, y_test)
    acc_list['dt'], dt = decisionTree_model(X_train, X_test, y_train, y_test)
    acc_list['lr'], lr = logisticReg_model(X_train, X_test, y_train, y_test)
    acc_list['nb'], nb = naiveBayes_model(X_train, X_test, y_train, y_test)

    today = np.array(today).reshape(1, today.shape[0])
    pred['knn'] = knn.predict(today)[0]
    pred['svm'] = svm.predict(today)[0]
    pred['dt'] = dt.predict(today)[0]
    pred['lr'] = lr.predict(today)[0]
    pred['nb'] = nb.predict(today)[0]

    return acc_list, pred


# In[ ]:



def risk_prediction_lstm(X_train, X_test, y_train, y_test, today):

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    y_train = np.array(y_train)
    y_train = y_train.reshape((y_train.shape[0], 1))

    y_test = np.array(y_test)
    y_test = y_test.reshape((y_test.shape[0], 1))

    #     print("shape: ", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    acc_lstm, lstm = LSTM_model(X_train, X_test, y_train, y_test, epochs=50)

    today = np.array(today).reshape(1, today.shape[0], 1)
    pred = 0 if lstm.predict(today)[0] < 0.5 else 1

    return acc_lstm, pred


# In[ ]:


def risk_prediction_tcn(X_train, X_test, y_train, y_test, today):
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    y_train = y_train.reshape((y_train.shape[0], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))

    acc_tcn, tcn = tcn_model(X_train, X_test, y_train, y_test)

    today = np.array(today).reshape(1, today.shape[0], 1)
    prob = tcn.predict(today)[0]
    pred = 0 if prob[0] > prob[1] else 1

    return acc_tcn, pred


# In[145]:


def get_today_info(company):
    today = "2022-05-25"
    y, m, d = int(today.split('-')[0]), int(today.split('-')[1]), int(today.split('-')[2])

    words, pos, neg = date_prediction(company, y, m, d)

    nextday = getNextDay(y, m, d)
    
    data = pd.read_csv("data/stock/{}_Today.csv".format(company))
    data['Date'] = data.index

    last_idx = data.shape[0] - 1

    stock = data
    ema = TA.EMA(stock, 50).round(decimals=8)
    adx = TA.ADX(stock).round(decimals=8)
    macd = TA.MACD(stock).round(decimals=8)
    rsi = TA.RSI(stock).round(decimals=8)
    sar = TA.SAR(stock).round(decimals=8)
    cci = TA.CCI(stock).round(decimals=8)
    stoch = TA.STOCH(stock).round(decimals=8)
    bop = TA.BOP(stock).round(decimals=8)
    do = TA.DO(stock).round(decimals=8)
    vwap = TA.VWAP(stock).round(decimals=8)

    hist_data = []
    for i in range(20, last_idx):
        hist_data.append([data['Open'][i], data['High'][i], data['Low'][i], data['Close'][i], data['Adj Close'][i],
                          data['Volume'][i]])
    indicators = []
    for i in range(20, last_idx + 1):
        ind = [ema[i], adx[i], macd['MACD'][i], rsi[i], sar[i], cci[i], stoch[i], bop[i], do['MIDDLE'][i], vwap[i]]
        indicators.append(ind)

    stock_data = np.concatenate((hist_data, indicators[:-1]), 1)
    today_data = [data['Open'][last_idx], data['High'][last_idx], data['Low'][last_idx], data['Close'][last_idx],
                  data['Adj Close'][last_idx], data['Volume'][last_idx]] + indicators[-1]

    scaler = MinMaxScaler()
    scaler.fit(stock_data)

    stock_data = scaler.transform(stock_data)
    today_data = scaler.transform([today_data])[0]

    today_data = list(today_data) + [pos, neg]

    company_list = ["GOOG", "AMZN", "MSFT", "AAPL"]
    # generate wordcloud
    keywords = company_list + ['tsla', 'stock', 'tickers']
    for c in keywords:
        words = words.replace(c.lower(), "")

        # make wordcloud
    if len(words) != 0:
        wc = WordCloud(
            max_words=200,
            background_color='white',
            width=2000,
            height=1200
        )

        word_cloud = wc.generate(words)
        word_cloud.background_color = 'white'
        plt.imshow(word_cloud)
        plt.axis("off")
        plt.show()

    return today_data
get_today_info("GOOG")


# In[149]:


def risk_prediction_all(company, start_time, end_time):
    stock_data, label = gather_data(company, start_time, end_time)
    today = get_today_info(company)
    today = np.array(today)
    X_train, X_test, y_train, y_test = train_test_split(
        stock_data, label, test_size=0.2, random_state=42)

    acc_list, pred_list = risk_prediction_ml(X_train, X_test, y_train, y_test, today)
    acc_list['lstm'], pred_list['lstm'] = risk_prediction_lstm(X_train, X_test, y_train, y_test, today)
    acc_list['tcn'], pred_list['tcn'] = risk_prediction_tcn(X_train, X_test, y_train, y_test, today)

    model = ['TCN', 'LSTM', 'Random Forest', 'Logistic Regression', 'SVM', 'Naive Bayes', 'KNN']
    acc = [acc_list['tcn'], acc_list['lstm'], acc_list['dt'], acc_list['lr'], acc_list['svm'], acc_list['nb'],
           acc_list['knn']]
    pred = [updown(pred_list['tcn']), updown(pred_list['lstm']), updown(pred_list['dt']), updown(pred_list['lr']),
            updown(pred_list['svm']), updown(pred_list['nb']), updown(pred_list['knn'])]
    df = pd.DataFrame(data={'Model': model, 'Accuracy': acc, 'Up/Down prediction': pred})
    print(df)

    #df.to_csv('data/updown-prediction-{}.csv'.format(company), index=False)
risk_prediction_all("GOOG","2018-01-01","2022-05-07")


# In[152]:


risk_prediction_all("AAPL","2018-01-01","2022-05-07")


# In[153]:


risk_prediction_all("AMZN","2018-01-01","2022-05-07")


# In[155]:


risk_prediction_all("MSFT","2018-01-01","2022-05-07")


# In[ ]:


##################################################################################


# In[225]:


def just_sentiment_data(company, start_time, end_time):

    # the time of tweets collected, change it when collecting more tweets to train the model
    start_time = "2018-01-01"
    end_time = "2022-05-07"

    sentiment = sentiment_analysis(company)
    
    data = sentiment
    data.drop(data[(data.pos == 0) & (data.neg == 0)].index, inplace=True)
    data = data.reset_index(drop=True)
    
    # data = stock

    stockforlabeling = pd.read_csv("data/stock/{}.csv".format(company))
    last_idx = stockforlabeling.shape[0] - 1

    # add label
    label = []
    for i in range(20, last_idx):
        if stockforlabeling['Open'][i + 1] >= stockforlabeling['Open'][i]:
            label.append(1)
        else:
            label.append(0)
    label = np.array(label)

    print("Num of class=0: ", sum(i == 0 for i in label))
    print("Num of class=1: ", sum(i == 1 for i in label))

    d=data.head(1075)
    d1=d.drop(['date'],axis=1)
    return d1,label   

just_sentiment_data("GOOG","2018-01-01","2022-05-07")  


# In[226]:


def get_today_info_just_sentiment(company):
    today = "2022-05-25"
    y, m, d = int(today.split('-')[0]), int(today.split('-')[1]), int(today.split('-')[2])

    words, pos, neg = date_prediction(company, y, m, d)

    nextday = getNextDay(y, m, d)
    

    company_list = ["GOOG", "AMZN", "MSFT", "AAPL"]
    # generate wordcloud
    keywords = company_list + ['tsla', 'stock', 'tickers']
    for c in keywords:
        words = words.replace(c.lower(), "")

        # make wordcloud
    if len(words) != 0:
        wc = WordCloud(
            max_words=200,
            background_color='white',
            width=2000,
            height=1200
        )

        word_cloud = wc.generate(words)
        word_cloud.background_color = 'white'
        plt.imshow(word_cloud)
        plt.axis("off")
        plt.show()

        return([pos,neg])
    
get_today_info_just_sentiment("GOOG")


# In[230]:


def risk_prediction_just_sentiment(company, start_time, end_time):
    
    stock_data, label = just_sentiment_data(company, start_time, end_time)
    today = get_today_info_just_sentiment(company)
    stock_data=stock_data.to_numpy()
    today = np.array(today)
    X_train, X_test, y_train, y_test = train_test_split(
        stock_data, label, test_size=0.2, random_state=42)

    acc_list, pred_list = risk_prediction_ml(X_train, X_test, y_train, y_test, today)
    acc_list['lstm'], pred_list['lstm'] = risk_prediction_lstm(X_train, X_test, y_train, y_test, today)
    acc_list['tcn'], pred_list['tcn'] = risk_prediction_tcn(X_train, X_test, y_train, y_test, today)

    model = ['TCN', 'LSTM', 'Random Forest', 'Logistic Regression', 'SVM', 'Naive Bayes', 'KNN']
    acc = [acc_list['tcn'], acc_list['lstm'], acc_list['dt'], acc_list['lr'], acc_list['svm'], acc_list['nb'],
           acc_list['knn']]
    pred = [updown(pred_list['tcn']), updown(pred_list['lstm']), updown(pred_list['dt']), updown(pred_list['lr']),
            updown(pred_list['svm']), updown(pred_list['nb']), updown(pred_list['knn'])]
    df = pd.DataFrame(data={'Model': model, 'Accuracy': acc, 'Up/Down prediction': pred})
    print(df)

risk_prediction_just_sentiment("GOOG","2018-01-01","2022-05-07")


# In[231]:


risk_prediction_just_sentiment("AAPL","2018-01-01","2022-05-07")


# In[232]:


risk_prediction_just_sentiment("AMZN","2018-01-01","2022-05-07")


# In[233]:


risk_prediction_just_sentiment("MSFT","2018-01-01","2022-05-07")


# In[ ]:


##################################################################################


# In[170]:


def get_today_info_just_stock(company):
    today = "2022-05-25"
    y, m, d = int(today.split('-')[0]), int(today.split('-')[1]), int(today.split('-')[2])


    nextday = getNextDay(y, m, d)
    
    data = pd.read_csv("data/stock/{}_Today.csv".format(company))
    data['Date'] = data.index

    last_idx = data.shape[0] - 1

    stock = data
    ema = TA.EMA(stock, 50).round(decimals=8)
    adx = TA.ADX(stock).round(decimals=8)
    macd = TA.MACD(stock).round(decimals=8)
    rsi = TA.RSI(stock).round(decimals=8)
    sar = TA.SAR(stock).round(decimals=8)
    cci = TA.CCI(stock).round(decimals=8)
    stoch = TA.STOCH(stock).round(decimals=8)
    bop = TA.BOP(stock).round(decimals=8)
    do = TA.DO(stock).round(decimals=8)
    vwap = TA.VWAP(stock).round(decimals=8)

    hist_data = []
    for i in range(20, last_idx):
        hist_data.append([data['Open'][i], data['High'][i], data['Low'][i], data['Close'][i], data['Adj Close'][i],
                          data['Volume'][i]])
    indicators = []
    for i in range(20, last_idx + 1):
        ind = [ema[i], adx[i], macd['MACD'][i], rsi[i], sar[i], cci[i], stoch[i], bop[i], do['MIDDLE'][i], vwap[i]]
        indicators.append(ind)

    stock_data = np.concatenate((hist_data, indicators[:-1]), 1)
    today_data = [data['Open'][last_idx], data['High'][last_idx], data['Low'][last_idx], data['Close'][last_idx],
                  data['Adj Close'][last_idx], data['Volume'][last_idx]] + indicators[-1]

    scaler = MinMaxScaler()
    scaler.fit(stock_data)

    stock_data = scaler.transform(stock_data)
    today_data = scaler.transform([today_data])[0]

    print(today_data.shape)
    return today_data
get_today_info_just_stock("GOOG")


# In[162]:


def just_stock_data(company, start_time, end_time):

    
    data = pd.read_csv("data/stock/{}.csv".format(company))
    

    

    last_idx = data.shape[0] - 1

    hist_data = []
    for i in range(20, last_idx):
        hist_data.append([data['Open'][i], data['High'][i], data['Low'][i], data['Close'][i], data['Adj Close'][i],
                          data['Volume'][i]])

    # Technical Indicators
    stock = data
    ema = TA.EMA(stock, 50).round(decimals=8)
    adx = TA.ADX(stock).round(decimals=8)
    macd = TA.MACD(stock).round(decimals=8)
    rsi = TA.RSI(stock).round(decimals=8)
    sar = TA.SAR(stock).round(decimals=8)
    cci = TA.CCI(stock).round(decimals=8)
    stoch = TA.STOCH(stock).round(decimals=8)
    bop = TA.BOP(stock).round(decimals=8)
    do = TA.DO(stock).round(decimals=8)
    vwap = TA.VWAP(stock).round(decimals=8)

    indicators = []
    for i in range(20, last_idx):
        ind = [ema[i], adx[i], macd['MACD'][i], rsi[i], sar[i], cci[i], stoch[i], bop[i], do['MIDDLE'][i], vwap[i]]
        indicators.append(ind)

    stock_data = np.concatenate((hist_data, indicators), 1)

    # add label
    label = []
    for i in range(20, last_idx):
        if data['Open'][i + 1] >= data['Open'][i]:
            label.append(1)
        else:
            label.append(0)
    label = np.array(label)

    print("Num of class=0: ", sum(i == 0 for i in label))
    print("Num of class=1: ", sum(i == 1 for i in label))

    # Min-Max normalization
    scaler = MinMaxScaler()
    scaler.fit(stock_data)

    stock_data = scaler.transform(stock_data)
    print(stock_data.shape)
    return stock_data, label    

just_stock_data("GOOG","2018-01-01","2022-05-07")  


# In[172]:


def risk_prediction_just_stock(company, start_time, end_time):
        
    stock = pd.read_csv("data/stock/{}.csv".format(company))
    stock_data, label = just_stock_data(company, start_time, end_time)
    today = get_today_info_just_stock(company)
    today = np.array(today)
    X_train, X_test, y_train, y_test = train_test_split(
        stock_data, label, test_size=0.2, random_state=42)

    acc_list, pred_list = risk_prediction_ml(X_train, X_test, y_train, y_test, today)
    acc_list['lstm'], pred_list['lstm'] = risk_prediction_lstm(X_train, X_test, y_train, y_test, today)
    acc_list['tcn'], pred_list['tcn'] = risk_prediction_tcn(X_train, X_test, y_train, y_test, today)

    model = ['TCN', 'LSTM', 'Random Forest', 'Logistic Regression', 'SVM', 'Naive Bayes', 'KNN']
    acc = [acc_list['tcn'], acc_list['lstm'], acc_list['dt'], acc_list['lr'], acc_list['svm'], acc_list['nb'],
           acc_list['knn']]
    pred = [updown(pred_list['tcn']), updown(pred_list['lstm']), updown(pred_list['dt']), updown(pred_list['lr']),
            updown(pred_list['svm']), updown(pred_list['nb']), updown(pred_list['knn'])]
    df = pd.DataFrame(data={'Model': model, 'Accuracy': acc, 'Up/Down prediction': pred})
    print(df)

    #df.to_csv('data/updown-prediction-{}.csv'.format(company), index=False)
risk_prediction_just_stock("GOOG","2018-01-01","2022-05-07")


# In[174]:


risk_prediction_just_stock("AAPL","2018-01-01","2022-05-07")


# In[176]:


risk_prediction_just_stock("AMZN","2018-01-01","2022-05-07")


# In[175]:


risk_prediction_just_stock("MSFT","2018-01-01","2022-05-07")


# In[248]:




