# -*- coding: utf-8 -*-
"""
Created on Fri May  5 03:15:40 2022

@author: dylan
"""

#%% pckgs/SEC API


from sec_api import QueryApi
from sec_api import ExtractorApi
import pandas as pd
import numpy as np

#%% Queries
queryApi = QueryApi(api_key="My SEC API Key")

query = {
  "query": { "query_string": { 
      "query": "ticker:AMD AND filedAt:{2016-01-01 TO 2020-12-31} AND formType:\"10-K\"" 
    } }
  ,
  "from": "0",
  "size": "10",
  "sort": [{ "filedAt": { "order": "desc" } }]
  
  }
query_2 ={
  "query": { "query_string": { 
      "query": "ticker:INTC AND filedAt:{2016-01-01 TO 2020-12-31} AND formType:\"10-K\"" 
    } }
  ,
  "from": "0",
  "size": "10",
  "sort": [{ "filedAt": { "order": "desc" } }]
  }
query_3 ={
  "query": { "query_string": { 
      "query": "ticker:NVDA AND filedAt:{2016-01-01 TO 2020-12-31} AND formType:\"10-K\"" 
    } }
  ,
  "from": "0",
  "size": "10",
  "sort": [{ "filedAt": { "order": "desc" } }]
  }
query_4 ={
  "query": { "query_string": { 
      "query": "ticker:ADI AND filedAt:{2016-01-01 TO 2020-12-31} AND formType:\"10-K\"" 
    } }
  ,
  "from": "0",
  "size": "10",
  "sort": [{ "filedAt": { "order": "desc" } }]
  }
query_5 ={
  "query": { "query_string": { 
      "query": "ticker:IBM AND filedAt:{2016-01-01 TO 2020-12-31} AND formType:\"10-K\"" 
    } }
  ,
  "from": "0",
  "size": "10",
  "sort": [{ "filedAt": { "order": "desc" } }]
  }
#iShares Expanded Tech Sector ETF (IGM)
query_6 ={
    "query": { "query_string": { 
        "query": "ticker:IGM AND filedAt:{2020-01-01 TO 2020-12-31} AND formType:\"N-CSR\"" 
      } }
    ,
    "from": "0",
    "size": "10",
    "sort": [{ "filedAt": { "order": "desc" } }]
    }
#The Next Generation Quantum Computing ETF (QTUM)
query_7 ={
    "query": { "query_string": { 
        "query": "ticker:QTUM AND filedAt:{2020-01-01 TO 2020-12-31} AND formType:\"N-CSR\"" 
      } }
    ,
    "from": "0",
    "size": "10",
    "sort": [{ "filedAt": { "order": "desc" } }]
    }



queries = pd.read_csv(r"C:\Users\dylan\OneDrive\Desktop\School\Modeling\HW3\NLP_Practice_Queries.CSV")

filings_AMD = queryApi.get_filings(query)
df_AMD = pd.json_normalize(filings_AMD['filings']) 

filings_INTC = queryApi.get_filings(query_2)
df_INTC = pd.json_normalize(filings_INTC['filings'])
filings_NVDA = queryApi.get_filings(query_3)
df_NVDA = pd.json_normalize(filings_NVDA['filings'])
df_NVDA = df_NVDA.drop([1,5])
df_NVDA = df_NVDA.reset_index()
df_NVDA = df_NVDA.drop(columns=('index'))

filings_ADI = queryApi.get_filings(query_4)
df_ADI = pd.json_normalize(filings_ADI['filings'])
filings_IBM = queryApi.get_filings(query_5)
df_IBM = pd.json_normalize(filings_IBM['filings'])


filing_url_AMD = df_AMD['linkToFilingDetails'].to_frame()
filing_url_INTC = df_INTC['linkToFilingDetails'].to_frame()
filing_url_NVDA = df_NVDA['linkToFilingDetails'].to_frame()
filing_url_ADI = df_ADI['linkToFilingDetails'].to_frame()
filing_url_IBM = df_IBM['linkToFilingDetails'].to_frame()


all_dfs = [filing_url_AMD, filing_url_INTC, filing_url_NVDA, filing_url_ADI, filing_url_IBM]


df_all = pd.concat(all_dfs).reset_index(drop=True)

#%% Finbert/Tokenizer

final_10k = []

for url in df_all.index:
     extractorApi = ExtractorApi("My SEC API Key")
     section_10k_item7 = extractorApi.get_section(df_all['linkToFilingDetails'][url],"7",'text')
     final_10k.append(section_10k_item7)

def txt2sentence(txt):
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(txt, language= "english")
    df=pd.DataFrame(sentences)
    return df  

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')  
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

#%% AMD 

sentiment = 0  
sentiment_df = pd.DataFrame()

for index in range(len(final_10k[0:4])):
    df_sentence = txt2sentence(final_10k[index])
    results = []
    for index, row in df_sentence.iterrows():
        try:
            results.append(nlp(row[0]))
        except:
            print('Could not read a line')   
    
    for i in results:
        if i[0]['label'] == 'neutral':
            sentiment = sentiment+0
        if i[0]['label']== 'negative':
            sentiment = sentiment-1
        if i[0]['label']== 'positive':
            sentiment = sentiment+1
    
AMD = [{'Ticker': 'AMD','Year':'2016-2020','Sentiment score':sentiment}]
sentiment_df= sentiment_df.append(AMD)
print(AMD)
results = results.clear()
#%% INTC
sentiment = 0  

for index in range(len(final_10k[5:9])):
    df_sentence = txt2sentence(final_10k[index+5])
    results = []
    for index, row in df_sentence.iterrows():
        try:
            results.append(nlp(row[0]))
        except:
            print('Could not read a line') 

    for i in results:
        if i[0]['label'] == 'neutral':
            sentiment = sentiment+0
        if i[0]['label']== 'negative':
            sentiment = sentiment-1
        if i[0]['label']== 'positive':
            sentiment = sentiment+1
    
INTC = [{'Ticker': 'INTC','Year':'2016-2020','Sentiment score': sentiment}]
sentiment_df= sentiment_df.append(INTC)
print(INTC)
results = results.clear()
#%% NVDA
sentiment = 0

for index in range(len(final_10k[10:14])):
    df_sentence = txt2sentence(final_10k[index+10])
    results = []
    for index, row in df_sentence.iterrows():
        try:
            results.append(nlp(row[0]))
        except:
            print('Could not read a line')
        
      
    for i in results:
        if i[0]['label'] == 'neutral':
            sentiment = sentiment+0
        if i[0]['label']== 'negative':
            sentiment = sentiment-1
        if i[0]['label']== 'positive':
            sentiment = sentiment+1
    
NVDA = [{'Ticker': 'NVDA','Year':'2016-2020','Sentiment score': sentiment}]
sentiment_df= sentiment_df.append(NVDA)
print(NVDA)
results = results.clear()
#%% ADI
sentiment = 0

for index in range(len(final_10k[15:19])):
    df_sentence = txt2sentence(final_10k[index+15])
    results = []
    for index, row in df_sentence.iterrows():
        try:
            results.append(nlp(row[0]))
        except:
            print('Could not read a line')

    for i in results:
        if i[0]['label'] == 'neutral':
            sentiment = sentiment+0
        if i[0]['label']== 'negative':
            sentiment = sentiment-1
        if i[0]['label']== 'positive':
            sentiment = sentiment+1
    
ADI = [{'Ticker': 'ADI','Year':'2016-2020','Sentiment score': sentiment}]
sentiment_df= sentiment_df.append(ADI)
print(ADI)
results = results.clear()
#%% IBM

sentiment = 0

for index in range(len(final_10k[20:25])):
    df_sentence = txt2sentence(final_10k[index+20])
    results = []
    for index, row in df_sentence.iterrows():
        try:
            results.append(nlp(row[0]))
        except:
            print('Could not read a line')

    for i in results:
        if i[0]['label'] == 'neutral':
            sentiment = sentiment+0
        if i[0]['label'] == 'negative':
            sentiment = sentiment-1
        if i[0]['label'] == 'positive':
            sentiment = sentiment+1
    
IBM = [{'Ticker': 'IBM','Year':'2016-2020','Sentiment score': sentiment}]
sentiment_df= sentiment_df.append(IBM)
print(IBM)
results = results.clear()

#%% price2ret & datareaders

ticker = ['AMD','INTC','NVDA','ADI','IBM']

def price2ret(prices, retType='simple'):
    if retType == 'simple':
        ret=(prices/prices.shift(1))-1
    else:
        ret=np.log(prices/prices.shift(1))
    return ret

import pandas_datareader as web 

AMD_Price= web.DataReader('AMD', 'yahoo', start='2015-01-01', end='2022-04-13')
AMD_Price['Returns']= price2ret(AMD_Price[['Adj Close']])

INTC_Price= web.DataReader('INTC', 'yahoo', start='2015-01-01', end='2022-04-13')
INTC_Price['Returns']= price2ret(INTC_Price[['Adj Close']])

NVDA_Price= web.DataReader('NVDA', 'yahoo', start='2015-01-01', end='2022-04-13')
NVDA_Price['Returns']= price2ret(NVDA_Price[['Adj Close']])

ADI_Price= web.DataReader('ADI', 'yahoo', start='2015-01-01', end='2022-04-13')
ADI_Price['Returns']= price2ret(ADI_Price[['Adj Close']])

IBM_Price= web.DataReader('IBM', 'yahoo', start='2015-01-01', end='2022-04-13')
IBM_Price['Returns']= price2ret(IBM_Price[['Adj Close']]) 
    


#%% Plots  ***Needs Revision***

import matplotlib.pyplot as plt
from datetime import datetime

plt.figure()
plt.plot(AMD_Price['Adj Close'], color='Black',)
plt.xlabel('Date')
plt.ylabel('Price')
plt.axvline(x=[datetime(2021,10,29)],color='Blue')
plt.axvline(x=[datetime(2020,10,30)],color='Blue')
plt.axvline(x=[datetime(2019,10,31)],color='Blue')
plt.axvline(x=[datetime(2018,11,5)],color='Blue')
plt.axvline(x=[datetime(2017,11,3)],color='Blue')
plt.title('AMD Price')

plt.figure()
plt.plot(AMD_Price['Returns'], color='Black',)
plt.xlabel('Date')
plt.ylabel('% Returns')
plt.axvline(x=[datetime(2021,10,29)],color='Blue')
plt.axvline(x=[datetime(2020,10,30)],color='Blue')
plt.axvline(x=[datetime(2019,10,31)],color='Blue')
plt.axvline(x=[datetime(2018,11,5)],color='Blue')
plt.axvline(x=[datetime(2017,11,3)],color='Blue')
plt.title('AMD Returns')


plt.figure()
plt.plot(INTC_Price['Adj Close'], color='Black',)
plt.xlabel('Date')
plt.ylabel('Price')
plt.axvline(x=[datetime(2021,2,26)],color='Blue')
plt.axvline(x=[datetime(2020,2,25)],color='Blue')
plt.axvline(x=[datetime(2019,2,27)],color='Blue')
plt.axvline(x=[datetime(2018,2,28)],color='Blue')
plt.axvline(x=[datetime(2017,3,14)],color='Blue')
plt.title('INTC Price')

plt.figure()
plt.plot(INTC_Price['Returns'], color='Black',)
plt.xlabel('Date')
plt.ylabel('% Returns')
plt.axvline(x=[datetime(2021,2,26)],color='Blue')
plt.axvline(x=[datetime(2020,2,25)],color='Blue')
plt.axvline(x=[datetime(2019,2,27)],color='Blue')
plt.axvline(x=[datetime(2018,2,28)],color='Blue')
plt.axvline(x=[datetime(2017,3,14)],color='Blue')
plt.title('INTC Returns')


plt.figure()
plt.plot(NVDA_Price['Adj Close'], color='Black',)
plt.xlabel('Date')
plt.ylabel('Price')
plt.axvline(x=[datetime(2021,2,3)],color='Blue')
plt.axvline(x=[datetime(2020,2,4)],color='Blue')
plt.axvline(x=[datetime(2019,2,5)],color='Blue')
plt.axvline(x=[datetime(2018,2,6)],color='Blue')
plt.title('NVDA Price')

plt.figure()
plt.plot(NVDA_Price['Returns'], color='Black',)
plt.xlabel('Date')
plt.ylabel('% Returns')
plt.axvline(x=[datetime(2021,2,3)],color='Blue')
plt.axvline(x=[datetime(2020,2,4)],color='Blue')
plt.axvline(x=[datetime(2019,2,5)],color='Blue')
plt.axvline(x=[datetime(2018,2,6)],color='Blue')
plt.title('NVDA Returns')


plt.figure()
plt.plot(ADI_Price['Adj Close'], color='Black',)
plt.xlabel('Date')
plt.ylabel('Price')
plt.axvline(x=[datetime(2021,1,15)],color='Blue')
plt.axvline(x=[datetime(2020,1,14)],color='Blue')
plt.axvline(x=[datetime(2019,1,15)],color='Blue')
plt.axvline(x=[datetime(2018,1,12)],color='Blue')
plt.axvline(x=[datetime(2017,1,13)],color='Blue')
plt.title('ADI Price')

plt.figure()
plt.plot(ADI_Price['Returns'], color='Black',)
plt.xlabel('Date')
plt.ylabel('% Returns')
plt.axvline(x=[datetime(2021,1,15)],color='Blue')
plt.axvline(x=[datetime(2020,1,14)],color='Blue')
plt.axvline(x=[datetime(2019,1,15)],color='Blue')
plt.axvline(x=[datetime(2018,1,12)],color='Blue')
plt.axvline(x=[datetime(2017,1,13)],color='Blue')
plt.title('ADI Returns')


plt.figure()
plt.plot(IBM_Price['Adj Close'], color='Black',)
plt.xlabel('Date')
plt.ylabel('Price')
plt.axvline(x=[datetime(2021,2,26)],color='Blue')
plt.axvline(x=[datetime(2020,2,20)],color='Blue')
plt.axvline(x=[datetime(2019,2,21)],color='Blue')
plt.axvline(x=[datetime(2018,2,28)],color='Blue')
plt.axvline(x=[datetime(2017,3,1)],color='Blue')
plt.title('IBM Price')

plt.figure()
plt.plot(IBM_Price['Returns'], color='Black',)
plt.xlabel('Date')
plt.ylabel('% Returns')
plt.axvline(x=[datetime(2021,2,26)],color='Blue')
plt.axvline(x=[datetime(2020,2,20)],color='Blue')
plt.axvline(x=[datetime(2019,2,21)],color='Blue')
plt.axvline(x=[datetime(2018,2,28)],color='Blue')
plt.axvline(x=[datetime(2017,3,1)],color='Blue')
plt.title('IBM Returns')

print(sentiment_df)

