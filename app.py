
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

import risk_prediction as rp


if st.button('Google get stock accuracies'):
    df=rp.risk_prediction_just_stock("GOOG", "2018-01-01", "2022-05-07")
    st.dataframe(df, 200, 100)

if st.button('Microsoft get stock accuracies'):
    df=rp.risk_prediction_just_stock("MSFT", "2018-01-01", "2022-05-07")
    st.dataframe(df, 200, 100)

if st.button('Amazon get stock accuracies'):
    df=rp.risk_prediction_just_stock("AMZN", "2018-01-01", "2022-05-07")
    st.dataframe(df, 200, 100)

if st.button('Apple get stock accuracies'):
    df=rp.risk_prediction_just_stock("AAPL", "2018-01-01", "2022-05-07")
    st.dataframe(df, 200, 100)



if st.button('Google get sentiment accuracies'):
    df=rp.risk_prediction_just_sentiment("GOOG", "2018-01-01", "2022-05-07")
    st.dataframe(df, 200, 100)

if st.button('Microsoft get sentiment accuracies'):
    df=rp.risk_prediction_just_sentiment("MSFT", "2018-01-01", "2022-05-07")
    st.dataframe(df, 200, 100)

if st.button('Amazon get sentiment accuracies'):
    df=rp.risk_prediction_just_sentiment("AMZN", "2018-01-01", "2022-05-07")
    st.dataframe(df, 200, 100)

if st.button('Apple get sentiment accuracies'):
    df=rp.risk_prediction_just_sentiment("AAPL", "2018-01-01", "2022-05-07")
    st.dataframe(df, 200, 100)



if st.button('Google get sentiment accuracies'):
    df=rp.risk_prediction_just_sentiment("GOOG", "2018-01-01", "2022-05-07")
    st.dataframe(df, 200, 100)

if st.button('Microsoft get sentiment accuracies'):
    df=rp.risk_prediction_just_sentiment("MSFT", "2018-01-01", "2022-05-07")
    st.dataframe(df, 200, 100)

if st.button('Amazon get sentiment accuracies'):
    df=rp.risk_prediction_just_sentiment("AMZN", "2018-01-01", "2022-05-07")
    st.dataframe(df, 200, 100)

if st.button('Apple get sentiment accuracies'):
    df=rp.risk_prediction_just_sentiment("AAPL", "2018-01-01", "2022-05-07")
    st.dataframe(df, 200, 100)



if st.button('Google get sentiment accuracies'):
    df=rp.risk_prediction_just_sentiment("GOOG", "2018-01-01", "2022-05-07")
    st.dataframe(df, 200, 100)

if st.button('Microsoft get all accuracies'):
    df=rp.risk_prediction_all("MSFT", "2018-01-01", "2022-05-07")
    st.dataframe(df, 200, 100)

if st.button('Amazon get all accuracies'):
    df=rp.risk_prediction_just_sentiment("AMZN", "2018-01-01", "2022-05-07")
    st.dataframe(df, 200, 100)

if st.button('Apple get all accuracies'):
    df=rp.risk_prediction_just_all("AAPL", "2018-01-01", "2022-05-07")
    st.dataframe(df, 200, 100)