
import streamlit as st
import pandas as pd



if st.button('Google get stock accuracies'):
    df = pd.read_pickle("./just_stockGOOG.pkl")
    st.dataframe(df, 200, 100)

if st.button('Microsoft get stock accuracies'):
    df = pd.read_pickle("./just_stockMSFT.pkl")
    st.dataframe(df, 200, 100)

if st.button('Amazon get stock accuracies'):
    df = pd.read_pickle("./just_stockAMZN.pkl")
    st.dataframe(df, 200, 100)

if st.button('Apple get stock accuracies'):
    df = pd.read_pickle("./just_stockAAPL.pkl")
    st.dataframe(df, 200, 100)



if st.button('Google get sentiment accuracies'):
    df = pd.read_pickle("./just_sentimentGOOG.pkl")
    st.dataframe(df, 200, 100)

if st.button('Microsoft get sentiment accuracies'):
    df = pd.read_pickle("./just_sentimentMSFT.pkl")
    st.dataframe(df, 200, 100)

if st.button('Amazon get sentiment accuracies'):
    df = pd.read_pickle("./just_sentimentAMZN.pkl")
    st.dataframe(df, 200, 100)

if st.button('Apple get sentiment accuracies'):
    df = pd.read_pickle("./just_sentimentAAPL.pkl")
    st.dataframe(df, 200, 100)



if st.button('Google get combined accuracies'):
    df = pd.read_pickle("./gatheredGOOG.pkl")
    st.dataframe(df, 200, 100)

if st.button('Microsoft get combined accuracies'):
    df = pd.read_pickle("./gatheredMSFT.pkl")
    st.dataframe(df, 200, 100)

if st.button('Amazon get combined accuracies'):
    df = pd.read_pickle("./gatheredAMZN.pkl")
    st.dataframe(df, 200, 100)

if st.button('Apple get combined accuracies'):
    df = pd.read_pickle("./gatheredAAPL.pkl")
    st.dataframe(df, 200, 100)

