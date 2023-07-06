import streamlit as st
from streamlit_option_menu import option_menu
from Predict_page import show_predict_page
from EDA_page import show_eda_page

with st.sidebar:
    selected = option_menu(
        menu_title = "Menu",
        options = ["Predict Model", "EDA"]
    )


if selected == "Predict Model":
    show_predict_page()
elif selected == "EDA":
    show_eda_page()