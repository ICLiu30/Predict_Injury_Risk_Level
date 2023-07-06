import streamlit as st
import pickle
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu

from pathlib import Path

dir = Path(__file__).resolve().parent  # Get the directory of the script file

path_to_model = dir / 'model' / 'rf_clf.pkl'
path_to_player = dir / 'data' / 'PlayerStats.csv'

with open(path_to_model, 'rb') as file:
    rf_clf = pickle.load(file)

df_player = pd.read_csv(path_to_player)

def format_player_data(player):
    return player.applymap(lambda x: '{:.2f}'.format(x) if isinstance(x, float) else x)

# Find Player function
def find_player(name, season):
    player = df_player[(df_player['SEARCH'] == name) & (df_player['SEASON'] == season)]
    return player

# Prediction function
def predict_risk_level(data):
    probability = rf_clf.predict_proba(data)[0][1]
    if probability > HIGH_RISK_THRESHOLD:
        return "High Risk"
    elif HIGH_RISK_THRESHOLD >= probability > MODERATE_RISK_THRESHOLD:
        return "Increased Risk"
    elif MODERATE_RISK_THRESHOLD >= probability > LOW_RISK_THRESHOLD:
        return "Moderate Risk"
    else:
        return "Low Risk"

df_player = df_player.drop(['INJURY', 'INJURED_TYPE'], axis=1)
df_display = format_player_data(df_player.head(3))

df_player["SEARCH"] = df_player["PLAYER_NAME"].str.lower()
df_player.insert(2, "SEARCH", df_player.pop("SEARCH"))

# Define the thresholds as constants
LOW_RISK_THRESHOLD = 0.25
MODERATE_RISK_THRESHOLD = 0.5
HIGH_RISK_THRESHOLD = 0.75

    
def show_predict_page():
    st.write("## Predicting Cumulative Injury Risk Level")
    st.write("Welcome to our NBA Player Cumulative Injury Risk Predictor. This tool utilizes advanced statistics and tracking data to estimate the overall cumulative injury risk for NBA players.")

    st.write("Our model provides a broad understanding of a player's risk of sustaining cumulative injuries rather than specifying the risk to individual body parts. Therefore, if you intend to use these"
             " predictions as a basis for preventive care or targeted training, you might want to first focus on common areas of cumulative injury in basketball players. These typically include the lower back, ankles, and knees.")
    st.write("**Please remember, while our model serves as a valuable tool in evaluating potential injury risks, it should be used as a supportive aid for decision-making, not a substitute for professional medical advice or treatment.**")

    st.write("For a detailed look at our methodology, including exploratory data analysis and a comprehensive report, please select **EDA** from the sidebar. In the meantime, feel free to explore the risk predictions for different players."
             " Our model categorizes injury risk into one of four levels: Low Risk, Moderate Risk, Increased Risk, and High Risk.")

    st.markdown("""
    - Low Risk: The model suggests a low likelihood of injury for this player based on their tracking data. Historically, about **10%** of players categorized as low risk have experienced injuries.
    - Moderate Risk: The player's data indicates a moderate risk of injury according to our model. In the past, approximately **25%** of players within this risk category experienced injuries.
    - Increased Risk: The player's data suggests an increased risk of injury, according to the model. In the past, about **50%** of players in this risk level have had injuries.
    - High Risk: The player's tracking data suggests a high risk of injury as per our model. Historically, about **90%** of players categorized as high risk have sustained injuries.
    """)

    st.write("It's important to understand that these \"risk\" levels are estimates of injury probability derived from our predictive model. The provided precision percentages reflect the historical"
             " accuracy of the model's predictions within each category. The actual risk of injury for a specific player may vary as these are statistical predictions and may not account for individual specificities.")

    mode = option_menu(
        menu_title=None,
        options=["Player", "Data"],
        default_index=0,
        orientation="horizontal"
    )
    if mode == 'Player':
        st.write("Our database includes data from players who have participated in a minimum of 20 games during a season. The available seasons range from 2013-14 to 2022-23. To explore the injury risk of a player in our database, simply input their name and the corresponding season.")
        st.write("Alternatively, if you have your own dataset or are interested in assessing the injury risk of a player not included in our database, you can select the **Data** option and input any player's statistics to generate an injury risk prediction.")
        col1, col2 = st.columns(2)
        name = col1.text_input("Player Name (For example: Stephen Curry)")
        season = col2.text_input("Season (For example: 22-23)")

        predict_button = st.button("Predict")
        name = name.lower()

        if predict_button:
            try:
                player = find_player(name, season)
                if player.empty:
                    raise ValueError("Apologies, but this player does not exist in our database for the entered season or the season does not exist.")             
            except ValueError as ve:
                st.write(ve)
            else:
                try:
                    # Show a loading message
                    st.table(format_player_data(player.drop("SEARCH", axis=1)))
                    player = player.drop(['PLAYER_NAME', "SEARCH", 'SEASON'], axis=1)
                    with st.spinner('Predicting...'):
                        # Call the prediction function
                        status = predict_risk_level(player)
                    # Show the result
                    st.success(f"Based on our model's analysis of the entered statistics, this player falls into the {status} category for potential injuries.")
                except Exception as e:
                    st.write(f"An error occurred during prediction: {e}")


    elif mode == 'Data':

        st.markdown("To evaluate a player's risk level based on your own data or to explore the risk level of a player or season not included in our database,"
                    " please input the following statistics. You can generally find this information on the [NBA Statistics](https://www.nba.com/stats/players/advanced)")
        st.write("(Note: We recommend entering players who have participated in at least 20 games, as our model is based on on-court data. Players with a small number of game participations may not be suitable for our prediction model.)")
        st.markdown("""
        **General Section:** MIN (Minutes Played), POSS (Possessions), USG% (Usage Percentage), FGA (Field Goal Attempts), PACE\n

        **Tracking Section:** FRONT CT TOUCHES (Front Court Touches), PAINT TOUCHES, AVG DRIB PER TOUCH (Average Dribbles per Touch),  DIST MILES, AVG SPEED.\n
        """)

        st.write("Here are three examples for reference:")
        st.table(df_display.head(3))
        
        col1, col2 = st.columns(2)
        dist_miles = col1.number_input("Miles Traveled per Game (DIST_MILES)", step=0.01)
        pace = col1.number_input("Pace per Game (PACE)", step=0.01)
        poss = col1.number_input("Possessions per Game (POSS)", step=1)
        front_ct_touches = col1.number_input("Frontcourt Touches per Game (FRONT CT TOUCHES)",step=0.1)
        min = col1.number_input("Minutes Played per Game (MIN)", step=0.1)

        fga_pg = col2.number_input("Field Goal Attempts per Game (FGA)", step=0.1)
        usg_pct = col2.number_input("Usage Percentage per Game (USG PCT)", step=0.01)
        paint_touches = col2.number_input("Paint Touches per Game (PAINT TOUCHES)", step=0.01)
        avg_speed = col2.number_input("Average Speed (AVG_SPEED)", step=0.01)
        avg_drib_per_touch = col2.number_input("Average Dribbles per Touch (AVG_DRIB_PER_TOUCH)", step=0.01)


        predict_button = st.button("Predict")

        if predict_button:
            if usg_pct > 1:
                usg_pct = usg_pct/100
            
            player_manual = pd.DataFrame({
                'DIST_MILES': [dist_miles],
                'PACE': [pace],
                'POSS': [poss],
                'FRONT_CT_TOUCHES': [front_ct_touches],
                'MIN': [min],
                'FGA_PG': [fga_pg],
                'USG_PCT': [usg_pct],
                'PAINT_TOUCHES': [paint_touches],
                'AVG_SPEED': [avg_speed],
                'AVG_DRIB_PER_TOUCH': [avg_drib_per_touch]
            })

            # Show a loading message
            with st.spinner('Predicting...'):
                    # Call the prediction function
                    status = predict_risk_level(player_manual)

            # Show the result
            st.success(f"Based on our model's analysis of the entered statistics, this player falls into the {status} category for potential injuries.")
