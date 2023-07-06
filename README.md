## NBA Cumulative Injury Prediction

This project leverages NBA player statistics and injury data to predict the likelihood of player injuries. The data, sourced from NBA.com and HashtagBasketball.com, includes player tracking stats and cumulative injury types such as sprains and strains.

Aiming to enhance player health management, the project applies various machine learning techniques including XGBoost, Random Forest, and SVM. The Random Forest model achieved promising results with a ROC_AUC score of 0.75 and an accuracy of 80%, given the limited data.

The application's novelty lies in its ability to define different risk levels associated with player injuries, thereby facilitating more informed decisions beyond the binary 'injured' or 'not injured' classification.

You can explore the [model application](https://predictinjuryrisklevel-33w2e6tvkch.streamlit.app) and if interested in the model training, check the NBA_Injury_Predcition.ipynb in this repository or visit my [Kaggle page](https://www.kaggle.com/code/icliu30/nba-cumulative-injury-prediction).

For the application's code, refer to the `web` directory in this repository.
