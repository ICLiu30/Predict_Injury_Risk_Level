import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from pathlib import Path

dir = Path(__file__).resolve().parent  # Get the directory of the script file

path_to_shap = dir / 'model' / 'shap_values.pkl'
path_to_player = dir / 'data' / 'PlayerStats.csv'
path_to_train = dir / 'data' / 'X_train.csv'

with open(path_to_shap, 'rb') as f:
    shap_values = pickle.load(f)

df_player = pd.read_csv(path_to_player)
X_train = pd.read_csv(path_to_train)

selected_features = ['DIST_MILES', 'PACE', 'POSS', 'FRONT_CT_TOUCHES', 'MIN', 'FGA_PG',
                     'USG_PCT', 'PAINT_TOUCHES', 'AVG_SPEED', 'AVG_DRIB_PER_TOUCH']

importances = np.array([0.1247882 , 0.13674124, 0.1333653 , 0.09538118, 0.10400137,
                        0.08574381, 0.08472132, 0.07205066, 0.07965443, 0.08355248])

def show_eda_page():
        st.write("## Exploratory Data Analysis")
        st.write("### Distribution of Cumulative Injuries: Key Areas of Focus")
        st.write('In this section, we provide an in-depth analysis of the distribution of injuries among players. We identify the most common types of injuries and depict their relative frequencies in a visual format. The aim is to shed light on the critical areas that demand immediate attention to improve players\' health and reduce game downtime.')
        proportions_player_injury = df_player['INJURED_TYPE'].value_counts() / len(df_player)
        labels = ['No Injury', 'Sprained Ankle', 'Sore Knee', 'Sore Ankle', 'Sore Lower Back', 'Knee Injury']
        labels = [labels[5], labels[1], labels[3], labels[2], labels[4]]
        sizes = proportions_player_injury.values.tolist()
        sizes = [sizes[5], sizes[1], sizes[3], sizes[2], sizes[4]]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
        fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, textinfo='label+percent', hole=0.4, hoverinfo='value', 
                                    marker=dict(colors=colors, line=dict(color='#000000', width=2)))])
        fig.update_layout(
        title={'text': 'Player Injury Proportions', 'font': {'size': 22}},  # add your title here and set the font size
        legend=dict(orientation='h', x=0.5, y=-0.1, xanchor='center', yanchor='top')
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('---')
        st.write("### Feature Importance and Distributions")
        st.write("In this section, we delve into the significant factors influencing player injuries. Utilizing the feature importance derived from our predictive model, we rank these factors based on their contribution to the model's predictive power. To enhance interactivity and facilitate a deeper exploration, we provide a select box that allows you to choose which feature's distribution you wish to examine.")
        st.write("Kernel Density Estimate (KDE) plots are used for visualizing the distribution of each feature. KDE plots represent the data's probability density function, letting us understand where the data points (in this case, players) are most densely packed, as represented by the peaks on the plot.")
        st.write("For instance, consider the KDE plot for the DIST MILES feature. It highlights a higher density of injured players at larger DIST MILES values, while lower DIST MILES values show a higher density of non-injured players. This trend underscores the likely correlation between the distance a player travels during a game (DIST MILES) and their likelihood of sustaining an injury. The select box allows for easy switching between features, providing you with a flexible and interactive tool for examining different aspects of the data.")
        
        selected_display = [feat.replace('_', ' ') for feat in selected_features]

        importance_df = pd.DataFrame({'features': selected_display, 'importance': importances})
        importance_df = importance_df.sort_values(by='importance', ascending=False)

        fig2 = go.Figure(go.Bar(
        y=importance_df['features'],
        x=importance_df['importance'],
        orientation='h',
        marker=dict(color='skyblue'),
        ))
        fig2.update_layout(
            title={'text': 'Feature Importance', 'font': {'size': 22}},  # add your title here and set the font size
            xaxis_title='Importance',
            yaxis_title='Features',
            width=500,
            height=550,
            xaxis=dict(
                title_font=dict(size=20),
                tickfont=dict(size=16),
            ),
            yaxis=dict(
                autorange='reversed',
                title_font=dict(size=20),
                tickfont=dict(size=16),
            ),
        )
        st.plotly_chart(fig2)

        selected_feature_display_kde = st.selectbox("Select feature to view KDE", selected_display, key='kde')

        selected_feature_kde = selected_feature_display_kde.replace(' ', '_')
        st.write("")
        st.markdown(f"#### KDE Plot of {selected_feature_display_kde}")

        df_filtered_0 = df_player[df_player['INJURY'] == 0]
        df_filtered_1 = df_player[df_player['INJURY'] == 1]

        sns.set_palette(['royalblue', 'darkturquoise'])

        fig3, ax = plt.subplots(figsize=(4, 3), dpi=200) 
        sns.kdeplot(data=df_filtered_0, x=selected_feature_kde, fill=True, label='Not Injured', common_norm=False)
        sns.kdeplot(data=df_filtered_1, x=selected_feature_kde, fill=True, label='Injured', common_norm=False)

        ax.tick_params(axis='x', labelsize=4.5)

        ax.set_xlabel(f'{selected_feature_display_kde}', fontsize=6)
        ax.set_ylabel('Density', fontsize=6)

        ax.yaxis.set_ticks([])

        ax.legend(fontsize=6)

        st.pyplot(fig3)

        st.markdown('---')
        st.write("### Understanding Feature Interactions through SHAP Values")
        st.write("The importance of a feature and its distribution (as visualized by KDE plots) can sometimes be deceiving. For instance, even though 'PACE' is a high-importance feature, the KDE plot shows significant overlap between injured and non-injured player distributions, making it difficult to discern its direct influence on injury risk. This is where SHAP (SHapley Additive exPlanations) values come in.")
        st.write("SHAP values, unlike KDE plots, take feature interactions into account. A common feature interaction might be between 'DIST MILES' and 'PACE' - where high distances covered at high pace might lead to an increased risk of injury. This interactive effect might not be observable in individual KDE plots but is captured by SHAP values.")
        st.write("In the SHAP value plot, the red line represents a SHAP value of zero. Points above this line indicate a higher impact on predicting player injuries, whereas points below suggest less influence. For instance, for 'DIST MILES', we can see an elevated concentration of points above the red line for values higher than 2, indicating a higher injury risk.")
        st.write("However, some features might not exhibit clear patterns in the SHAP plot due to complex interactions or inherent randomness in the data. These complexities underline the importance of understanding model predictions through SHAP values.")
        st.write("With our prediction model and the insights from the SHAP plot, team managers can make informed decisions about adjusting player behaviors to reduce injury risks, for example, by keeping 'DIST MILES' below certain thresholds to remain in the 'safety zone' (below the red line). This synergy of predictive modeling and interpretability tools empowers teams to better protect their players.")
        
        selected_feature_display_shap = st.selectbox("Select feature to view SHAP", selected_display, key='shap')


        selected_feature_shap = selected_feature_display_shap.replace(' ', '_')

        shap_df = pd.DataFrame(shap_values[1], columns=selected_features)

        scatter = go.Figure()

        scatter.add_trace(go.Scatter(
            x=X_train[selected_feature_shap],
            y=shap_df[selected_feature_shap],
            mode='markers',
            name='Data',
            marker=dict(size=3)  
        ))

   
        scatter.add_shape(
            type='line',
            y0=0, y1=0,
            x0=min(X_train[selected_feature_shap]), x1=max(X_train[selected_feature_shap]),
            line=dict(color='Red', width=2),  
        )

        scatter.update_layout(
            title={'text' :f'SHAP value of {selected_feature_display_shap}',
                   'font':dict(size=22)},
            autosize=False,
            width=650,  
            height=500,  
            plot_bgcolor='white',  
            xaxis=dict(
                title=selected_feature_display_shap,
                title_font=dict(size=18, color='DarkBlue'),  
                gridcolor='lightgrey',  
            ),
            yaxis=dict(
                title='SHAP Value',
                title_font=dict(size=18, color='DarkBlue'), 
                gridcolor='lightgrey',  
            )
        )
        st.plotly_chart(scatter)

        st.markdown('---')
        st.write("### Confidence in Predictions: Modeling Uncertainty")
        st.write('Establishing confidence in our model predictions is key to their usefulness in real-world applications. We present the average injury risk ratio along with a 95% confidence interval for each risk level category. This analysis provides an uncertainty range for our predictions, highlighting the inherent variability in data and modelling processes, and thus supporting robust decision-making.')
        average_ratios = [0.089157, 0.230045, 0.477567, 0.933685]

        conf_intervals = [(0.069154, 0.109160), (0.200591, 0.259500), (0.432563, 0.522572), (0.856938, 1.010433)]

        yerr = [[average - lower for average, (lower, upper) in zip(average_ratios, conf_intervals)],
                [upper - average for average, (lower, upper) in zip(average_ratios, conf_intervals)]]

        error_y=dict(type='data', array=yerr[0] + yerr[1], visible=True)

        fig4 = go.Figure(data=go.Bar(name='Risk Levels', x=['Low Risk', 'Moderate Risk', 'Increased Risk', 'High Risk'],
                                    y=average_ratios, error_y=error_y))

        fig4.update_layout(
            title={
                'text': 'Average Injured Ratio with 95% Confidence Interval for each Risk Level',
                'font': dict(size=22)  
            },
            xaxis=dict(
                tickfont=dict(size=16), 
            ),
            yaxis=dict(
                title='Average Injured Ratio',
                title_font=dict(size=18),  
                tickfont=dict(size=16),  
            )
        )
        st.plotly_chart(fig4)

        st.write("---")
        st.write("### Final Thoughts and Further Exploration")
        st.write("The sections above provide a brief exploration of our feature analysis process, emphasizing the significance of each feature and their interplay in predicting player injuries. However, this is just the tip of the iceberg.")
        st.write("Our complete methodology, including data cleaning, preprocessing, and model training steps, is meticulously documented in our repositories on Kaggle and GitHub. We strongly encourage interested readers to delve deeper into our work for a comprehensive understanding of our process and insights. We believe this will enable you to appreciate the depth and rigor of our approach.")
        st.write("Thank you for your time and interest in our work. We hope it was insightful and sparked your curiosity. We look forward to any feedback, questions, or discussions you might have as we continue to improve and expand upon our analysis.")