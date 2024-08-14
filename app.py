import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu # for setting up menu bar
import matplotlib.pyplot as plt # for data analysis and visualization
import seaborn as sns
import plotly.express as px
from plotly import graph_objs as go # for creating interactive visualizations
#import geopandas as gpd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
import numpy as np
import pickle
from io import BytesIO


#-----------Web page setting-------------------#
page_title = "AMR Web App"
page_icon = "🦠🧬"
viz_icon = "📊"
stock_icon = "📋"
picker_icon = "👇"
layout = "centered"

#--------------------Page configuration------------------#
st.set_page_config(page_title = page_title, page_icon = page_icon, layout = layout)

#--------------------Web App Design----------------------#

selected = option_menu(
    menu_title = page_title + " " + page_icon,
    options = ['Home', 'Analysis', 'Train Model', 'Make a Forecast', 'Make Prediction', 'About'],
    icons = ["house-fill", "book-half", "gear", "activity", "robot", "envelope-fill"],
    default_index = 0,
    orientation = "horizontal"
)


#Antimicrobial Resistance in Europe Data
euro_link_id = "1aNulNFQQzvoDh75hbtZJ7_QcW4fRl9rs"
euro_link = f'https://drive.google.com/uc?id={euro_link_id}'
euro_df = pd.read_csv(euro_link)
#euro_df.head()

# Omadacycline Gram-Negative Data
negative_link_id = "1jHO-NFMsauUGVx9pfW5RPMvNGc6G0wAT"
negative_link = f'https://drive.google.com/uc?id={negative_link_id}'
gram_neg = pd.read_csv(negative_link)

# Omadacycline Gram-Negative Data
positive_link_id = "1NHR41hfyCN26EmQ7SrLCrhSBJnAVDQo0"
positive_link = f'https://drive.google.com/uc?id={positive_link_id}'
gram_pos = pd.read_csv(positive_link)

# Home page
if selected == "Home":
    st.subheader("Welcome to AMR Web App")
    st.write("Some dummy texts here")


if selected == "Analysis":
    datasets = ["Antimicrobial Resistance in Europe Data", "Gram-Negative Bactirial Surveilance Data", 
                "Gram-Positive Bactirial Surveilance Data"]
    st.subheader("Select oreferred dataset")
    selected_dataset = st.selectbox("Pick a dataset " + picker_icon, datasets)

    if selected_dataset == "Antimicrobial Resistance in Europe Data":

        analysis = ["Descriptive Statistics", "Resistance Trend Analysis", "Comparative Analysis",
                    "Demographic Analysis", "Bacteria Analysis", "Antibiotics Analysis"]
        st.subheader("Select analysis")
        selected_analysis = st.selectbox("Pick analysis type " + picker_icon, analysis)

        if selected_analysis == "Demographic Analysis":
            data = euro_df
            data['Distribution'] = data['Distribution'].str.split(',').str[1].str.split(' ').str[-1]
            age_subset = data[data['Distribution'] == "age"]
            age_count = age_subset['Category'].value_counts()
            #st.subheader("Bar Chart for Categorical Variables")
            #st.write("Bar Chart of Age Group")
            #st.bar_chart(age_subset, x='Category')

            fig = px.bar(age_subset, x='Category', title='Age Group Distribution',
                        labels={'Category': 'Age Group', 'count': 'Frequency'})
            st.plotly_chart(fig)
            st.info("Add summary analysis here!")

            gender_subset = data[data['Distribution'] == "gender"]
            gender_count = gender_subset['Category'].value_counts()

            fig = px.bar(gender_subset, x='Category', title='Gender Distribution',
                        labels={'Category': 'Gender', 'count': 'Frequency'})
            st.plotly_chart(fig)
            st.info("Add summary analysis here!")

            st.write("Top 10 Countries in Europe of Surveilance Data")
            country_count = euro_df['RegionName'].value_counts().head(10)
            st.bar_chart(country_count)
            st.info("Add summary analysis here!")
        
        if selected_analysis == "Descriptive Statistics":
            fig = px.histogram(
                euro_df, 
                x='Value', 
                nbins=30, 
                title='Histogram of Resistance Percentages',
                labels={'Value': 'Resistance Percentage'}, 
                color_discrete_sequence=['blue']
            )

            fig.update_layout(
                xaxis_title='Resistance Percentage',
                yaxis_title='Frequency',
                bargap=0.1,
                height=600
            )

            st.plotly_chart(fig)

            st.info("We can add some summary analysis here.")

            st.subheader("Bar Chart for Categorical Variables")
            st.write("Bar Chart of Bacteria Types")
            bacteria_count = euro_df['Bacteria'].value_counts()
            st.bar_chart(bacteria_count)
            st.info("Add summary analysis here!")

            st.write("Bar Chart of Antibiotic Types")
            antibiotic_count = euro_df['Antibiotic'].value_counts()
            st.bar_chart(antibiotic_count)
            st.info("Add summary analysis here!")

            st.subheader("Pie Chart of Distribution by Category")
            st.write("Pie Chart of Distribution by Category")
            gender_distribution = euro_df['Category'].value_counts()
            fig = px.pie(names=gender_distribution.index, values=gender_distribution.values)
            st.plotly_chart(fig)
            st.info("Add summary analysis here!")
        
        if selected_analysis == "Resistance Trend Analysis":
            st.subheader("Time Series Analysis")
            aggregated_data = euro_df.groupby('Time').agg({'Value': 'mean'}).reset_index()
            fig1 = px.line(
                aggregated_data,
                x='Time', 
                y='Value',  
                title='Resistance Trend Over Time',
                labels={'Time': 'Year', 'Value': 'Average Resistance'},
                markers=True  
            )

            st.plotly_chart(fig1)
            st.info("Add summary analysis here!")

            st.subheader("Geographical Analysis")
            fig2 = px.choropleth(
                euro_df,
                locations='RegionName',
                locationmode='country names',
                color='Value',
                hover_name='RegionName',
                color_continuous_scale=px.colors.sequential.Plasma_r,
                title='Antimicrobial Resistance in Europe',
                scope='europe'
            )

            st.plotly_chart(fig2)
        
            st.info("Add summary analysis here!")

        if selected_analysis == "Comparative Analysis":
            st.subheader("Antibiotic Efficacy Comparison")
            fig1 = px.box(
                euro_df,
                x='Antibiotic',
                y='Value',
                title='Antibiotic Efficacy Comparison',
                labels={'Antibiotic': 'Antibiotic', 'Value': 'Resistance Percentage'},
                category_orders={'Antibiotic': sorted(euro_df['Antibiotic'].unique())}
            )

            fig1.update_layout(
                title={'text': 'Resistance Trends Over Time', 'x': 0.5},
                xaxis_title='Year',
                yaxis_title='Resistance Percentage',
                height=800,  
                width=800   
            )

            st.plotly_chart(fig1)
            st.info("Add summary analysis here!")

            st.subheader("Bacteria-Antibiotic Interaction")
            st.write("Bacteria-Antibiotic Interaction Heatmap")
            interaction_data = euro_df.pivot_table(index='Bacteria', columns='Antibiotic', values='Value', aggfunc='mean')

            fig2 = px.imshow(
                interaction_data,
                color_continuous_scale='Viridis',
                title='Bacteria-Antibiotic Interaction Heatmap',
                labels={'color': 'Resistance Percentage'},
                aspect='auto'
            )

            fig2.update_layout(
                title={'text': 'Bacteria-Antibiotic Interaction Heatmap', 'x': 0.5},
                xaxis_title='Antibiotic',
                yaxis_title='Bacteria',
                height=1200,  
                width=800   
            )
            st.plotly_chart(fig2)
            st.info("Add summary analysis here!")

if selected == "Train Model":
    # Dummy feature encoding
    data_encoded = euro_df
    data_encoded['Distribution'] = data_encoded['Distribution'].str.split(',').str[1].str.split(' ').str[-1]
    data_encoded = pd.get_dummies(euro_df, columns=['Distribution', 'RegionName', 'Bacteria', 'Antibiotic', 'Category'])
    data_encoded = data_encoded.drop(columns = ['Unit', 'RegionCode', 'Unnamed: 0'], axis = 1)
    # Feature and target
    X = data_encoded.drop('Value', axis=1)
    y = (euro_df['Value'] > 50).astype(int)  # Binary classification: 0 for non-resistant, 1 for resistant

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Select model to train
    model_list = ['Random Forest', 'Logistic Regression', 'Support Vector Machine (SVM)',
                  'Gradient Boosting Classifier', 'K-Nearest Neighbors (KNN)', 
                  'Decision Tree Classifier', 'Extreme Gradient Boosting (XGBoost)',
                  'Neural Network (MLPClassifier)', 'CatBoost Classifier', 'LightGBM Classifier']
    
    st.subheader("Select Algorithm")
    model_selected = st.selectbox("Pick an alogorithm to train on " + picker_icon, model_list)
    
    def model_training_and_analysis(model_selected, X_train, y_train, X_test, y_test):
        # Initialize the model based on selection
        if model_selected == "Random Forest":
            model = RandomForestClassifier()
        elif model_selected == "Logistic Regression":
            model = LogisticRegression()
        elif model_selected == "Support Vector Machine (SVM)":
            model = SVC()
        elif model_selected == "Gradient Boosting Classifier":
            model = GradientBoostingClassifier()
        elif model_selected == "K-Nearest Neighbors (KNN)":
            model = KNeighborsClassifier()
        elif model_selected == "Decision Tree Classifier":
            model = DecisionTreeClassifier()
        elif model_selected == "Extreme Gradient Boosting (XGBoost)":
            model = XGBClassifier()
        elif model_selected == "Neural Network (MLPClassifier)":
            model = MLPClassifier()
        elif model_selected == "CatBoost Classifier":
            model = CatBoostClassifier(silent=True)
        elif model_selected == "LightGBM Classifier":
            model = LGBMClassifier()
        
        # Train the model
        model.fit(X_train, y_train)

        # Feature importance (if applicable)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Prepare data for Plotly Express
            importance_df = pd.DataFrame({
                'Feature': [X_train.columns[i] for i in indices],
                'Importance': importances[indices]
            })

            # Clean up feature names
            def clean_feature_name(name):
                parts = name.split('_')
                return parts[-1] if len(parts) > 1 else name
            
            importance_df['Cleaned_Feature'] = importance_df['Feature'].apply(clean_feature_name)

            st.subheader("Feature Importance Analysis")
            # Plot using Plotly Express
            fig = px.bar(importance_df, x='Cleaned_Feature', y='Importance', title='Feature Importance',
                        labels={'Cleaned_Feature': 'Feature Name', 'Importance': 'Importance Value'},
                        height=1000)
            fig.update_xaxes(tickangle=-90)
            st.plotly_chart(fig)

        # Predictions
        st.subheader("Prediction Score")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_percentage = accuracy * 100

        st.markdown(f"Accuracy Score: **{accuracy_percentage:.2f}%**")

        # Confusion Matrix Analysis
        st.subheader("Confusion Matrix Analysis")
        cm = confusion_matrix(y_test, y_pred)

        # Convert confusion matrix to DataFrame for Plotly
        cm_df = pd.DataFrame(cm, index=['Resistant', 'Suceptible'], columns=['Predicted Resistant', 'Suceptible'])

        # Plot confusion matrix using Plotly Express
        fig1 = px.imshow(cm_df, text_auto=True, color_continuous_scale='Viridis',
                        labels={'x': 'Predicted Label', 'y': 'Actual Label', 'color': 'Count'},
                        title='Confusion Matrix')
        fig1.update_xaxes(side="bottom")  # Move x-axis labels to bottom
        st.plotly_chart(fig1)

        # Provide download button to download the trained model
        st.subheader("Download Trained Model")
        pickle_buffer = BytesIO()
        pickle.dump(model, pickle_buffer)
        pickle_buffer.seek(0)

        st.download_button(
            label="Download Model",
            data=pickle_buffer,
            file_name=f"{model_selected.replace(' ', '_')}_model.pkl",
            mime="application/octet-stream"
        )


    model_training_and_analysis(model_selected, X_train, y_train, X_test, y_test)

if selected == "Make a Forecast":
    data = euro_df
    data = data.drop(['Unnamed: 0', 'Unit', 'RegionCode'], axis=1)
    data['Distribution'] = data['Distribution'].str.split(',').str[1].str.split(' ').str[-1]
    st.subheader("Select Bacteria and a corresponding Antibiotic")
    bacteria_selected = st.selectbox("Pick a bacteria " + picker_icon, data['Bacteria'].unique())
    anti_selected = st.selectbox("Pick a antibiotic " + picker_icon, data['Antibiotic'].unique())

    filtered_data = data[(data['Bacteria'] == bacteria_selected) & (data['Antibiotic'] == anti_selected)]
    filtered_data = filtered_data.drop(columns=['Bacteria', 'Antibiotic'])

    if filtered_data.empty:
        st.info(f"**{bacteria_selected}** does not apply to **{anti_selected}**")
    else:
        # Encode categorical variables
        #le = LabelEncoder()
        #filtered_data['RegionName'] = le.fit_transform(filtered_data['RegionName'])
        #filtered_data['Category'] = le.fit_transform(filtered_data['Category'])
        #filtered_data['Distribution'] = le.fit_transform(filtered_data['Distribution'])
        #filtered_data['Year'] = pd.to_datetime(filtered_data['Time']).dt.year
        
        # Prepare features and target
        filtered_data['ds'] = pd.to_datetime(filtered_data['Time'], format='%Y') #.dt.year
        filtered_data['y'] = filtered_data['Value']

        # Encode categorical variables (RegionName and Category)
        filtered_data['RegionName_encoded'] = pd.factorize(filtered_data['RegionName'])[0]
        filtered_data['Category_encoded'] = pd.factorize(filtered_data['Category'])[0]
        filtered_data['Distribution_encoded'] = pd.factorize(filtered_data['Distribution'])[0]

        # Initialize the Prophet model
        model = Prophet()

        # Add regressors
        model.add_regressor('RegionName_encoded')
        model.add_regressor('Category_encoded')
        model.add_regressor('Distribution_encoded')

        # Fit the model
        model.fit(filtered_data[['ds', 'y', 'RegionName_encoded', 'Category_encoded', 'Distribution_encoded']])

        # Make future dataframe for the next 5 years
        future = model.make_future_dataframe(periods=5, freq='YE')

        # Include the same values of RegionName_encoded and Category_encoded for future predictions
        future['RegionName_encoded'] = filtered_data['RegionName_encoded'].iloc[-1]
        future['Category_encoded'] = filtered_data['Category_encoded'].iloc[-1]
        future['Distribution_encoded'] = filtered_data['Distribution_encoded'].iloc[-1]


        # Predict future values
        forecast = model.predict(future)

        # Plot the results
        fig = plt.figure(figsize=(10, 6))  # Adjust the figsize tuple as needed
        ax = fig.add_subplot(111)
        fig = model.plot(forecast, ax=ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Resistance Value")
        ax.set_title(f"Forecast for {bacteria_selected} with {anti_selected}")


        #fig.show()
        st.pyplot(fig)





if selected == "Make Prediction":
    st.markdown("This is where the user selects the individual features to make the predictions on")


if selected == "About":
    st.markdown("About the Team Members")
    st.markdown("About the Competition")
    st.markdown("About the Web App")

    


    #st.write("Summary Statistics")
    #data = euro_df
    #st.write(data.describe())

