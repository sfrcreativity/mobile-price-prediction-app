import streamlit as st
import pandas as pd
import numpy as np
import math
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier  # Example classifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import random

#Load & prep
train_data = pd.read_csv("train.csv")   #mobile_train.csv

# Select the features and target variable
X = train_data.drop(columns=['price_range'])  # Features (all columns except price)
y = train_data['price_range']  # Target (price column)

# --- Train Models ---
# RandomForest
rf_model = RandomForestClassifier(random_state=11)
rf_model.fit(X, y)

# XGBoost
xgb_model = XGBClassifier(objective='multi:softmax', num_class=4, eval_metric='mlogloss', random_state=11)
xgb_model.fit(X, y)

# Logistic Regression
log_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)
log_model.fit(X, y)

# Model dictionary
models = {
    "RandomForestClassifier": rf_model,
    "XGBoost": xgb_model,
    "LogisticRegression": log_model
}

# Price mapping
price_map = {
    0: {"label": "Low Cost", "avg_price": 14000},
    1: {"label": "Medium Cost", "avg_price": 25000},
    2: {"label": "High Cost", "avg_price": 42000},
    3: {"label": "Very High Cost", "avg_price": 70000}
}

# Split training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit app
st.title("📱 Mobile Price Prediction")

st.write("---")
# Model Selection
menu_choice = st.sidebar.radio("Menu:", ["App", "Dataset Overview", "Feature Importance", "Features Comparison","Correlation Analysis", "Accuracy Test", "About"])

if menu_choice == "App":
    st.write("This is the mobile price prediction application using various machine learning models trained on mobile phone features. Select a model and customize the phone features to classify its price range.")

    model_choice = st.radio("Select Algorithm Model:", list(models.keys()), horizontal=True)

    st.write("Customize the phone features:")
    # --- Display Section ---
    with st.expander("Display Features"):
        col1, col2 = st.columns(2)
        with col1:
            sc_h = st.slider("Screen Height (cm)", 5, 20, 13)
            sc_w = st.slider("Screen Width (cm)", 5, 20, 7)
            px_height = st.slider("Pixel Height", 400, 3000, 1080)
            px_width = st.slider("Pixel Width", 400, 3000, 1920)
            touch_screen = st.checkbox("Touch Screen", value=True)
        with col2:
            pass  # Could add more display features later

    # --- Memory Section ---
    with st.expander("Memory & Storage"):
        ram = st.slider("RAM (MB)", 256, 8000, 2048, step=256)
        int_memory = st.slider("Internal Memory (GB)", 2, 256, 32)

    # --- Camera Section ---
    with st.expander("Camera Features"):
        fc = st.slider("Front Camera (MP)", 0, 20, 5)
        pc = st.slider("Primary Camera (MP)", 0, 50, 12)

    # --- Connectivity Section ---
    with st.expander("Connectivity Features"):
        blue = st.checkbox("Bluetooth", value=True)
        wifi = st.checkbox("WiFi", value=True)
        three_g = st.checkbox("3G Support", value=True)
        four_g = st.checkbox("4G Support", value=True)
        dual_sim = st.checkbox("Dual SIM", value=True)

    # --- Battery & Performance Section ---
    with st.expander("Battery & Performance"):
        battery_power = st.slider("Battery Power (mAh)", 500, 5000, 1500, step=100)
        talk_time = st.slider("Talk Time (hours)", 1, 30, 10)
        n_cores = st.slider("Number of CPU Cores", 1, 16, 4)
        clock_speed = st.slider("Clock Speed (GHz)", 0.5, 3.0, 2.0, step=0.1)
        mobile_wt = st.slider("Mobile Weight (grams)", 80, 250, 150)
        m_dep = st.slider("Mobile Depth (cm)", 0.1, 1.0, 0.7, step=0.01)

    # --- Create DataFrame from Inputs ---
    features = pd.DataFrame([{
        "battery_power": battery_power,
        "blue": int(blue),
        "clock_speed": clock_speed,
        "dual_sim": int(dual_sim),
        "fc": fc,
        "four_g": int(four_g),
        "int_memory": int_memory,
        "m_dep": m_dep,
        "mobile_wt": mobile_wt,
        "n_cores": n_cores,
        "pc": pc,
        "px_height": px_height,
        "px_width": px_width,
        "ram": ram,
        "sc_h": sc_h,
        "sc_w": sc_w,
        "talk_time": talk_time,
        "three_g": int(three_g),
        "touch_screen": int(touch_screen),
        "wifi": int(wifi)
    }])

    # --- Predict Button ---
    if st.button("Predict Price"):

        pred_class = models[model_choice].predict(features)[0]
        category = price_map[pred_class]["label"]
        base_price = price_map[pred_class]["avg_price"]
        final_price = base_price + random.randint(-1500, 1500)

        st.success(f"Predicted Price Category: {category}")
        st.info(f"Estimated Actual Price: Rs. {final_price:,}")

elif menu_choice == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("This section provides an overview of the mobile price dataset used for training the models.")

    st.subheader("Sample Data")
    st.dataframe(train_data.head())

    st.subheader("Dataset Description")
    st.write(train_data.describe())

elif menu_choice == "Feature Importance":
    st.header("Feature Improtance")
    st.write("This section visualizes the importance of different features in predicting the mobile price range using the selected machine learning model.")

    model_choice = st.radio("Select Algorithm Model:", list(models.keys()), horizontal=True)
    
    st.subheader("Feature Importance in {}".format(model_choice))

    if model_choice == "LogisticRegression":
        # For Logistic Regression, we can use the coefficients as feature importance
        coefficients = log_model.coef_
        importance = np.mean(np.abs(coefficients), axis=0)
        #feature_importance = pd.DataFrame({
        #    'Feature': X.columns,
        #    'Importance': importance
        #}).sort_values(by='Importance', ascending=True)

        
        #fig, ax = plt.subplots(figsize=(8, 6))
        #sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax, palette='viridis')
        #st.pyplot(fig)

    else:
        classifier = models[model_choice]
        # Get feature importances from the trained model
        importance = classifier.feature_importances_

    # Create a DataFrame for easy plotting
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=True)

    # Display the table
    # print(feature_importance_df)

    # 🔹 Visualize feature importance
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax, palette='coolwarm')
            # Add text values inside bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", label_type='edge', fontsize=9, padding=3, color='black')
    ax.invert_yaxis()  # most important at top
    ax.set_title('Feature Importance for Price Prediction')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    # Display the plot in Streamlit
    st.pyplot(fig)

elif menu_choice == "Features Comparison":
    st.header("Features Comparison")
    st.write("This section compares different features in the dataset against the price range using boxplots.")

    # --- Create boxplots for each feature ---
    for feature in X.columns:
        fig, ax = plt.subplots()
        sns.boxplot(x='price_range', y=feature, data=train_data, ax=ax, palette='coolwarm')
        ax.set_title(f'{feature} vs Price Range')
        ax.set_xlabel('')
        ax.set_ylabel(feature)
        st.pyplot(fig)

elif menu_choice == "Correlation Analysis":
    st.header("Correlation Analysis")
    st.write("This section provides a correlation analysis of the features in the dataset using a heatmap visualization.")
    #corr = train_data.corr()
    #st.dataframe(corr)
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    #plt.figure(figsize=(10, 8))
    sns.heatmap(
        train_data.corr(),
        annot=True,                # show numbers
        fmt=".2f",                 # format numbers
        cmap="coolwarm",           # color scheme
        annot_kws={"size": 8},     # text size inside cells
        cbar_kws={'label': 'Correlation'}  # colorbar label
    )
    
    # Display the plot in Streamlit
    st.pyplot(fig)

    #st.write("Correlation analysis functionality is under development.")

elif menu_choice == "Accuracy Test":
    st.header("Accuracy Score and Confusion Matrix")
    st.write("This section displays the accuracy score and confusion matrix for the selected model on the test dataset. Select a model to view its performance metrics.")

    model_choice = st.radio("Select Algorithm Model:", list(models.keys()), horizontal=True)
    
    # Instantiate
    classifier = models[model_choice]    
    # Train the model on the training data
    classifier.fit(X_train, y_train)
    # Predict the price range for the test data
    y_pred = classifier.predict(X_test)

    st.write("XGBoost Accuracy:", accuracy_score(y_test, y_pred))

    # Generate classification report as dict
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Convert to DataFrame for nicer display
    report_df = pd.DataFrame(report_dict).transpose()

    st.subheader("📊 Classification Report")
    st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    # ✅ Visualization
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False,
                xticklabels=['Low', 'Medium', 'High', 'Very High'],
                yticklabels=['Low', 'Medium', 'High', 'Very High'])
    ax.set_xlabel('Predicted Cost')
    ax.set_ylabel('Actual Cost')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

elif menu_choice == "About":
    st.header("About")
    c1, c2 = st.columns(2)
    with c1:
        st.image()
    with c2:
        st.write("The app uses machine learning models to predict mobile phone prices based on various features. It is built using Streamlit and trained on a dataset of mobile phone specifications and their corresponding price ranges.")
        st.write("Dataset Source: [Mobile Price Classification Dataset on Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)")
        st.write("Developed by: Syed Fazlur Rehman")
        st.write("Version: 1.0")