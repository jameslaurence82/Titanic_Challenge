
from flask import Flask, request
import pandas as pd
import os
import json
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Define file paths
model_path = os.path.join(os.path.pardir, os.path.pardir, 'models')
model_fp = os.path.join(model_path, 'lr_model.pkl')
scaler_fp = os.path.join(model_path, 'lr_scaler.pkl')

scaler = pickle.load(open(scaler_fp, 'rb'))
model = pickle.load(open(model_fp, 'rb'))

# set column order to match pickle files
columns = [
    'Age', 'Fare', 'Family_Size', 'Is_Mother', 'Is_Male', 'Is_Female', 
    'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 
    'Deck_Z', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Title_Lady', 'Title_Master', 
    'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Sir', 
    'Fare_Bin_very_low', 'Fare_Bin_low', 'Fare_Bin_high', 'Fare_Bin_very_high', 
    'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Age_State_Adult', 
    'Age_State_Child', 'Age_State_Senior', 'Age_State_Teenager'
]


# Define an API endpoint that accepts POST requests
@app.route('/api', methods=['POST'])
def make_prediction():
    # Parse incoming JSON data
    data = json.dumps(request.get_json(force=True))
    # ceate df from json string
    df = pd.read_json(data)
    # extract passengerId's
    passenger_ids = df['PassengerId'].ravel()
    #actual survived values
    actuals = df['Survived'].ravel()
    # Extract required columns as numpy array
    X = df[columns].to_numpy().astype('float')
    # transform the input
    X_scaled = scaler.transform(X)
    # make predictions
    predictions = model.predict(X_scaled)
    # create response dataframe
    df_response = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Predicted': predictions,
        'Actual': actuals
    })
    return df_response.to_json()

# Run the app on port 10001 with debug mode enabled
if __name__ == '__main__':
    app.run(port=10001, debug=True)
