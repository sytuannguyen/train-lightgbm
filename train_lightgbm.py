import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
warnings.simplefilter('ignore')

# Function to load data and train the model
@st.cache(suppress_st_warning=True)
def load_and_train_model(train_data_path, test_data_path):
    # Load train data
    train_data = pd.read_csv(train_data_path)
    
    # Load test data
    test_data = pd.read_csv(test_data_path)

    # Define features and target variable
    features = ['VehYear', 'VehicleAge', 'WheelTypeID', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice', 'MMRCurrentAuctionAveragePrice',
                'MMRAcquisitionAuctionCleanPrice', 'MMRCurrentAuctionCleanPrice', 'VehBCost', 'WarrantyCost']
    target = 'IsBadBuy'
    
    # Initialize Stratified K-Fold
    num_folds = 5
    kf = StratifiedKFold(n_splits=num_folds, random_state=42, shuffle=True)

    # Initialize empty array to store test predictions
    test_predictions = np.zeros((len(test_data),))

    # Perform 5-fold cross-validation
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_data, train_data[target])):
        print(f'Fold {fold + 1}')

        # Split data into train and validation sets
        train_set = train_data.loc[train_idx, features], train_data.loc[train_idx, target]
        valid_set = train_data.loc[valid_idx, features], train_data.loc[valid_idx, target]

        # Create LightGBM datasets
        train_data_lgb = lgb.Dataset(train_set[0], label=train_set[1])
        valid_data_lgb = lgb.Dataset(valid_set[0], label=valid_set[1], reference=train_data_lgb)

        # Define LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }

        # Train LightGBM model
        num_round = 10000
        early_stopping_rounds = 100
        model = lgb.LGBMClassifier(**params)
        model.train(params, train_data_lgb, num_round, valid_sets=[valid_data_lgb])

        # Make predictions on the test set for this fold
        test_predictions += bst.predict(test_data[features]) / num_folds

    # Add predictions to the test_data DataFrame
    test_data['IsBadBuy'] = test_predictions

    return test_data

# Streamlit UI elements
st.title('DontGetKicked Prediction App')

# File upload for train and test data
train_data_file = st.file_uploader('Upload Train Data (CSV)', type=['csv'])
test_data_file = st.file_uploader('Upload Test Data (CSV)', type=['csv'])

if train_data_file is not None and test_data_file is not None:
    # Load and train the model
    test_predictions_df = load_and_train_model(train_data_file, test_data_file)

    # Display the predictions
    st.subheader('Test Data Predictions')
    st.write(test_predictions_df)
