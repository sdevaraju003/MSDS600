import pandas as pd
from pycaret.classification import predict_model, load_model


def load_data(filepath):
    """loads churn data into a dataframe from a string filepath"""
    df = pd.read_csv(filepath, index_col='customerID')
    return df 


def make_predictions(df):
    'load our model'
    model = load_model('RP_Churn')
    """Uses the pycaret best model to make predictions on data in the df dataframe"""
    predictions = predict_model(model, data=df)
    predictions.rename({'prediction_label': 'Churn_prediction'}, axis=1, inplace=True)
    predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No churn'}, inplace=True)
    
    return predictions['Churn_prediction']

if __name__ == "__main__":
    df = load_data('new_churn_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)    