import re
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE

def main():
    """
    This function is the main function to run.
    If user inputs 1, it runs the set of functions to run the model on existing data
    If user inputs 2, it runs the set of functions that update the model with new data
    
    Input: None (1x user input)
    Output: None
    """
    while True:
        selection = input("Please select one of the following options: \n \n 1: Run existing model. If you select this option, you will be prompted to input the spreadsheet with your current universe of targets and its corresponding NGP data. \n \n 2: Add data to model. If you select this option you will be prompted to input the spreadsheet with a previous quarter's universe of targets and their corresponding NGP data. \n \n Type any other key to exit and end the program") 
        
        if selection == "1":
            df_targets = solicit_inputs()
            run_model(df_targets)
            print(f"Please find your predictions in the modified file model_data.csv")
            break
        elif selection == "2":
            df_targets = solicit_inputs()
            update_model(df_targets)
            print("Loaded new data into the model. If you would like to add more or run the model, select again")
        else:
            break

def solicit_inputs():
    """
    This function takes the two input files from the user as well as a request for the relevant quarter
    It then calls a helper to clean and join the two files, returning to main
    
    Input: None (3x user inputs)
    Output: df_targets (DataFrame): a pandas dataframe with all relevant features 
    """
    
    print("\n In order to use this function, we will need an input file with potential donors for consideration. We also need an input file with all of their past contribution history. For simplicity's sake, create your NGP list of candidates as a search, and then from the search page open a contribution report. Remove all date filters, and then export it, this will be the second input. If you don't have the search saved, you can add 1000 VANIDs per step into a search \n")
    
    dataset = input("Please input the file name of your spreadsheet with potential candidates, no extension:")
    ngp_upload = input("Please input the corresponding spreadsheet with NGP contribution data, no extension:")
    
    # Matches any string with a q, a number 1-4, and a 4 number year
    pattern = r"^[Qq]\s*([1-4])\s*,*\s*(\d{4})$"
    
    while True:
        quarter = input("For which quarter is this data? i.e. if this is a Q1 mailer for 2023, type Q1 2023")
        
        match = re.match(pattern, quarter) 
        if match:
            quart = match.group(1)
            year = match.group(2)
            break
        else: 
            print("Valid quarter not detected. Please enter in the format Q1 2023")
    
    df_targets = load_clean_dataset(dataset, ngp_upload, quart, year)
    
    return df_targets

def load_clean_dataset(targets, history, quarter, year):
    """
    This function is called in solicit_inputs. 
    load_clean_dataset loads the two files named in solicit input and cleans them.
    After cleaning the two files are merged and new features are created
    The returned file has all of the columns necessary to fit the model or predict outcomes
    For convenience, files are accepted in either csv or xlsx format
    
    Input:
        targets (str): the string name of the targets sheet
        history (str): the string name of the contribution history sheet
        quarter (str): the quarter number (1 through 4) for which we are/were targeting
        year (str): the year for which this data was targeted
        
    Output:
        df_encoded (DataFrame): the cleaned and joined final dataset
    """
    
    # targets_df is the DataFrame corresponding to the spreadsheet of potential donors for inclusion
    try:
        targets_df = pd.read_csv(f'{targets}.csv')
    except: # Maybe add specific exception
        targets_df = pd.read_excel(f'{targets}.xlsx')
        
    # history_df is the DataFrame corresponding to the contribution histories of the above potential donors
    try:
        history_df = pd.read_csv(f'{history}.csv')
    except: # Maybe add specific exception
        history_df = pd.read_excel(f'{history}.xlsx')

    
    # data munging for both dfs
    targets_df.loc[:,'VANID'] = targets_df.loc[:,'VANID'].astype(str)
    targets_df.loc[:,'Region'] = targets_df.loc[:,'Region'].str.replace(" ","")
    
    history_df = history_df.iloc[:-3,:]
    history_df.loc[:,'Amount'] = history_df.loc[:,'Amount'].astype(float)
    history_df.loc[:,'Date Received'] = pd.to_datetime(history_df.loc[:,'Date Received'])
    
    # calling bound_quarter to set boundary dates for consideration
    early_bound, late_bound = bound_quarter(quarter, year)
    
    # Creating an aggregation of financial histories per donor to be merged into targets_df
    financials = (
                  history_df.loc[(history_df.loc[:,'Date Received'] <= early_bound),:]
                  .groupby('VANID')
                  .agg({'Amount': ['sum', 'count','mean'], 'Date Received': ['max','min']})
                   )
    
    financials.columns = financials.columns.droplevel()
    
    targets_df = pd.merge(targets_df, financials, how='inner', on='VANID')
    targets_df = targets_df.rename(columns={'sum': 'Total Given', 'count': 'Contributions Made', 
                                        'mean': 'Average Contribution', 'max': 'Last Contribution Date',
                                        'min':'First Contribution Date'})
    
    # Creating an aggregation of giving in the target quarter to merge into history_df
    # When we are creating predictions this will just be a bunch of zeros
    gave = (
            history_df.loc[(history_df.loc[:,'Date Received'] >= early_bound) 
                           & (history_df.loc[:,'Date Received'] < late_bound),:]
            .groupby('VANID')
            .agg({'Amount': ['sum']})
           )
    
    gave.columns = gave.columns.droplevel()
    
    targets_df = pd.merge(targets_df, gave, how='left', on='VANID')
    
    # Binary category: 1 if gave, else 0
    targets_df.loc[:,'Contributed'] = 0
    targets_df.loc[(targets_df.loc[:,'sum'].notna()),'Contributed'] = 1
    
    # Fills in those without any contributions in target period with zeros
    targets_df.loc[:,'sum'] = targets_df.loc[:,'sum'].fillna(0)
    targets_df = targets_df.rename(columns={'sum': 'Amount Given'})
    
    targets_df.loc[:,'Days Since Highest'] = (early_bound - targets_df.loc[:,'Highest Amount Date']).dt.days
    targets_df.loc[:,'Days Since Last'] = (early_bound - targets_df.loc[:,'Last Contribution Date']).dt.days
    targets_df.loc[:,'Days Since First'] = (early_bound - targets_df.loc[:,'First Contribution Date']).dt.days
    
    # Filter the dataframe to keep only relevant columns
    targets_df = targets_df.loc[:,['VANID','Region','Last Given Amount', 'Days Since Highest','Highest Amount Given',
                                   'Total Given','Contributions Made','Average Contribution','Days Since Last',
                                   'Days Since First','Amount Given','Contributed']]
    
    # Turns region into a categorical dummy variable
    df_encoded = pd.get_dummies(targets_df,columns=['Region'], drop_first=True)
    
    return df_encoded
    
def bound_quarter(quarter, year):
    """
    This function takes a quarter and year and outputs the first and last date in the relevant quarter
    
    Input:
        quarter (str): the quarter number (1 through 4) for which we are/were targeting
        year (str): the year for which this data was targeted
    Output:
        early_bound (pd datetime): the first date in the relevant quarter
        late_bound (pd datetime): the last date in the relevant quarter
    """
    
    QDATE = {'1': ('1/1/', '3/30/'), '2': ('4/1/', '6/30/'), '3': ('7/1/', '9/30/'), '4': ('10/1/','12/31/')}
    early_bound, late_bound = (date + year for date in QDATE[quarter])
    
    return pd.to_datetime(early_bound), pd.to_datetime(late_bound)

def fit_model(model=LogisticRegressionCV):
    """
    This function builds and fits a logistic regression model for use with the finance mailer project
    
    Input:
        model (sklearn model): optional, otherwise uses LogisticRegressionCV. Must have max_iter as an argument
    Ouptut:
        platt_calibrated_model: fitted and calibrated model
    """
    
    fit_data = pd.read_csv("model_data.csv")
    
    X = fit_data.drop(columns=['Amount Given','Contributed'])
    y = fit_data.loc[:,'Contributed']
    
    smote = SMOTE(random_state=42)
    X_over, y_over = smote.fit_resample(X, y)
    
    lr = model(max_iter=1000)
    lr.fit(X_over, y_over)
    platt_calibrated_model = CalibratedClassifierCV(lr, method='sigmoid', cv='prefit')
    platt_calibrated_model.fit(X_over, y_over)
    
    return platt_calibrated_model
    
def update_inputs(df_targets):
    """
    This function takes in a pandas df with new data to add for fitting the model and combines it with existing data
    It overwrites the existing filename
    
    Input:
        df_targets (DataFrame): a combined pandas dataframe already cleaned and joined
    """
    
    existing = pd.read_csv("model_data.csv")
    
    combined_data = pd.concat([existing, df_targets])
    
    # Optionally could add de-duping here to allow for the possibility someone accidentally uploads the same file again
    combined_data.to_csv("model_data.csv")
    
def run_model(df_targets):
    """
    This function takes a new sheet of potential targets
    and predicts the likelihood of them contributing if sent a mailer
    It calls fit_model to build the model and then calculates predictions.
    It outputs to a csv named pred_ followed by the date
    
    Input:
        df_targets (DataFrame): a combined pandas dataframe already cleaned and joined
    """
    
    lr = fit_model()
    
    X = df_targets.drop(columns=['Amount Given','Contributed'])
    y = df_targets.loc[:,'Contributed']
    
    calibrated_probs = lr.predict_proba(X)[:,1]
    
    isotonic_model = IsotonicRegression()
    isotonic_model.fit(calibrated_probs, y)
    final_calibrated_probs = isotonic_model.transform(calibrated_probs)
    
    df_targets['Predicted Probability'] = final_calibrated_probs
    
    df_targets = df_targets.sort_index('Predicted Probability',ascending=False)
    
    df_targets.to_csv(f'preds_{str(datetime.now().date())}')
    