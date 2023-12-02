import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm
import patsy

import time

import warnings

warnings.filterwarnings('ignore')


def preprocess(data):
    tmp = pd.to_datetime(data['Date'])
    data['Wk'] = tmp.dt.isocalendar().week
    data['Yr'] = tmp.dt.year
    data['Wk'] = pd.Categorical(data['Wk'], categories=[i for i in range(1, 53)])  # 52 weeks 
    data['IsHoliday'] = data['IsHoliday'].apply(int)
    return data

def predict(train_url, test_url):

    # Reading train data
    file_path = train_url
    train = pd.read_csv(file_path)

    # Reading test data
    file_path = test_url
    test = pd.read_csv(file_path)
    
    #SVD 
    d = 8
    test_depts = test['Dept'].unique()
    train_new = pd.DataFrame()
    for dept in test_depts:
        train_dept_data = train[train['Dept'] == dept]        
        selected_columns = train_dept_data[['Store', 'Date', 'Weekly_Sales']]

        pivoted=selected_columns.pivot(index='Date', columns='Store', values='Weekly_Sales').fillna(0)
        index_names = pivoted.index
        column_names = pivoted.columns
        train_dept_ts = np.array(pivoted)
        
        store_mean = np.mean(train_dept_ts,axis=0)
        train_dept_ts_new = train_dept_ts - store_mean
        U, S, V = np.linalg.svd(train_dept_ts_new)
        d_max = min(d,len(S))
        X_new = np.dot(U[:,:d_max],np.dot(np.diag(S[:d_max]),V[:d_max,:]))+ store_mean
        df = pd.DataFrame(X_new, columns=column_names, index=index_names)
        train_dept_new = pd.melt(df.reset_index(),id_vars='Date', var_name='Store',value_name='Weekly_Sales_SVD')
        train_dept_new['Dept'] = dept
        train_new = pd.concat([train_new, train_dept_new], ignore_index=True)
    train = train.merge(train_new,on = ['Date','Store','Dept'],how='left').fillna(0).drop('Weekly_Sales',axis=1)
    

    # pre-allocate a pd to store the predictions
    test_pred = pd.DataFrame()

    train_pairs = train[['Store', 'Dept']].drop_duplicates(ignore_index=True)
    test_pairs = test[['Store', 'Dept']].drop_duplicates(ignore_index=True)
    unique_pairs = pd.merge(train_pairs, test_pairs, how = 'inner', on =['Store', 'Dept'])

    train_split = unique_pairs.merge(train, on=['Store', 'Dept'], how='left')
    train_split = preprocess(train_split)
    y, X = patsy.dmatrices('Weekly_Sales_SVD ~ Weekly_Sales_SVD + Store + Dept + Yr  + Wk+ IsHoliday', 
                           data = train_split, 
                           return_type='dataframe')
    train_split = dict(tuple(X.groupby(['Store', 'Dept'])))


    test_split = unique_pairs.merge(test, on=['Store', 'Dept'], how='left')
    test_split = preprocess(test_split)
    y, X = patsy.dmatrices('Yr ~ Store + Dept + Yr  + Wk+ IsHoliday', 
                           data = test_split, 
                           return_type='dataframe')
    X['Date'] = test_split['Date']
    test_split = dict(tuple(X.groupby(['Store', 'Dept'])))

    keys = list(train_split)

    for key in keys:
        X_train = train_split[key]
        X_test = test_split[key]
        holidays = X_test['IsHoliday']

        Y = X_train['Weekly_Sales_SVD']
        X_train = X_train.drop(['Weekly_Sales_SVD','Store', 'Dept', 'IsHoliday'], axis=1)
        X_test = X_test.drop(['IsHoliday'], axis=1)
        
        X_train['Yr_square'] = X_train['Yr']**2
        X_test['Yr_square'] = X_test['Yr']**2
        

        cols_to_drop = X_train.columns[(X_train == 0).all()]
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)

        cols_to_drop = []
        for i in range(len(X_train.columns) - 1, 1, -1):  # Start from the last column and move backward
            col_name = X_train.columns[i]
            # Extract the current column and all previous columns
            tmp_Y = X_train.iloc[:, i].values
            tmp_X = X_train.iloc[:, :i].values

            coefficients, residuals, rank, s = np.linalg.lstsq(tmp_X, tmp_Y, rcond=None)
            if np.sum(residuals) < 1e-16:
                    cols_to_drop.append(col_name)

        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)
        

        model = sm.OLS(Y, X_train).fit()
        mycoef = model.params.fillna(0)
        
        tmp_pred = X_test[['Store', 'Dept', 'Date']]
        X_test = X_test.drop(['Store', 'Dept', 'Date'], axis=1)

        tmp_pred['Weekly_Pred'] = np.dot(X_test, mycoef)
        tmp_pred['IsHoliday'] = holidays.apply(bool)
        test_pred = pd.concat([test_pred, tmp_pred], ignore_index=True)

    test_pred['Store'] = test_pred['Store'].astype(int)
    test_pred['Dept'] = test_pred['Dept'].astype(int)
    test_pred = test.merge(test_pred, on = ['Store','Dept','Date','IsHoliday'], how = 'left')
    test_pred['Weekly_Pred'].fillna(0, inplace=True)
    file_path = 'mypred.csv'
    test_pred.to_csv(file_path, index=False)
    

if __name__ == '__main__':
    train_url = 'train.csv'
    test_url = 'test.csv'
    
    predict(train_url, test_url)