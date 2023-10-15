import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold

import warnings

warnings.filterwarnings('ignore')


def preprocessing(train_url, test_url):

    df_train = pd.read_csv(train_url)
    df_train['Garage_Yr_Blt'] = df_train['Garage_Yr_Blt'].fillna(0)
    y = np.log(df_train['Sale_Price'])
    df_train = df_train.drop(['PID','Sale_Price'],axis = 1)

    
    imbalance_threshold = 0.95  # For example, 95% of the samples belonging to one category
    imbalanced_columns = []
    for column in df_train.columns:
        value_counts = df_train[column].value_counts(normalize=True)
        most_common_category = value_counts.idxmax()
        most_common_category_percentage = value_counts.max()    
    if most_common_category_percentage >= imbalance_threshold:
        imbalanced_columns.append(column)
    df_train = df_train.drop(imbalanced_columns,axis = 1)

    numeric_feats = df_train.dtypes[df_train.dtypes != "object"].index

    features_to_winsor = numeric_feats
    def revalue_column(x, M):
        if x < M:
            return x
        else:
            return M

    percentile = 0.95
    Winzorization_M = []
    for column in features_to_winsor:
        M = df_train[column].quantile(percentile)
        df_train[column] = df_train[column].apply(lambda x: revalue_column(x, M))
        Winzorization_M.append(M)
        
    

    categorical_features = df_train.select_dtypes(include=['object', 'category'])
    numerical_features = df_train.select_dtypes(exclude=['object', 'category'])

    scaler = StandardScaler()
    scaled_numerical_features = scaler.fit_transform(numerical_features)
    scaled_numerical_df = pd.DataFrame(scaled_numerical_features, columns=numerical_features.columns)

    encoder = OneHotEncoder(sparse=False,handle_unknown='ignore')
    encoded_categorical_features = encoder.fit_transform(categorical_features)
    encoded_categorical_df = pd.DataFrame(encoded_categorical_features, columns=encoder.get_feature_names_out(input_features=categorical_features.columns))

    df_encoded = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1)



    df_test = pd.read_csv(test_url)
    df_PID = df_test[['PID']]
    df_test['Garage_Yr_Blt'] = df_test['Garage_Yr_Blt'].fillna(0)
    df_test = df_test.drop(['PID'],axis = 1)
    df_test = df_test.drop(imbalanced_columns,axis = 1)
    for column, M in zip(features_to_winsor,Winzorization_M):
        df_test[column] = df_test[column].apply(lambda x: revalue_column(x, M))

    categorical_features_test = df_test.select_dtypes(include=['object', 'category'])
    numerical_features_test = df_test.select_dtypes(exclude=['object', 'category'])
    scaled_numerical_features_test = scaler.transform(numerical_features_test)
    scaled_numerical_df_test = pd.DataFrame(scaled_numerical_features_test, columns=numerical_features.columns)
    encoded_categorical_features_test = encoder.transform(categorical_features_test)
    encoded_categorical_df_test = pd.DataFrame(encoded_categorical_features_test, columns=encoder.get_feature_names_out(input_features=categorical_features.columns))
    df_encoded_test = pd.concat([scaled_numerical_df_test, encoded_categorical_df_test], axis=1)

    return df_encoded, y, df_encoded_test, df_PID
    


def model_fit(model, df_encoded, y, df_encoded_test, df_PID, random_state = 0):

    if model == 'elastic_net':
        elastic_net = ElasticNet(l1_ratio = 0.5)
        alphas = np.logspace(-4, 3, 15)
        param_grid = {'alpha': alphas}
        kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)  # Example 5-fold cross-validation

        grid_search = GridSearchCV(elastic_net, param_grid, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(df_encoded, y)

        best_alpha = grid_search.best_params_['alpha']
        best_elastic_net = ElasticNet(alpha=best_alpha, l1_ratio = 0.5, random_state = random_state)
        best_elastic_net.fit(df_encoded, y)

        y_pred = best_elastic_net.predict(df_encoded_test)


    if model == 'xgboost':
        xgb_model = XGBRegressor(n_estimators=5000, learning_rate=0.05, max_depth=6, subsample = 0.5, random_state=random_state)
        xgb_model.fit(df_encoded, y)
        y_pred = xgb_model.predict(df_encoded_test)
    
    df_PID['Sale_Price'] = np.exp(y_pred)
    return df_PID
    

if __name__ == '__main__':
    train_url = 'train.csv'
    test_url = 'test.csv'
    
    df_encoded, y, df_encoded_test, df_PID = preprocessing(train_url, test_url)
    y_pred1 = model_fit('elastic_net', df_encoded, y, df_encoded_test, df_PID)
    y_pred2 = model_fit('xgboost', df_encoded, y, df_encoded_test, df_PID)
    
    y_pred1.to_csv('mysubmission1.txt', index=False)
    y_pred2.to_csv('mysubmission2.txt', index=False)