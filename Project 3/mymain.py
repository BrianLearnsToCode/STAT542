import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, make_scorer

from xgboost import XGBClassifier



def predict(train_url, test_url, vocab_path):
    
    with open(vocab_path, 'r') as file:
        lines = file.readlines()
    vocabulary_dict_loaded = {}
    for line in lines:
        key, value = line.strip().split(': ')
        vocabulary_dict_loaded[key] = int(value)
        
    train = pd.read_csv(train_url, sep='\t', header=0, dtype=str)
    test =  pd.read_csv(test_url, sep='\t', header=0, dtype=str)
    train['review'] = train['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)
    test['review'] = test['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)

    stop_words = set(stopwords.words('english'))
    new_vectorizer = TfidfVectorizer(
        vocabulary = vocabulary_dict_loaded,
        preprocessor=lambda x: x.lower(),  # Convert to lowercase
        stop_words=stop_words,             # Remove stop words
        ngram_range=(1, 2),               # Use 1- to 4-grams
        min_df=0.001,                        # Minimum term frequency
        max_df=0.5,                       # Maximum document frequency
        token_pattern=r"\b[\w+\|']+\b" # Use word tokenizer: See Ethan's comment below
    )

    
    dtm_train = new_vectorizer.fit_transform(train['review'])   
    dtm_test = new_vectorizer.transform(test['review'])
    
    xgb_clf = XGBClassifier(
                            max_depth = 4,
                            n_estimators = 500,
                            learning_rate = 0.2,
                            use_label_encoder=False, 
                            min_child_weight = 6,
                            eval_metric='logloss', 
                            objective='binary:logistic').fit(dtm_train, train['sentiment'].astype(int))
    
    logreg_cv = LogisticRegressionCV(solver='liblinear').fit(dtm_train, train['sentiment'])
    
    proba_positive_class_1 = logreg_cv.predict_proba(dtm_test)[:,1]
    proba_positive_class_2 = xgb_clf.predict_proba(dtm_test)[:,1]
    proba_positive_class = 0.7*proba_positive_class_1+0.3*proba_positive_class_2
    
    mypred = pd.DataFrame()
    mypred['id'] = test['id']
    mypred['prob'] = proba_positive_class    
    file_path = 'mypred.csv'
    mypred.to_csv(file_path, index=False, sep=',')
    
    

if __name__ == '__main__':
    train_url = 'train.tsv'
    test_url = 'test.tsv'
    vocab_path = 'myvocab.txt'    
    predict(train_url, test_url,vocab_path)