'''
functions required to retrain model if required
'''
import os
import logging
import json
import random
from pickle import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from process_utils import prepare_data

MODEL_PATH = f"{os.getcwd()}/models/rf_model.pkl"
DATA_PATH = f'{os.getcwd()}/data/data.json'

logging.getLogger().setLevel(logging.INFO)

def load_data() -> list[dict]:
    'loads data from a json file'
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_model(model:object) -> None:
    'saves model to  a pickle file'
    with open(MODEL_PATH, 'wb') as model_file:
        dump(model, model_file)

def prepare_train_data(data: list[dict]) -> tuple[list, list]:
    'prepares data for training'
    labels = [x.get('label') for x in data]
    features = [prepare_data(x.get('talent'), x.get('job')) for x in data]
    return features, labels

def get_train_test_data(x:list, y:list) -> tuple[list,list,list,list]:
    'splits data into train/test datasets'
    n = len(y)
    random.seed(30)
    train_idx = random.sample(list(range(n)), k=int(0.75*n))
    test_idx = [i for i in range(n) if i not in train_idx]
    x_train = [x[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    x_test = [x[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return x_train, y_train, x_test, y_test

def train_model() -> None:
    'trains model'
    data = load_data()
    features, labels = prepare_train_data(data)
    feat_train, l_train, feat_test, l_test = get_train_test_data(features, labels)
    rf = RandomForestClassifier(random_state=0)
    rf.fit(feat_train, l_train)
    labels_pred = rf.predict(feat_test)
    accuracy = accuracy_score(l_test, labels_pred)
    logging.info('model retrained with accuracy:%s',accuracy)
    save_model(rf)
