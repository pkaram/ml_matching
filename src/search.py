import os
import pickle
from operator import itemgetter
from process_utils import prepare_data
from model_utils import train_model

MODEL_PATH = f"{os.getcwd()}/models/rf_model.pkl"

class Search:
    "class to perform single and bulk matching"
    def __init__(self, retrain: bool=False) -> None:
        self.model = self.retrain_load_model() if retrain else self.load_model()

    def retrain_load_model(self) -> object:
        'retrain & load scikit-learn model'
        train_model()
        return self.load_model()

    def load_model(self) -> object:
        'load scikit-learn model'
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model

    def match(self, talent: dict, job: dict) -> dict:
        '''
        This method takes a talent and job as input and uses the machine learning
        model to predict the label. Together with a calculated score, the dictionary
        returned has the following schema:
        {
           "talent": ...,
           "job": ...,
           "label": ...,
           "score": ...
        }        
        '''
        data = prepare_data(talent, job)
        score = self.model.predict_proba([data])[:,1][0]
        label = score >= 0.5
        return {
            'talent':talent,
            'job':job,
            'label':label,
            'score':score
        }

    def match_bulk(self, talents: list[dict], jobs: list[dict]) -> list[dict]:
        '''
        This method takes a multiple talents and jobs as input and uses the machine
        learning model to predict the label for each combination. Together with a
        calculated score, the list returned (sorted descending by score!) has the
        following schema:
        
        [
           {
             "talent": ...,
             "job": ...,
             "label": ...,
             "score": ...
           },
           {
             "talent": ...,
             "job": ...,
             "label": ...,
             "score": ...
           },
           ...
        ]
        '''
        res = [self.match(talents[i],jobs[i]) for i in range(len(talents))]
        return sorted(res, key=itemgetter('score'), reverse=True)
