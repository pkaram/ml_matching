'''
Script to run examples on match and match_bulk methods of class Search.
The optional functionality of retraining the model is also integrated.
'''
import argparse
import random
from search import Search
from model_utils import load_data

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--match_bulk", action="store_true", help="flag for match bulk")
    parser.add_argument("--retrain", action="store_true", help="flag for retraining")
    args = parser.parse_args()
    s = Search(retrain=args.retrain)
    d = load_data()
    if args.match_bulk:
        d = random.sample(d, k=10)
        talents = [i.get('talent') for i in d]
        jobs = [i.get('job') for i in d]
        res = s.match_bulk(talents, jobs)
    else:
        d = random.choice(d)
        talent = d.get('talent')
        job = d.get('job')
        res = s.match(talent,job)
    print(res)
