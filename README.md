 # ML matching task

To run notebook and scripts conda was used to create environment and install dependencies:

```
conda create -n ml_matching python=3.11 -y
conda activate ml_matching
pip install -r requirements.txt
```

### Part 1

The model was developed and trained on a [jupyter notebook](notebooks/task.ipynb) where comments on the steps made and summary notes are also included.

### Part 2

To examine *match* and *match_bulk* functionality a script and related utilities have been developed, based on what has been created on the jupyter notebook.

For **match** we select a random item from given data and get data required by executing the following:

```
python src/main.py
```

For **match_bulk** we select 10 random items from given data and get data required by executing the following:

```
python src/main.py --match_bulk
```

Retraining functionality has also been integrated and can be used with *retrain* flag. After retraining the model is overwritten and for all predictions following the new model is used.

```
python src/main.py --retrain
python src/main.py --retrain --match_bulk
```
