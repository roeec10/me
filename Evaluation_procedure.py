import pandas as pd
from Pre_procesing import chose_dataset
from cross_validation_test import cross_validation
from Bagging_algorithm import MyBaggingID3
from sklearn.ensemble import BaggingClassifier
import itertools
import wandb

# Login to the weights & biases website using an API key.
wandb.login(key='b11dbe2ba6ec30d835bbed8fc0d3c4186d6919a1')

# Define a list of dataset names and bagging methods to evaluate.
dataset = ['car', 'monks', 'mushrooms', 'tic-tac-toe', 'spect']
method = ['MyBaggingID3', 'sklearn_BaggingClassifier']

# Set the hyperparameters for each bagging method.
Bagging_ensemble = []
Bagging_ensemble.append(MyBaggingID3(n_estimators=200, max_samples=0.7, max_features=0.9))
Bagging_ensemble.append(BaggingClassifier(n_estimators=200, max_samples=float(0.7), max_features=float(0.9)))

# Initialize a dataframe to store the performance measures and fit runtime for each dataset and bagging method.
df_template = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
               'roc_auc': [], 'fit_runtime': []}
Evaluation_metrics = pd.DataFrame(df_template)

ind = 0
# Iterate over each dataset and evaluate the predictive performance using k-fold cross validation.
for i in range(len(dataset)):
    # Load specific dataset and perform pre-processing.
    X_data, y_data = chose_dataset(dataset[i])
    # Iterate over each bagging method.
    for j in range(len(Bagging_ensemble)):
        Evaluation_metrics.loc[ind] = cross_validation(X_data, y_data,
                                                       estimator=Bagging_ensemble[j], cv=5, repetitions=2)
        ind += 1

# Add the bagging method type to the dataframe.
Evaluation_metrics.insert(0, 'Method', method * len(dataset))
# Add the dataset name to the dataframe.
dataset_names = list(itertools.chain.from_iterable(itertools.repeat(x, len(method)) for x in dataset))
Evaluation_metrics.insert(0, 'Dataset_name', dataset_names)
# Log the Evaluation_metrics table to the weights & biases website.
table = wandb.Table(dataframe=Evaluation_metrics)
run = wandb.init(project='HW1', name='predictive performance ')
wandb.log({'Evaluation_metrics': table})
wandb.finish()
