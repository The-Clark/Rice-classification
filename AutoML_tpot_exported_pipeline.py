import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('/Users/dee/Desktop/Arogundade/Submission/filtered_rice.csv', sep=',')
features = tpot_data.drop('CLASS', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['CLASS'], random_state=42)

# Average CV score on the training set was: 0.9815506749967009
exported_pipeline = make_pipeline(
    PCA(iterated_power=6, svd_solver="randomized"),
    StandardScaler(),
    MLPClassifier(alpha=0.0001, learning_rate_init=0.001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
