import os

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_context('paper')
sns.set_style('white')


class QcClassifier:
    """
    A classifier that will remove features with a low variance and will scale the features prior to the classification
    task.
    """

    def __init__(self, min_variance=0.0001, scaling_mode='robust', estimator=None):
        """

        Args:
            min_variance: Features with a training-set variance lower than this threshold will be removed.
            scaling_mode: Remove the mean and scale to unit variance if 'standard', remove the median and scale to the
                interquartile range if 'robust'.
            estimator: The classification estimator.
        """
        self._vt = VarianceThreshold(min_variance)
        if scaling_mode == 'standard':
            self._scaler = StandardScaler()
        elif scaling_mode == 'robust':
            self._scaler = RobustScaler()
        else:
            self._scaler = StandardScaler()
        if estimator is not None:
            self._estimator = estimator
        else:
            self._estimator = RandomForestClassifier(1000, n_jobs=-1, random_state=0)

        self._transform_pipeline = make_pipeline(self._vt, self._scaler)

    def fit(self, X, y):
        """
        Train the classifier on the training set (X, y).

        Args:
            X: The training input samples.
            y: The target class labels.

        Returns:
            Returns self.
        """
        return self._estimator.fit(self._transform_pipeline.fit_transform(X, y), y)

    def predict(self, X):
        """
        Predict class for X.

        Args:
            X: The input samples.

        Returns:
            The predicted classes.
        """
        return self._estimator.predict(self._transform_pipeline.transform(X))

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Args:
            X: The input samples.

        Returns:
            The class probabilities of the input samples.
        """
        return self._estimator.predict_proba(self._transform_pipeline.transform(X))

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected.

        Args:
            indices: If True, the return value will be an array of integers, rather than a boolean mask.

        Returns:
            An index that selects the retained features from a feature vector. If indices is False, this is a boolean
            array of shape [# input features], in which an element is True iff its corresponding feature is selected for
            retention. If indices is True, this is an integer array of shape [# output features] whose values are
            indices into the input feature vector.
        """
        return self._vt.get_support(indices)


if __name__ == '__main__':
    # load the QC metrics and the quality assignments
    instruments = ['Exactive', 'LTQ_IonTrap', 'Orbitrap', 'VOrbitrap']
    metrics = ['idfree', 'idbased', 'instrument']
    for instrument in instruments:
        quality = pd.read_csv('data/{}-quality.txt'.format(instrument), '\t', index_col='Filename')
        predictions = []
        for metric in metrics:
            if os.path.exists('data/{}-{}.txt'.format(instrument, metric)):
                # read the metrics
                data = pd.read_csv('data/{}-{}.txt'.format(instrument, metric), '\t', index_col='Filename').fillna(0)
                data['Quality'] = quality
                data = data[pd.notnull(data['Quality'])]

                # convert to X, y arrays for classification
                X = data.drop('Quality', 1).values
                y = np.array([0 if quality in ['good', 'ok'] else 1 for quality in data['Quality']])

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0, stratify=data['Quality'])

                qcc = QcClassifier()
                qcc.fit(X_train, y_train)

                predictions.append((metric, y_test, qcc.predict_proba(X_test)[:, 1]))

        # plot ROC curves for all metric types
        plt.figure()

        metric_labels = {'idfree': 'ID-free', 'idbased': 'ID-based', 'instrument': 'Instrument'}
        for metric, y_true, y_score in predictions:
            # compute ROC
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            # plot the ROC curve
            fpr = np.insert(fpr, 0, 0)
            tpr = np.insert(tpr, 0, 0)
            plt.plot(fpr, tpr, label='{} (AUC = {:.2f})'.format(metric_labels[metric], auc))

        # plot the random ROC curve at 0.5
        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.legend(loc='lower right')

        plt.savefig('{}-roc.pdf'.format(instrument))
        plt.close()

        ##########

        # ensemble classifier

        avg = np.average([pred[2] for pred in predictions], axis=0)
        predictions.append(('Ensemble', predictions[0][1], avg))

        plt.figure()

        labels = [metric_labels.get(metric, metric) for metric, _, _ in predictions]
        sns.barplot(x=labels, y=[average_precision_score(y_true, y_score) for metric, y_true, y_score in predictions])

        plt.ylim(0.9, 1)

        plt.xlabel('Metrics type')
        plt.ylabel('Average precision')

        plt.savefig('{}-ensemble.pdf'.format(instrument))
        plt.close()
