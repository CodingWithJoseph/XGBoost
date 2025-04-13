import math
from collections import defaultdict
import numpy as np
import pandas as pd

"""
Reference Papers and Website: 
A Scalable Tree Boosting System: https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf
Random Realization: https://randomrealizations.com/posts/xgboost-from-scratch/
"""


class XGBoostModel:
    def __init__(self, params, seed=2025):
        """
        :param params: Dictionary containing the hyperparameters for XGBoostModel
        :param seed: Seed defaulted to a number for reproducibility
        """
        self.boosters = None

        # Use defaultdict because key errors are handled out of box
        self.params = defaultdict(lambda: None, params)

        # Row subsampling: if it is not set sample the entire dataset
        self.subsample = self.params['subsample'] if self.params['subsample'] else 1.0

        # Learning rate slows down learning enabling the model to be less sensitive
        self.learning_rate = self.params['learning_rate'] if self.params['learning_rate'] else 0.3

        # Base score represent the models initial predictions
        self.base_score = self.params['base_score'] if self.params['base_score'] else 0.5

        # Max depth below the root node: root node + max depth
        self.max_depth = self.params['max_depth'] if self.params['max_depth'] else 5

        # Set the random number generate with a specified seed for reproducibility
        self.random_number_generator = np.random.default_rng(seed=seed)

    def fit(self, X, y, objective, num_boost_rounds, verbose=False):
        """
        :param X: Feature Space
        :param y: Label vector
        :param objective: An objective containing the loss, gradient, and hessian functions
        :param num_boost_rounds: Number of rounds we should create trees and make predictions
        :param verbose: A flag used to report metrics during training
        """
        predictions = self.base_score * np.ones(shape=y.shape)
        self.boosters = []
        for boost_round in range(num_boost_rounds):
            gradients = objective.gradient(y, predictions)
            hessians = objective.hessian(y)

            sample_idxs = None if self.subsample == 1.0 \
                else self.random_number_generator.choice(a=len(y), size=math.floor(self.subsample * len(y)),
                                                         replace=False)

            booster = BoostedTree(
                X=X,
                gradients=gradients,
                hessians=hessians,
                max_depth=self.max_depth,
                params=self.params,
                idxs=sample_idxs
            )

            predictions += self.learning_rate * booster.predict(X)
            self.boosters.append(booster)

            if verbose:
                print(f'[{boost_round}] train loss = {objective.loss(y, predictions)}')

    def predict(self, X):
        return self.base_score + self.learning_rate * np.sum([booster.predict(X) for booster in self.boosters], axis=0)


class BoostedTree:
    def __init__(self, X, gradients, hessians, max_depth, params, idxs=None):
        assert max_depth >= 0, 'max_depth must be non-negative'

        self.threshold = -1
        self.split_idx = -1

        # Params should already be wrapped in a defaultdict
        self.params = params

        # Set feature space for tree
        self.X = X

        # Gradients from previous tree: convert gradient vector to np.array if applicable and set
        self.gradients = gradients.values if isinstance(gradients, pd.Series) else gradients

        # Hessians from previous tree: convert hessian vector to np.array if applicable and set
        self.hessians = hessians.values if isinstance(hessians, pd.Series) else hessians

        self.row_idxs = idxs if idxs is not None else np.arange(len(gradients))

        # Number of examples for current tree
        self.num_examples = len(self.row_idxs)

        # Number of features for current tree
        self.num_features = X.shape[1]

        # Set the maximum depth for the tree
        self.max_depth = max_depth

        # Not sure how this works just yet: comment coming soon
        self.min_child_weight = params['min_child_weight'] if params['min_child_weight'] else 1.0

        # Regularization lambda used to scale the weight summation term
        self.reg_lambda = params['reg_lambda'] if params['reg_lambda'] else 1.0

        # Gamma (constant): gamma*T where T is the number of leafs - used to penalize trees with complex structures
        self.gamma = params['gamma'] if params['gamma'] else 0.0

        # Column subsampling: if not set use the entire column
        self.col_subsample_by_node = params["col_subsample"] if params['col_subsample'] else 1.0

        # Optimal weight value described in EQ 5 in Scalable Tree Boosting System
        self.optimal_weight = -self.gradients[idxs].sum() / (self.hessians[idxs].sum() + self.reg_lambda)

        # Keep track of the best score
        self.best_score = 0.0

        # By calling _maybe_insert_child not in the initialization function we set of a chain reaction
        # building the entire structure of a given boosted tree until max_depth is 0
        self._maybe_insert_child_node()

    def _maybe_insert_child_node(self):
        if self.max_depth <= 0:
            return

        # Consider each feature for finding best split
        for feature in range(self.num_features):
            self.find_split(feature)

        # If a split was not found then we can assume the node is a leaf
        if self.is_leaf:
            return

        # The data to be split into left and right nodes
        x = self.X.values[self.row_idxs, self.split_idx]

        left_idx = np.nonzero(x <= self.threshold)
        right_idx = np.nonzero(x > self.threshold)

        self.left = BoostedTree(
            X=self.X,
            gradients=self.gradients,
            hessians=self.hessians,
            max_depth=self.max_depth - 1,
            params=self.params,
            idxs=self.row_idxs[left_idx]
        )

        self.right = BoostedTree(
            X=self.X,
            gradients=self.gradients,
            hessians=self.hessians,
            max_depth=self.max_depth - 1,
            params=self.params,
            idxs=self.row_idxs[right_idx]
        )

    def find_split(self, feature_idx):
        """
        Refer to EQ 7 from A Scalable Tree Boosting System
        :param feature_idx:
        """
        x = self.X.values[self.row_idxs, feature_idx]
        gradients = self.gradients[self.row_idxs]
        hessians = self.hessians[self.row_idxs]

        sort_idx = np.argsort(x)
        sorted_x = x[sort_idx]
        sorted_gradients = gradients[sort_idx]
        sorted_hessians = hessians[sort_idx]

        gradient_sum = sorted_gradients.sum()
        hessian_sum = sorted_hessians.sum()

        right_child_gradient_sum = gradient_sum
        right_child_hessian_sum = hessian_sum

        left_child_gradient_sum = 0.0
        left_child_hessian_sum = 0.0

        for i in range(0, self.num_examples - 1):
            gradient = sorted_gradients[i]
            hessian = sorted_hessians[i]
            possible_split_value = sorted_x[i]
            next_split_value = sorted_x[i + 1]

            right_child_gradient_sum -= gradient
            right_child_hessian_sum -= hessian

            left_child_gradient_sum += gradient
            left_child_hessian_sum += hessian

            if left_child_hessian_sum < self.min_child_weight or possible_split_value == next_split_value:
                continue

            if right_child_hessian_sum < self.min_child_weight:
                break

            current_score = 0.5 * (
                    (left_child_gradient_sum ** 2 / (left_child_hessian_sum + self.reg_lambda)) +
                    (right_child_gradient_sum ** 2 / (right_child_hessian_sum + self.reg_lambda)) -
                    (gradient_sum ** 2 / (hessian_sum + self.reg_lambda))
            ) - (self.gamma / 2)

            # Current score represents the gain: that is, the improvement of the models predictions
            if current_score > self.best_score:
                self.best_score = current_score
                self.split_idx = feature_idx
                self.threshold = (possible_split_value + next_split_value) / 2

    def predict(self, X):
        return np.array([self._predict_row(row) for i, row in X.iterrows()])

    def _predict_row(self, row):
        if self.is_leaf:
            return self.optimal_weight
        child = self.left if row.iloc[self.split_idx] <= self.threshold \
            else self.right
        return child._predict_row(row)

    @property
    def is_leaf(self):
        return self.best_score == 0.0


class SquaredErrorObjective:
    @staticmethod
    def loss(y, predictions): return np.mean((y - predictions) ** 2)

    @staticmethod
    def gradient(y, predictions): return predictions - y

    @staticmethod
    def hessian(y): return np.ones(len(y))