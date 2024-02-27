import os
from enum import Enum

import pandas as pd
from boruta import BorutaPy
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from numpy import mean
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
from sklearn.metrics._scorer import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer
from sklearn.tree import plot_tree
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from DataController import DataController


class Bayes_Hyper_Params(Enum):
    K_NEIGHBORS = Integer(2, 10)
    OVER_SAMPLING_STRATEGY = Real(0.1, 0.5, prior='log-uniform')
    UNDER_SAMPLING_STRATEGY = Real(0.5, 0.9, prior='log-uniform')
    CV = Integer(2, 10)
    RANDOM_STATE = [42]
    REPLACEMENT = Categorical([True, False])
    METHOD = Categorical(['sigmoid', 'isotonic'])
    MAX_DEPTH = Integer(50, 500)
    MIN_SAMPLES_SPLIT = Real(0.00001, 1.0, prior='log-uniform')
    MIN_SAMPLES_LEAF = Real(0.00001, 1.0, prior='log-uniform')
    MIN_IMPURITY_DECREASE = Real(0.00001, 1.0, prior='log-uniform')
    CCP_ALPHA = Real(0.00001, 1.0, prior='log-uniform')
    CRITERION = Categorical(['gini', 'entropy', 'log_loss'])
    BAL_CRITERION = Categorical(['gini', 'entropy'])
    CLASS_WEIGHT = Categorical([None, 'balanced'])
    BAL_CLASS_WEIGHT = Categorical([None, 'balanced_subsample', 'balanced'])
    OOB_SCORE = Categorical([False, True])
    N_ESTIMATORS = Integer(50, 600)
    C = Real(1e-6, 1000, prior='log-uniform')
    MAX_ITER = [1000]
    NN_MAX_ITER = Integer(150, 1000)
    HIDDEN_LAYER_SIZES = Integer(100, 500)
    ACTIVATION = Categorical(['logistic', 'tanh', 'relu'])
    ALPHA = Real(0.0001, 5.0, prior='log-uniform')
    batch_size = Integer(150, 1000)
    LEARNING_RATE_INIT = Real(0.0001, 1.0, prior='log-uniform')
    TOL = Real(1e-5, 1e-3, prior='log-uniform')
    BETA_1 = Real(0.01, 0.9, prior='log-uniform')
    BETA_2 = Real(0.01, 0.999, prior='log-uniform')
    EPSILON = Real(1e-8, 1e-6, prior='log-uniform')
    N_ITER_NO_CHANGE = Integer(10, 20)
    shuffle = Categorical([False, True])
    early_stopping = Categorical([False, True])
    power_t = Real(0.01, 5.0, prior='log-uniform')
    momentum = Real(0.001, 1.0, prior='log-uniform')
    max_fun = Integer(15000, 50000)
    GAMMA = Real(1e-6, 1e+1, prior='log-uniform')
    DEGREE = Integer(2, 9)
    N_JOBS = [-1]


class MyPipeline(Pipeline):
    # https://stackoverflow.com/questions/51412845/best-found-pca-estimator-to-be-used-as-the-estimator-in-rfecv/51418655#51418655
    @property
    def coef_(self):
        return self._final_estimator.coef_

    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_


class ModelController(object):

    @staticmethod
    def configure_Decision_Tree_Model(X, y, pipeline, cv):
        """ configure Decision tree """

        if isinstance(pipeline.steps[2][1], CalibratedClassifierCV):
            decisionTree = {
                "over__k_neighbors": Bayes_Hyper_Params.K_NEIGHBORS.value,
                "over__sampling_strategy": Bayes_Hyper_Params.OVER_SAMPLING_STRATEGY.value,
                "over__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                "under__sampling_strategy": Bayes_Hyper_Params.UNDER_SAMPLING_STRATEGY.value,
                "under__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                "under__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                "model__cv": Bayes_Hyper_Params.CV.value,
                "model__method": Bayes_Hyper_Params.METHOD.value,
                "model__n_jobs": Bayes_Hyper_Params.N_JOBS.value,
                "model__base_estimator__max_depth": Bayes_Hyper_Params.MAX_DEPTH.value,
                "model__base_estimator__min_samples_split": Bayes_Hyper_Params.MIN_SAMPLES_SPLIT.value,
                "model__base_estimator__min_samples_leaf": Bayes_Hyper_Params.MIN_SAMPLES_LEAF.value,
                "model__base_estimator__min_impurity_decrease": Bayes_Hyper_Params.MIN_IMPURITY_DECREASE.value,
                "model__base_estimator__criterion": Bayes_Hyper_Params.CRITERION.value,
                "model__base_estimator__ccp_alpha": Bayes_Hyper_Params.CCP_ALPHA.value,
                "model__base_estimator__class_weight": Bayes_Hyper_Params.CLASS_WEIGHT.value,
                "model__base_estimator__random_state": Bayes_Hyper_Params.RANDOM_STATE.value
            }
        else:
            decisionTree = {
                "over__k_neighbors": Bayes_Hyper_Params.K_NEIGHBORS.value,
                "over__sampling_strategy": Bayes_Hyper_Params.OVER_SAMPLING_STRATEGY.value,
                "over__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                "under__sampling_strategy": Bayes_Hyper_Params.UNDER_SAMPLING_STRATEGY.value,
                "under__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                "under__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                "model__max_depth": Bayes_Hyper_Params.MAX_DEPTH.value,
                "model__min_samples_split": Bayes_Hyper_Params.MIN_SAMPLES_SPLIT.value,
                "model__min_samples_leaf": Bayes_Hyper_Params.MIN_SAMPLES_LEAF.value,
                "model__min_impurity_decrease": Bayes_Hyper_Params.MIN_IMPURITY_DECREASE.value,
                "model__criterion": Bayes_Hyper_Params.CRITERION.value,
                "model__ccp_alpha": Bayes_Hyper_Params.CCP_ALPHA.value,
                "model__class_weight": Bayes_Hyper_Params.CLASS_WEIGHT.value,
                "model__random_state": Bayes_Hyper_Params.RANDOM_STATE.value
            }

        search = ModelController.configure_BayesSearchCV(decisionTree, cv, pipeline)

        search.fit(X, y)

        print(search.total_iterations)

        pipeline = search.best_estimator_
        return pipeline

    @staticmethod
    def configure_random_Forest_Model(X, y, model, pipeline, cv):

        if isinstance(pipeline.steps[2][1], CalibratedClassifierCV):
            RandomForest = {
                "over__k_neighbors": Bayes_Hyper_Params.K_NEIGHBORS.value,
                "over__sampling_strategy": Bayes_Hyper_Params.OVER_SAMPLING_STRATEGY.value,
                "over__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                "under__sampling_strategy": Bayes_Hyper_Params.UNDER_SAMPLING_STRATEGY.value,
                "under__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                "under__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                "model__cv": Bayes_Hyper_Params.CV.value,
                "model__method": Bayes_Hyper_Params.METHOD.value,
                "model__n_jobs": Bayes_Hyper_Params.N_JOBS.value,
                "model__base_estimator__n_estimators": Bayes_Hyper_Params.N_ESTIMATORS.value,
                "model__base_estimator__max_depth": Bayes_Hyper_Params.MAX_DEPTH.value,
                "model__base_estimator__min_samples_split": Bayes_Hyper_Params.MIN_SAMPLES_SPLIT.value,
                "model__base_estimator__min_samples_leaf": Bayes_Hyper_Params.MIN_SAMPLES_LEAF.value,
                "model__base_estimator__min_impurity_decrease": Bayes_Hyper_Params.MIN_IMPURITY_DECREASE.value,
                "model__base_estimator__criterion": Bayes_Hyper_Params.CRITERION.value,
                "model__base_estimator__ccp_alpha": Bayes_Hyper_Params.CCP_ALPHA.value,
                "model__base_estimator__oob_score": Bayes_Hyper_Params.OOB_SCORE.value,
                "model__base_estimator__class_weight": Bayes_Hyper_Params.BAL_CLASS_WEIGHT.value,
                "model__base_estimator__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                "model__base_estimator__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                "model__base_estimator__n_jobs": Bayes_Hyper_Params.N_JOBS.value
            }

        else:
            RandomForest = {
                "over__k_neighbors": Bayes_Hyper_Params.K_NEIGHBORS.value,
                "over__sampling_strategy": Bayes_Hyper_Params.OVER_SAMPLING_STRATEGY.value,
                "over__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                "under__sampling_strategy": Bayes_Hyper_Params.UNDER_SAMPLING_STRATEGY.value,
                "under__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                "under__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                "model__n_estimators": Bayes_Hyper_Params.N_ESTIMATORS.value,
                "model__max_depth": Bayes_Hyper_Params.MAX_DEPTH.value,
                "model__oob_score": Bayes_Hyper_Params.OOB_SCORE.value,
                "model__min_samples_split": Bayes_Hyper_Params.MIN_SAMPLES_SPLIT.value,
                "model__min_samples_leaf": Bayes_Hyper_Params.MIN_SAMPLES_LEAF.value,
                "model__min_impurity_decrease": Bayes_Hyper_Params.MIN_IMPURITY_DECREASE.value,
                "model__criterion": Bayes_Hyper_Params.CRITERION.value,
                "model__ccp_alpha": Bayes_Hyper_Params.CCP_ALPHA.value,
                "model__class_weight": Bayes_Hyper_Params.BAL_CLASS_WEIGHT.value,
                "model__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                "model__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                "model__n_jobs": Bayes_Hyper_Params.N_JOBS.value
            }

        if isinstance(model, BalancedRandomForestClassifier):
            RandomForest['model__criterion'] = Bayes_Hyper_Params.BAL_CRITERION.value

        elif isinstance(model, BalancedRandomForestClassifier) and isinstance(pipeline.steps[2][1], CalibratedClassifierCV):
            RandomForest['model__base_estimator__criterion'] = Bayes_Hyper_Params.BAL_CRITERION.value
        elif isinstance(model, RandomForestClassifier):
            RandomForest.pop('model__replacement')
        else:
            RandomForest.pop('model__base_estimator__replacement')

        search = ModelController.configure_BayesSearchCV(RandomForest, cv, pipeline)
        search.fit(X, y)

        pipeline = search.best_estimator_
        return pipeline

    @staticmethod
    def configure_SVC_Model(X, y, pipeline, cv):

        if isinstance(pipeline.steps[2][1], CalibratedClassifierCV):
            SVCParams = [
                {"over__k_neighbors": Bayes_Hyper_Params.K_NEIGHBORS.value,
                 "over__sampling_strategy": Bayes_Hyper_Params.OVER_SAMPLING_STRATEGY.value,
                 "over__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "under__sampling_strategy": Bayes_Hyper_Params.UNDER_SAMPLING_STRATEGY.value,
                 "under__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                 "under__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "model__cv": Bayes_Hyper_Params.CV.value,
                 "model__method": Bayes_Hyper_Params.METHOD.value,
                 "model__n_jobs": Bayes_Hyper_Params.N_JOBS.value,
                 "model__base_estimator__kernel": Categorical(["poly"]),
                 "model__base_estimator__gamma": Bayes_Hyper_Params.GAMMA.value,
                 "model__base_estimator__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "model__base_estimator__C": Bayes_Hyper_Params.C.value,
                 "model__base_estimator__degree": Bayes_Hyper_Params.DEGREE.value,
                 "model__base_estimator__class_weight": Bayes_Hyper_Params.CLASS_WEIGHT.value,
                 "model__base_estimator__max_iter": Bayes_Hyper_Params.MAX_ITER.value,
                 "model__base_estimator__tol": Bayes_Hyper_Params.TOL.value
                 },

                {"over__k_neighbors": Bayes_Hyper_Params.K_NEIGHBORS.value,
                 "over__sampling_strategy": Bayes_Hyper_Params.OVER_SAMPLING_STRATEGY.value,
                 "over__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "under__sampling_strategy": Bayes_Hyper_Params.UNDER_SAMPLING_STRATEGY.value,
                 "under__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                 "under__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "model__cv": Bayes_Hyper_Params.CV.value,
                 "model__method": Bayes_Hyper_Params.METHOD.value,
                 "model__n_jobs": Bayes_Hyper_Params.N_JOBS.value,
                 "model__base_estimator__kernel": Categorical(["rbf"]),
                 "model__base_estimator__gamma": Bayes_Hyper_Params.GAMMA.value,
                 "model__base_estimator__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "model__base_estimator__C": Bayes_Hyper_Params.C.value,
                 "model__base_estimator__class_weight": Bayes_Hyper_Params.CLASS_WEIGHT.value,
                 "model__base_estimator__max_iter": Bayes_Hyper_Params.MAX_ITER.value,
                 "model__base_estimator__tol": Bayes_Hyper_Params.TOL.value
                 },

                {"over__k_neighbors": Bayes_Hyper_Params.K_NEIGHBORS.value,
                 "over__sampling_strategy": Bayes_Hyper_Params.OVER_SAMPLING_STRATEGY.value,
                 "over__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "under__sampling_strategy": Bayes_Hyper_Params.UNDER_SAMPLING_STRATEGY.value,
                 "under__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                 "under__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "model__cv": Bayes_Hyper_Params.CV.value,
                 "model__method": Bayes_Hyper_Params.METHOD.value,
                 "model__n_jobs": Bayes_Hyper_Params.N_JOBS.value,
                 "model__base_estimator__kernel": Categorical(["sigmoid"]),
                 "model__base_estimator__gamma": Bayes_Hyper_Params.GAMMA.value,
                 "model__base_estimator__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "model__base_estimator__C": Bayes_Hyper_Params.C.value,
                 "model__base_estimator__class_weight": Bayes_Hyper_Params.CLASS_WEIGHT.value,
                 "model__base_estimator__max_iter": Bayes_Hyper_Params.MAX_ITER.value,
                 "model__base_estimator__tol": Bayes_Hyper_Params.TOL.value
                 }
            ]
        else:
            SVCParams = [
                {"over__k_neighbors": Bayes_Hyper_Params.K_NEIGHBORS.value,
                 "over__sampling_strategy": Bayes_Hyper_Params.OVER_SAMPLING_STRATEGY.value,
                 "over__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "under__sampling_strategy": Bayes_Hyper_Params.UNDER_SAMPLING_STRATEGY.value,
                 "under__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                 "under__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "model__kernel": Categorical(["poly"]),
                 "model__gamma": Bayes_Hyper_Params.GAMMA.value,
                 "model__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "model__C": Bayes_Hyper_Params.C.value,
                 "model__degree": Bayes_Hyper_Params.DEGREE.value,
                 "model__class_weight": Bayes_Hyper_Params.CLASS_WEIGHT.value,
                 "model__max_iter": Bayes_Hyper_Params.MAX_ITER.value,
                 "model__tol": Bayes_Hyper_Params.TOL.value
                 },

                {"over__k_neighbors": Bayes_Hyper_Params.K_NEIGHBORS.value,
                 "over__sampling_strategy": Bayes_Hyper_Params.OVER_SAMPLING_STRATEGY.value,
                 "over__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "under__sampling_strategy": Bayes_Hyper_Params.UNDER_SAMPLING_STRATEGY.value,
                 "under__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                 "under__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "model__kernel": Categorical(["rbf"]),
                 "model__gamma": Bayes_Hyper_Params.GAMMA.value,
                 "model__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "model__C": Bayes_Hyper_Params.C.value,
                 "model__class_weight": Bayes_Hyper_Params.CLASS_WEIGHT.value,
                 "model__max_iter": Bayes_Hyper_Params.MAX_ITER.value,
                 "model__tol": Bayes_Hyper_Params.TOL.value
                 },

                {"over__k_neighbors": Bayes_Hyper_Params.K_NEIGHBORS.value,
                 "over__sampling_strategy": Bayes_Hyper_Params.OVER_SAMPLING_STRATEGY.value,
                 "over__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "under__sampling_strategy": Bayes_Hyper_Params.UNDER_SAMPLING_STRATEGY.value,
                 "under__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                 "under__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "model__kernel": Categorical(["sigmoid"]),
                 "model__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                 "model__gamma": Bayes_Hyper_Params.GAMMA.value,
                 "model__C": Bayes_Hyper_Params.C.value,
                 "model__class_weight": Bayes_Hyper_Params.CLASS_WEIGHT.value,
                 "model__max_iter": Bayes_Hyper_Params.MAX_ITER.value,
                 "model__tol": Bayes_Hyper_Params.TOL.value
                 }
            ]

        search = ModelController.configure_BayesSearchCV(SVCParams, cv, pipeline)
        search.fit(X, y)

        pipeline = search.best_estimator_
        return pipeline

    @staticmethod
    def configure_MLPClassifier_Model(X, y, pipeline, cv):

        if isinstance(pipeline.steps[2][1], CalibratedClassifierCV):
            MLPCParams = [
                {
                    "over__k_neighbors": Bayes_Hyper_Params.K_NEIGHBORS.value,
                    "over__sampling_strategy": Bayes_Hyper_Params.OVER_SAMPLING_STRATEGY.value,
                    "over__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                    "under__sampling_strategy": Bayes_Hyper_Params.UNDER_SAMPLING_STRATEGY.value,
                    "under__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                    "under__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                    "model__cv": Bayes_Hyper_Params.CV.value,
                    "model__method": Bayes_Hyper_Params.METHOD.value,
                    "model__n_jobs": Bayes_Hyper_Params.N_JOBS.value,
                    "model__base_estimator__solver": ["adam"],
                    "model__base_estimator__hidden_layer_sizes": Bayes_Hyper_Params.HIDDEN_LAYER_SIZES.value,
                    "model__base_estimator__activation": Bayes_Hyper_Params.ACTIVATION.value,
                    "model__base_estimator__alpha": Bayes_Hyper_Params.ALPHA.value,
                    "model__base_estimator__learning_rate_init": Bayes_Hyper_Params.LEARNING_RATE_INIT.value,
                    "model__base_estimator__max_iter": Bayes_Hyper_Params.MAX_ITER.value,
                    "model__base_estimator__random_state": [42],
                    "model__base_estimator__tol": Bayes_Hyper_Params.TOL.value,
                    "model__base_estimator__beta_1": Bayes_Hyper_Params.BETA_1.value,
                    "model__base_estimator__beta_2": Bayes_Hyper_Params.BETA_2.value,
                    "model__base_estimator__epsilon": Bayes_Hyper_Params.EPSILON.value,
                    "model__base_estimator__n_iter_no_change": Bayes_Hyper_Params.N_ITER_NO_CHANGE.value
                },

                {
                    "over__k_neighbors": Bayes_Hyper_Params.K_NEIGHBORS.value,
                    "over__sampling_strategy": Bayes_Hyper_Params.OVER_SAMPLING_STRATEGY.value,
                    "over__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                    "under__sampling_strategy": Bayes_Hyper_Params.UNDER_SAMPLING_STRATEGY.value,
                    "under__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                    "under__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                    "model__cv": Bayes_Hyper_Params.CV.value,
                    "model__method": Bayes_Hyper_Params.METHOD.value,
                    "model__n_jobs": Bayes_Hyper_Params.N_JOBS.value,
                    "model__base_estimator__solver": ["sgd"],
                    "model__base_estimator__hidden_layer_sizes": Bayes_Hyper_Params.HIDDEN_LAYER_SIZES.value,
                    "model__base_estimator__activation": Bayes_Hyper_Params.ACTIVATION.value,
                    "model__base_estimator__alpha": Bayes_Hyper_Params.ALPHA.value,
                    "model__base_estimator__learning_rate": Categorical(['constant', 'invscaling', 'adaptive']),
                    "model__base_estimator__learning_rate_init": Bayes_Hyper_Params.LEARNING_RATE_INIT.value,
                    "model__base_estimator__power_t": Bayes_Hyper_Params.power_t.value,
                    "model__base_estimator__momentum": Bayes_Hyper_Params.momentum.value,
                    "model__base_estimator__nesterovs_momentum": Categorical([True, False]),
                    "model__base_estimator__max_iter": Bayes_Hyper_Params.MAX_ITER.value,
                    "model__base_estimator__random_state": [42],
                    "model__base_estimator__tol": Bayes_Hyper_Params.TOL.value,
                    "model__base_estimator__n_iter_no_change": Bayes_Hyper_Params.N_ITER_NO_CHANGE.value
                },

                {
                    "over__k_neighbors": Bayes_Hyper_Params.K_NEIGHBORS.value,
                    "over__sampling_strategy": Bayes_Hyper_Params.OVER_SAMPLING_STRATEGY.value,
                    "over__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                    "under__sampling_strategy": Bayes_Hyper_Params.UNDER_SAMPLING_STRATEGY.value,
                    "under__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                    "under__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                    "model__cv": Bayes_Hyper_Params.CV.value,
                    "model__method": Bayes_Hyper_Params.METHOD.value,
                    "model__n_jobs": Bayes_Hyper_Params.N_JOBS.value,
                    "model__base_estimator__solver": ["lbfgs"],
                    "model__base_estimator__hidden_layer_sizes": Bayes_Hyper_Params.HIDDEN_LAYER_SIZES.value,
                    "model__base_estimator__activation": Bayes_Hyper_Params.ACTIVATION.value,
                    "model__base_estimator__alpha": Bayes_Hyper_Params.ACTIVATION.value,
                    "model__base_estimator__max_iter": Bayes_Hyper_Params.MAX_ITER.value,
                    "model__base_estimator__random_state": [42],
                    "model__base_estimator__tol": Bayes_Hyper_Params.TOL.value,
                    "model__base_estimator__max_fun": Bayes_Hyper_Params.max_fun.value
                }
            ]
        else:
            MLPCParams = [
                {
                    "over__k_neighbors": Bayes_Hyper_Params.K_NEIGHBORS.value,
                    "over__sampling_strategy": Bayes_Hyper_Params.OVER_SAMPLING_STRATEGY.value,
                    "over__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                    "under__sampling_strategy": Bayes_Hyper_Params.UNDER_SAMPLING_STRATEGY.value,
                    "under__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                    "under__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                    "model__solver": ["adam"],
                    "model__hidden_layer_sizes": Bayes_Hyper_Params.HIDDEN_LAYER_SIZES.value,
                    "model__activation": Bayes_Hyper_Params.ACTIVATION.value,
                    "model__alpha": Bayes_Hyper_Params.ALPHA.value,
                    "model__learning_rate_init": Bayes_Hyper_Params.LEARNING_RATE_INIT.value,
                    "model__max_iter": Bayes_Hyper_Params.NN_MAX_ITER.value,
                    "model__random_state": [42],
                    "model__tol": Bayes_Hyper_Params.TOL.value,
                    "model__early_stopping": Bayes_Hyper_Params.early_stopping.value,
                    "model__batch_size": Bayes_Hyper_Params.batch_size.value,
                    "model__shuffle": Bayes_Hyper_Params.shuffle.value,
                    "model__beta_1": Bayes_Hyper_Params.BETA_1.value,
                    "model__beta_2": Bayes_Hyper_Params.BETA_2.value,
                    "model__epsilon": Bayes_Hyper_Params.EPSILON.value,
                    "model__n_iter_no_change": Bayes_Hyper_Params.N_ITER_NO_CHANGE.value
                },

                {
                    "over__k_neighbors": Bayes_Hyper_Params.K_NEIGHBORS.value,
                    "over__sampling_strategy": Bayes_Hyper_Params.OVER_SAMPLING_STRATEGY.value,
                    "over__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                    "under__sampling_strategy": Bayes_Hyper_Params.UNDER_SAMPLING_STRATEGY.value,
                    "under__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                    "under__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                    "model__solver": ["sgd"],
                    "model__hidden_layer_sizes": Bayes_Hyper_Params.HIDDEN_LAYER_SIZES.value,
                    "model__activation": Bayes_Hyper_Params.ACTIVATION.value,
                    "model__alpha": Bayes_Hyper_Params.ALPHA.value,
                    "model__learning_rate": Categorical(['constant', 'invscaling', 'adaptive']),
                    "model__learning_rate_init": Bayes_Hyper_Params.LEARNING_RATE_INIT.value,
                    "model__power_t": Bayes_Hyper_Params.power_t.value,
                    "model__momentum": Bayes_Hyper_Params.momentum.value,
                    "model__nesterovs_momentum": Categorical([True, False]),
                    "model__max_iter": Bayes_Hyper_Params.NN_MAX_ITER.value,
                    "model__early_stopping": Bayes_Hyper_Params.early_stopping.value,
                    "model__random_state": [42],
                    "model__tol": Bayes_Hyper_Params.TOL.value,
                    "model__shuffle": Bayes_Hyper_Params.shuffle.value,
                    "model__batch_size": Bayes_Hyper_Params.batch_size.value,
                    "model__n_iter_no_change": Bayes_Hyper_Params.N_ITER_NO_CHANGE.value
                },

                {
                    "over__k_neighbors": Bayes_Hyper_Params.K_NEIGHBORS.value,
                    "over__sampling_strategy": Bayes_Hyper_Params.OVER_SAMPLING_STRATEGY.value,
                    "over__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                    "under__sampling_strategy": Bayes_Hyper_Params.UNDER_SAMPLING_STRATEGY.value,
                    "under__replacement": Bayes_Hyper_Params.REPLACEMENT.value,
                    "under__random_state": Bayes_Hyper_Params.RANDOM_STATE.value,
                    "model__solver": ["lbfgs"],
                    "model__hidden_layer_sizes": Bayes_Hyper_Params.HIDDEN_LAYER_SIZES.value,
                    "model__activation": Bayes_Hyper_Params.ACTIVATION.value,
                    "model__alpha": Bayes_Hyper_Params.ALPHA.value,
                    "model__max_iter": Bayes_Hyper_Params.NN_MAX_ITER.value,
                    "model__random_state": [42],
                    "model__tol": Bayes_Hyper_Params.TOL.value,
                    "model__early_stopping": Bayes_Hyper_Params.early_stopping.value,
                    "model__shuffle": Bayes_Hyper_Params.shuffle.value,
                    "model__batch_size": Bayes_Hyper_Params.batch_size.value,
                    "model__max_fun": Bayes_Hyper_Params.max_fun.value
                }
            ]
        search = ModelController.configure_BayesSearchCV(MLPCParams, cv, pipeline)
        search.fit(X, y)
        print(search.best_score_)
        print(search.total_iterations)
        print(search.best_estimator_)
        pipeline = search.best_estimator_

        return pipeline

    @staticmethod
    def Create_Decision_Tree_plot(model, X, features, title):
        one_hot_encoded_frame = pd.DataFrame.sparse.from_spmatrix(X, columns=features)
        plt.figure(figsize=[20, 10])
        plot_tree(model.steps[2][1], filled=True, rounded=True, max_depth=3, fontsize=10,
                  feature_names=one_hot_encoded_frame.columns, class_names=['disorder', 'no disorder'])
        plt.title(title)
        plt.show()

    @staticmethod
    def classifier_testing(model, X, y, cv, to_print):

        f2_scores = cross_val_score(model, X, y, scoring=make_scorer(fbeta_score, beta=2), cv=cv, n_jobs=-1, error_score=False)
        f2_scores = mean(f2_scores)
        f1_scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
        f1_scores = mean(f1_scores)
        roc_auc_scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
        roc_auc_scores = mean(roc_auc_scores)
        precision_scores = cross_val_score(model, X, y, scoring='precision', cv=cv, n_jobs=-1)
        precision_scores = mean(precision_scores)
        recall_scores = cross_val_score(model, X, y, scoring='recall', cv=cv, n_jobs=-1)
        recall_scores = mean(recall_scores)

        if to_print:
            print(f'> {model}')
            print(f'> Roc Auc: {roc_auc_scores:.3f}')
            print(f'> Mean f1: {f1_scores:.3f}')
            print(f'> Mean f2: {f2_scores:.3f}')
            print(f'> recall: {recall_scores:.3f}')
            print(f'> precision: {precision_scores:.3f}')

        return {'Roc Auc': roc_auc_scores, 'f1': f1_scores,
                'f2': f2_scores, 'recall': recall_scores, 'precision': precision_scores}

    @staticmethod
    def column_encoding(X, features):
        # determine categorical and numerical features
        numerical_ix = X.select_dtypes(include=['int64', 'float64', 'int']).columns
        categorical_ix = X.select_dtypes(include=['object', 'bool', 'category']).columns
        X = X.astype(str)

        # define the data preparation for the columns and transform X
        t = [('cat', OneHotEncoder(), categorical_ix), ('scale', StandardScaler(), numerical_ix),  ('norm', Normalizer(), numerical_ix)]
        # t = [('cat', OneHotEncoder(), categorical_ix), ('scale', StandardScaler(), numerical_ix)]
        col_transform = ColumnTransformer(transformers=t)

        features = features.drop('(D) Any prescribed Antipsychotic medications taken in last 7 days (binary)')
        X_trans = col_transform.fit_transform(X)
        features = col_transform.get_feature_names_out(features)

        return X_trans, features

    @staticmethod
    def configure_BayesSearchCV(params, cv, pipeline):

        search = BayesSearchCV(estimator=pipeline,
                               cv=cv,
                               search_spaces=params,
                               scoring=make_scorer(fbeta_score, beta=2),
                               # optimizer_kwargs={'base_estimator': 'RF', 'random_state': 42},
                               n_iter=150,
                               random_state=42,
                               n_jobs=-1,
                               verbose=10,
                               n_points=1
                               )
        return search

    @staticmethod
    def produce_boruta(X, y):

        X = X.toarray()

        if os.path.exists('boruta_feat_selector.obj'):
            feat_selector = DataController.load_pickled_object(name='boruta_feat_selector.obj')

            return feat_selector
        else:
            forest = RandomForestClassifier(n_jobs=-1, max_depth=7, random_state=42)
            feat_selector = BorutaPy(forest, max_iter=200, n_estimators='auto', verbose=2, random_state=42, perc=95)

            feat_selector.fit(X, y)
            DataController.pickle_object('boruta_feat_selector', feat_selector)

        return feat_selector
