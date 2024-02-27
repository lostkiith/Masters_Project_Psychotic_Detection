import os
import sys
import warnings
from time import sleep

import imblearn
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn import FunctionSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from DataController import DataController
from ModelController import ModelController


def ModellingDriver():
    try:
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"

            # prep and transform dataset
        X, y, features = DataController.prep_data(DataController.create_dataframe_from_DTA())

        # cross validation
        cv = StratifiedShuffleSplit(train_size=0.80, test_size=0.20, n_splits=10, random_state=42)

        print('Loading Classifiers')
        # produce classifiers
        classifiers = [configure_decision_tree(X, y, cv, features), configure_random_forests(X, y, cv, features),
                       configure_SVC(X, y, cv, features),
                       configure_MLPClassifier(X, y, cv, use_rfe=True, features=features)]

        create_list_of_classifiers_scores(X, y, cv, classifiers, features)

        # create_visualizations(X, y, cv, classifiers, features)

        print('Done')

    except TypeError as err:
        print(f"Unexpected {err=}, {type(err)=}")


def configure_decision_tree(X, y, cv, features) -> [Pipeline, Pipeline]:
    """returns a configured and calibrated classifier using scoring as metric for calibrating.

           returns a configured classifier that is configured by scoring list and a calibrated classifier that uses the
           CalibratedClassifierCV and the scoring metric.

            :param X: dataframe
                x as returned by the `column_encoding`.
            :param y: dataframe
                y as returned by the `prep_data`.
            :param cv: cross-validation object
                cv from 'sklearn.model_selection'
            :param bayes: to use Bayesian optimization
                cv from 'sklearn.model_selection'
           :returns calibrated_DecisionTree, configured_DecisionTree: a configured and calibrated classifier.

           """

    X, features = ModelController.column_encoding(X, features)

    over = SMOTE(n_jobs=-1, random_state=42)
    under = RandomUnderSampler(random_state=42)
    steps = [('over', over), ('under', under), ('model', DecisionTreeClassifier(random_state=42))]
    pipeline = Pipeline(steps=steps)

    # check if decision tree has been created if not create and configure
    if os.path.exists('configured_DecisionTree.obj'):
        configured_DecisionTree = DataController.load_pickled_object(name='configured_DecisionTree.obj')
    else:
        DecisionTree = DecisionTreeClassifier(random_state=42)

        pipeline.steps.pop()
        pipeline.steps.append(('model', DecisionTree))

        rfe = ModelController.produce_boruta(X, y)
        X_trans = rfe.transform(X)

        configured_DecisionTree = ModelController.configure_Decision_Tree_Model(X_trans, y, pipeline, cv)
        DataController.pickle_object('configured_DecisionTree', configured_DecisionTree)

    # check if calibrated decision tree has been created if not create and configure
    if os.path.exists('calibrated_DecisionTree.obj'):
        calibrated_DecisionTree = DataController.load_pickled_object(name='calibrated_DecisionTree.obj')
    else:
        calibrated = CalibratedClassifierCV(DecisionTreeClassifier(random_state=42), n_jobs=-1)

        pipeline.steps.pop()
        pipeline.steps.append(('model', calibrated))

        rfe = ModelController.produce_boruta(X, y)
        X_trans = rfe.transform(X)

        calibrated_DecisionTree = ModelController.configure_Decision_Tree_Model(X_trans, y, pipeline, cv)
        DataController.pickle_object('calibrated_DecisionTree', calibrated_DecisionTree)

    # return calibrated_DecisionTree and configured_DecisionTree pipelines
    return calibrated_DecisionTree, configured_DecisionTree


def configure_random_forests(X, y, cv, features) -> [RandomForestClassifier, CalibratedClassifierCV, BalancedRandomForestClassifier, CalibratedClassifierCV]:
    """returns a configured and calibrated classifier using scoring as metric for calibrating.

           returns a configured classifier that is configured by scoring list and a calibrated classifier that uses the
           CalibratedClassifierCV and the scoring metric.

            :param X: dataframe
                x as returned by the `column_encoding`.
            :param y: dataframe
                y as returned by the `prep_data`.
            :param cv: cross-validation object
                cv from 'sklearn.model_selection'
            :param features: list of features
                features as returned by the `column_encoding`.

           :returns calibrated_RandomForest, configured_randomForest, calibrated_balancedRandomForest, configured_balancedRandomForest:
                a configured and calibrated classifier.

           """
    X, features = ModelController.column_encoding(X, features)

    over = SMOTE(n_jobs=-1, random_state=42)
    under = RandomUnderSampler(random_state=42)
    steps = [('over', over), ('under', under), ('model', RandomForestClassifier(random_state=42))]
    pipeline = Pipeline(steps=steps)

    # check if decision tree has been created if not create and configure
    if os.path.exists('configured_randomForest.obj'):
        configured_randomForest = DataController.load_pickled_object(name='configured_randomForest.obj')
    else:
        ForestClassifier = RandomForestClassifier(random_state=42, n_jobs=-1)

        pipeline.steps.pop()
        pipeline.steps.append(('model', ForestClassifier))

        rfe = ModelController.produce_boruta(X, y)
        X_trans = rfe.transform(X)

        configured_randomForest = ModelController.configure_random_Forest_Model(X_trans, y, ForestClassifier, pipeline,
                                                                                cv)
        DataController.pickle_object('configured_randomForest', configured_randomForest)

    # check if calibrated decision tree has been created if not create and configure
    if os.path.exists('calibrated_RandomForest.obj'):
        calibrated_RandomForest = DataController.load_pickled_object(name='calibrated_RandomForest.obj')
    else:
        calibrated = CalibratedClassifierCV(RandomForestClassifier(random_state=42, n_jobs=-1), n_jobs=-1)

        pipeline.steps.pop()
        pipeline.steps.append(('model', calibrated))

        rfe = ModelController.produce_boruta(X, y)
        X_trans = rfe.transform(X)

        calibrated_RandomForest = ModelController.configure_random_Forest_Model(X_trans, y, calibrated, pipeline, cv)
        DataController.pickle_object('calibrated_RandomForest', calibrated_RandomForest)

    if os.path.exists('configured_balancedRandomForest.obj'):
        configured_balancedRandomForest = DataController.load_pickled_object(name='configured_balancedRandomForest.obj')
    else:
        balancedRandomForest = imblearn.ensemble.BalancedRandomForestClassifier(random_state=42, n_jobs=-1)

        pipeline.steps.pop()
        pipeline.steps.append(('model', balancedRandomForest))

        rfe = ModelController.produce_boruta(X, y)
        X_trans = rfe.transform(X)

        configured_balancedRandomForest = ModelController.configure_random_Forest_Model(X_trans, y,
                                                                                        balancedRandomForest, pipeline,
                                                                                        cv)
        DataController.pickle_object('configured_balancedRandomForest', configured_balancedRandomForest)

    # check if calibrated decision tree has been created if not create and configure
    if os.path.exists('calibrated_balancedRandomForest.obj'):
        calibrated_balancedRandomForest = DataController.load_pickled_object(name='calibrated_balancedRandomForest.obj')
    else:
        calibrated = CalibratedClassifierCV(
            imblearn.ensemble.BalancedRandomForestClassifier(random_state=42, n_jobs=-1), n_jobs=-1)

        pipeline.steps.pop()
        pipeline.steps.append(('model', calibrated))

        rfe = ModelController.produce_boruta(X, y)
        X_trans = rfe.transform(X)

        calibrated_balancedRandomForest = ModelController.configure_random_Forest_Model(X_trans, y, calibrated,
                                                                                        pipeline, cv)
        DataController.pickle_object('calibrated_balancedRandomForest', calibrated_balancedRandomForest)

    return calibrated_RandomForest, configured_randomForest, calibrated_balancedRandomForest, configured_balancedRandomForest


def configure_SVC(X, y, cv, features) -> [SVC, CalibratedClassifierCV]:
    """returns a configured and calibrated classifier using scoring as metric for calibrating.

           returns a configured classifier that is configured by scoring list and a calibrated classifier that uses the
           CalibratedClassifierCV and the scoring metric.

            :param X: dataframe
                x as returned by the `column_encoding`.
            :param y: dataframe
                y as returned by the `prep_data`.
            :param cv: cross-validation object
                cv from 'sklearn.model_selection'

           :returns calibrated_SVC, configured_SVC: a configured and calibrated classifier.

           """
    X, features = ModelController.column_encoding(X, features)

    over = SMOTE(n_jobs=-1, random_state=42)
    under = RandomUnderSampler(random_state=42)
    steps = [('over', over), ('under', under), ('model', SVC(random_state=42))]
    pipeline = Pipeline(steps=steps)

    # check if decision tree has been created if not create and configure
    if os.path.exists('configured_SVC.obj'):
        configured_SVC = DataController.load_pickled_object(name='configured_SVC.obj')
    else:
        SVCClassifier = SVC(random_state=42)

        pipeline.steps.pop()
        pipeline.steps.append(('model', SVCClassifier))

        rfe = ModelController.produce_boruta(X, y)
        X_trans = rfe.transform(X)

        configured_SVC = ModelController.configure_SVC_Model(X_trans, y, pipeline, cv)
        DataController.pickle_object('configured_SVC', configured_SVC)

    # check if calibrated decision tree has been created if not create and configure
    if os.path.exists('calibrated_SVC.obj'):
        calibrated_SVC = DataController.load_pickled_object(name='calibrated_SVC.obj')
    else:
        calibrated = CalibratedClassifierCV(SVC(random_state=42), n_jobs=-1)

        pipeline.steps.pop()
        pipeline.steps.append(('model', calibrated))

        rfe = ModelController.produce_boruta(X, y)
        X_trans = rfe.transform(X)

        calibrated_SVC = ModelController.configure_SVC_Model(X_trans, y, pipeline, cv)
        DataController.pickle_object('calibrated_SVC', calibrated_SVC)

    return calibrated_SVC, configured_SVC


def configure_MLPClassifier(X, y, cv, use_rfe, features) -> [Pipeline, Pipeline]:
    """returns a configured and calibrated classifier using scoring as metric for calibrating.

           returns a configured classifier that is configured by scoring list and a calibrated classifier that uses the
           CalibratedClassifierCV and the scoring metric.

            :param X: dataframe
                x as returned by the `column_encoding`.
            :param y: dataframe
                y as returned by the `prep_data`.
            :param cv: cross-validation object
                cv from 'sklearn.model_selection'
           :returns calibrated_MLPClassifier, configured_MLPClassifier: a configured and calibrated MLPClassifier.

           """
    X, features = ModelController.column_encoding(X, features)
    over = SMOTE(n_jobs=-1, random_state=42)
    under = RandomUnderSampler(random_state=42)
    steps = [('over', over), ('under', under), ('model', MLPClassifier(random_state=42))]
    pipeline = Pipeline(steps=steps)

    # check if decision tree has been created if not create and configure
    if os.path.exists('configured_MLPClassifier.obj') and use_rfe:
        configured_MLPClassifier = DataController.load_pickled_object(name='configured_MLPClassifier.obj')

    elif use_rfe:
        MLPmodel = MLPClassifier(random_state=42)

        pipeline.steps.pop()
        pipeline.steps.append(('model', MLPmodel))

        rfe = ModelController.produce_boruta(X, y)
        X_trans = rfe.transform(X)

        configured_MLPClassifier = ModelController.configure_MLPClassifier_Model(X_trans, y, pipeline, cv)
        DataController.pickle_object('configured_MLPClassifier', configured_MLPClassifier)

    elif os.path.exists('configured_MLPClassifier_no_RFE.obj') and use_rfe is False:
        configured_MLPClassifier = DataController.load_pickled_object(name='configured_MLPClassifier_no_RFE.obj')
    else:
        MLPmodel = MLPClassifier(random_state=42)
        pipeline.steps.pop()
        pipeline.steps.append(('model', MLPmodel))

        configured_MLPClassifier = ModelController.configure_MLPClassifier_Model(X, y, pipeline, cv)
        DataController.pickle_object('configured_MLPClassifier_no_RFE', configured_MLPClassifier)

    return configured_MLPClassifier


# helper

def takeF2(e):
    return e[1].get('f2')


def create_list_of_classifiers_scores(X, y, cv, classifiers, features) -> list:
    """returns a list of a classifier and its F2 score.

           returns a list of each classifier found in the passed list with its F2 score from 'classifier_testing' produced
           using the best F2 scoring rfe for X transformation.

               :param X: daraframe
                x as returned by the `column_encoding`.
               :param y: daraframe
                y as returned by the `prep_data`.
               :param cv: cross-validation object
                cv from 'sklearn.model_selection'
               :param classifiers: list of classifiers objects

           :return classifiers_with_scores: list of classifier and its score

           """
    print('Testing nine classifiers with cross validation - this can take a few minutes')
    classifiers_with_scores = []
    X, features = ModelController.column_encoding(X, features)
    rfe = ModelController.produce_boruta(X, y)
    X_trans = rfe.transform(X)
    counter = 1

    for classifierArray in classifiers:
        if isinstance(classifierArray, Pipeline):
            classifier_score = ModelController.classifier_testing(classifierArray, X_trans, y, cv, to_print=False)
            classifiers_with_scores.append((classifierArray, classifier_score))
            print(f'Testing of classifier {counter} Complete')
            counter = counter + 1
        else:
            for classifier in classifierArray:
                classifier_score = ModelController.classifier_testing(classifier, X_trans, y, cv, to_print=False)
                classifiers_with_scores.append((classifier, classifier_score))
                print(f'Testing of classifier {counter} Complete')
                counter = counter + 1

    classifiers_with_scores.sort(key=takeF2, reverse=True)
    print('Printing out classifiers scores best performing First')
    sleep(10)
    for score in classifiers_with_scores:
        print(f'The Classifier {score[0].steps[2][1]}.')
        print('metric Scores')
        print(f'ROC AUC: {score[1].get("Roc Auc"):.3f}')
        print(f'Precision: {score[1].get("precision"):.3f}')
        print(f'Recall: {score[1].get("recall"):.3f}')
        print(f'F1: {score[1].get("f1"):.3f}')
        print(f'F2: {score[1].get("f2"):.3f}')
        print('')
        sleep(10)



def create_visualizations(X, y, cv, classifiers, features):
    X, features = ModelController.column_encoding(X, features)
    rfe = ModelController.produce_boruta(X, y)
    X_trans = rfe.transform(X)
    name = 'Neural Network'

    """
    train_indx, test_indx = next(
        StratifiedShuffleSplit(train_size=0.8, test_size=0.2, n_splits=10, random_state=42).split(X_trans, y))
    X_train, X_test, y_train, y_test = X_trans[train_indx], X_trans[test_indx], y[train_indx], y[test_indx]

    classifiers[0].fit(X_train, y_train)
    y_pred = classifiers[0].predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()
    """

    # Prec_Recall_Conf_Display(X_trans, y, classifiers, name)

    # Prec_Recall_Conf_Display_four(X_trans, y, classifiers, name)

    # Calibration_Display(X_trans, classifiers, y)

    # Compare_sampling()


def Calibration_Display(X_trans, classifiers, y):

    train_indx, test_indx = next(
        StratifiedShuffleSplit(train_size=0.8, test_size=0.2, n_splits=10, random_state=42).split(X_trans, y))
    X_train, X_test, y_train, y_test = X_trans[train_indx], X_trans[test_indx], y[train_indx], y[test_indx]

    clf_list = [
        (classifiers[0], "Neural Network"),
        # (classifiers[1][0], "Calibrated RF"),
        # (classifiers[1][2], "Calibrated Balanced RF"),
    ]
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)
    colors = plt.cm.get_cmap("Dark2")
    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    for i, (clf, name) in enumerate(clf_list):
        # clf.fit(X_train, y_train)
        display = CalibrationDisplay.from_estimator(
            clf,
            X_trans,
            y,
            n_bins=5,
            strategy='uniform',  # quantile
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
        )
        calibration_displays[name] = display
    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")
    # Add histogram
    """
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
    for i, (_, name) in enumerate(clf_list):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=8,
            label=name,
            color=colors(i),
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")
    """

    plt.tight_layout()
    plt.show()


def Compare_sampling():
    RandomUnderSampler(random_state=42),
    SMOTE(random_state=42),

    X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                               n_redundant=0, n_repeated=0, n_classes=2,
                               n_clusters_per_class=1,
                               weights=[0.01, 0.99],
                               class_sep=0.8, random_state=0)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    samplers = [
        FunctionSampler(),
        RandomUnderSampler(random_state=42),
        SMOTE(random_state=42),
    ]
    for ax, sampler in zip(axs.ravel(), samplers):
        title = "Original dataset" if isinstance(sampler, FunctionSampler) else None
        plot_resampling(X, y, sampler, ax, title=title)
    fig.tight_layout()
    plt.show()


def plot_resampling(X, y, sampler, ax, title=None):
    X_res, y_res = sampler.fit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor="k")
    if title is None:
        title = f"Resampling with {sampler.__class__.__name__}"
    ax.set_title(title)
    sns.despine(ax=ax, offset=10)


def Prec_Recall_Conf_Display(X_trans, y, classifiers, name):
    ax = plt.gca()

    # PrecisionRecallDisplay
    # RocCurveDisplay
    cali_mod = PrecisionRecallDisplay.from_estimator(classifiers[0][0], X_trans, y, name=f'Calibrated {name}')
    mod = PrecisionRecallDisplay.from_estimator(classifiers[0][1], X_trans, y, name=name)
    cali_mod.plot(ax=ax)
    mod.plot(ax=ax)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    cali_mod_con = ConfusionMatrixDisplay.from_estimator(classifiers[0][0], X_trans, y)
    mod_con = ConfusionMatrixDisplay.from_estimator(classifiers[0][1], X_trans, y)
    cali_mod_con.plot(ax=ax1)
    mod_con.plot(ax=ax2)
    plt.show()



def Prec_Recall_Conf_Display_four(X_trans, y, classifiers, name):
    ax = plt.gca()
    # PrecisionRecallDisplay
    # RocCurveDisplay
    cali_mod1 = PrecisionRecallDisplay.from_estimator(classifiers[0][0], X_trans, y, name=f'Calibrated {name}')
    mod1 = PrecisionRecallDisplay.from_estimator(classifiers[0][1], X_trans, y, name=name)
    cali_mod2 = PrecisionRecallDisplay.from_estimator(classifiers[0][2], X_trans, y, name=f'Calibrated balanced {name}')
    mod2 = PrecisionRecallDisplay.from_estimator(classifiers[0][3], X_trans, y, name=f'Balanced {name}')
    cali_mod1.plot(ax=ax)
    mod1.plot(ax=ax)
    cali_mod2.plot(ax=ax)
    mod2.plot(ax=ax)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    cali_mod_con = ConfusionMatrixDisplay.from_estimator(classifiers[0][0], X_trans, y)
    mod_con = ConfusionMatrixDisplay.from_estimator(classifiers[0][1], X_trans, y)
    cali_mod_con.plot(ax=ax1)
    mod_con.plot(ax=ax2)
    plt.show()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    cali_mod_con = ConfusionMatrixDisplay.from_estimator(classifiers[0][2], X_trans, y)
    mod_con = ConfusionMatrixDisplay.from_estimator(classifiers[0][3], X_trans, y)
    cali_mod_con.plot(ax=ax1)
    mod_con.plot(ax=ax2)
    plt.show()


ModellingDriver()
