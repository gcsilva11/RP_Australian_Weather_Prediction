import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import numpy as np
from itertools import product
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.kde import KernelDensity
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report

# Kernel Density classifier class, for Parzen classifier.
class KDEClassifier(BaseEstimator, ClassifierMixin):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
    """Bayesian generative classification based on KDE

    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """

    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self

    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


# Data Preprocessing function, where all the data is treated to be ready for classification
def data_preprocessing(dataset_name, dummies, region, plots=False):
    full_df = pd.read_csv(dataset_name)
    if region != "All":
        states = {
            'New South Wales': {'WaggaWagga', 'Canberra', 'NorahHead', 'Cobar', 'Wollongong', 'Albury', 'BadgerysCreek',
                                'SydneyAirport', 'Moree', 'Newcastle', 'CoffsHarbour', 'NorfolkIsland', 'Penrith',
                                'Williamtown', 'Tuggeranong', 'Sydney', 'MountGinini'},
            'Victoria': {'Ballarat', 'Nhil', 'Dartmoor', 'MelbourneAirport', 'Sale', 'Melbourne', 'Bendigo', 'Mildura',
                         'Watsonia', 'Richmond', 'Portland'},
            'Queensland': {'Brisbane', 'GoldCoast', 'Townsville', 'Cairns'},
            'South Australia': {'MountGambier', 'Adelaide', 'Woomera', 'Nuriootpa'},
            'Western Australia': {'Albany', 'SalmonGums', 'PearceRAAF', 'Perth', 'Witchcliffe', 'Walpole',
                                  'PerthAirport'},
            'Tasmania': {'Hobart', 'Launceston'},
            'Northern Territory': {'Katherine', 'Darwin', 'AliceSprings', 'Uluru'}}
        full_df = full_df[full_df['Location'].isin(states[region])]
    # Removing unnecessary features
    df = full_df.drop('Date', 1)
    df = df.drop('RISK_MM', 1)
    df = df.drop('Location', 1)  # Unnecessary for now
    # Missing Values - Feature missing 20% Values - remove it.
    df = df.loc[:, df.isnull().mean() < .2]
    # Eliminate measurements with at least one missing value too.
    df = df.dropna()

    # Remove target (Rain Tomorrow)
    df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)
    df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
    df_strings = df.select_dtypes(exclude=['number'])

    if plots:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True)
        plt.show()
        df.hist(bins=20, figsize=(20, 15))
        plt.show()

        df_strings.apply(pd.value_counts).plot(kind='bar')
        plt.show()

    target = df['RainTomorrow']
    df = df.drop('RainTomorrow', 1)

    if dummies:
        df = pd.get_dummies(df, columns=df_strings.columns)
    else:
        df = df.drop(df_strings, axis=1)
        cardinals = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        df_strings = df_strings.replace(dict(map(reversed, enumerate(cardinals))))
        df = pd.concat([df, df_strings], axis=1)

    # Normalize (0 and 1)
    min_max_scaler = MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df.loc[:, :] = np_scaled

    return full_df, df, target


# Selection of features, and classification of the model
def feature_redux_and_classify(df, target, selection, reduction, classifier, n_features, n_reduction=None):
    # 1 - Feature selection
    if n_reduction is None:
        n_reduction = n_features
    sequence = []
    if selection == "Kruskal-Wallis":
        # 1.1 - Kruskal
        kruskal_stats = []
        for column in df:
            stats, _ = ss.kruskal(df[column], target)
            kruskal_stats.append((column, stats))
        kruskal_stats.sort(key=lambda x: x[1], reverse=True)
        selected_columns = [kruskal_stats[i][0] for i in range(n_features)]
        df = df[selected_columns]
    elif selection == "ROC":
        # 1.2 - Roc
        roc_values = []
        for column in df:
            est = LogisticRegression(solver='liblinear', class_weight='balanced')
            est.fit(df[column].to_frame(), target)
            roc_values.append((column, roc_auc_score(target, est.predict(df[column].to_frame()))))
        roc_values.sort(key=lambda x: x[1], reverse=True)
        selected_columns = [roc_values[i][0] for i in range(n_features)]
        df = df[selected_columns]
    elif selection == "K-Best":
        # sequence.append(('select_best', SelectKBest(k=n_features, score_func=mutual_info_classif)))
        skb = SelectKBest(k=n_features, score_func=mutual_info_classif)
        df = skb.fit_transform(df, target)
    elif selection == "RFE":
        # RFE
        estimator = LogisticRegression(solver='liblinear', class_weight='balanced')
        rfe = RFE(estimator, n_features)
        df = rfe.fit_transform(df, target)

    # 2 - Dimension reduction
    if reduction == "PCA":
        # 2.1 - PCA
        sequence.append(('PCA', PCA(n_components=n_reduction)))
    elif reduction == "LDA":
        # 2.2 - LDA
        sequence.append(('LDARed', LinearDiscriminantAnalysis()))

    # 3 - Classifiers
    if classifier == "Euclidean":
        # 3.1 - Euclidean
        sequence.append(('Euclidean', NearestCentroid(metric='euclidean', shrink_threshold=None)))
    elif classifier == "Mahalanobis":
        # 3.2 - Mahalanobis
        sequence.append(('Mahalanobis', NearestCentroid(metric='mahalanobis', shrink_threshold=None)))
    elif classifier == "Bayes":
        # Naive Gaussian Bayes
        sequence.append(('Bayes', GaussianNB()))
    elif classifier == "K-Nearest":
        # K-Nearest Neighbors
        sequence.append(('K-Nearest', KNeighborsClassifier(n_neighbors=5)))
    elif classifier == "SVC":
        # SVC
        sequence.append(('SVC', SVC(gamma='auto')))
    elif classifier == "Parzen Window":
        # Parzen (Kernel Density Estimation)
        sequence.append(('Parzen', KDEClassifier(kernel='gaussian', bandwidth=1)))
    else:
        # 3.3 - Fisher LDA
        sequence.append(('LDAClass', LinearDiscriminantAnalysis()))

    pipe = Pipeline(sequence)
    kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state=10)
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}

    cv_results = cross_validate(pipe, df, target, cv=kfold, scoring=scoring)
    return cv_results


# Function called by the GUI to run the script
def run(region, sel, sel_n, red, red_n, clas):
    d = True
    if sel_n == '':
        sel_n = 1
    else:
        sel_n = int(sel_n)
    if red_n == '':
        red_n = 1
    else:
        red_n = int(red_n)

    if red == "LDA" and clas == "Mahalanobis":
        print("LDA and Mahalanobis are incompatible")
        return
    if red_n > sel_n:
        print("Reduction amount can't be bigger than selection")
        return
    original_data, treated_data, target_data = data_preprocessing('weatherAUS.csv', d, region)
    cv_results = feature_redux_and_classify(treated_data, target_data, sel, red, clas, sel_n, red_n)
    for k in list(cv_results.keys()):
        if k.startswith('train'):
            del cv_results[k]
    for a in cv_results:
        cv_results[a] = (cv_results[a].mean(), cv_results[a].std())
    id_str = "{},{},{},{},{},{}".format("Dummies" if d else "Factors", sel, sel_n, red, red_n, clas)
    aux = (id_str, cv_results)
    print(aux)


# Funcition used to go over all possible combinations for testing purposes
if __name__ == '__main__':
    filename = 'weatherAUS.csv'
    selection_methods = [
        "None",
        "Kruskal-Wallis",
        "ROC",
        "K-Best",
        "RFE"
    ]
    reduction_methods = [
        "None",
        "PCA",
        "LDA"
    ]
    classifiers = [
        "Euclidean",
        "Mahalanobis",
        "LDA",
        "Bayes",
        "K-Nearest",
        "SVC",
        "Parzen Window"
    ]

    dimensions = [
        4,
        8,
        12
    ]

    regions = [
        "All",
        "New South Wales",
        "Victoria",
        "Queensland",
        "Western Australia",
        "South Australia",
        "Tasmania",
        "Northern Territory"
    ]

    categoricals = (
        True,
        False
    )
    # Scenario A - General Classifier
    i = 0
    all_results = []
    scores_f1 = []
    scores_acc = []
    for d in categoricals:
        for reg in regions:
            original_data, treated_data, target_data = data_preprocessing(filename, d, reg)
            for x in product(selection_methods, reduction_methods, classifiers, dimensions):
                if x[1] == "LDA" and x[2] == "Mahalanobis":
                    # incompatible
                    continue
                if x[0] == "None" and x[1] != "PCA" and x[3] != 4:
                    # useless iterations because dimensions are irrelevant
                    continue
                if x[1] == "LDA" and x[2] == "LDA":
                    # useless iterations because they are the same as None, LDA
                    continue
                i += 1
                cv_results = feature_redux_and_classify(treated_data, target_data, *x)

                id_str = "{},{},{},{},{}".format("Dummies" if d else "Factors", *x)
                print("Just completed " + str(i) + ": " + id_str)

                for a in cv_results:
                    cv_results[a] = cv_results.mean()
                aux = (id_str, cv_results['test_f1_score'], cv_results)
                print(aux)
                all_results.append(aux)

                #scores_acc.append(cv_results['test_accuracy'])
                #scores_f1.append(cv_results['test_f1_score'])

    """
    fig = plt.figure()
    fig.suptitle('State Accuracy Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(scores_acc)
    #ax.set_xticklabels(['All', 'NSW', 'VIC', 'QNL', 'WAU', 'SAU', 'TAS', 'NRT'])
    ax.set_xticklabels(classifiers)
    plt.show()

    fig = plt.figure()
    fig.suptitle('State F1 Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(scores_f1)
    #ax.set_xticklabels(['All', 'NSW', 'VIC', 'QNL', 'WAU', 'SAU', 'TAS', 'NRT'])
    ax.set_xticklabels(classifiers)
    plt.show()
    """
    all_results.sort(key=lambda x: x[1], reverse=True)
    with open("results.txt", 'w') as f:
        for result in all_results:
            print(result, file=f)
