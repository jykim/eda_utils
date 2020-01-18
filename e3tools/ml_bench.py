import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, cross_val_predict, learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.inspection import plot_partial_dependence

from e3tools.eda_table import EDATable

from collections import OrderedDict, Counter
from imblearn.under_sampling import RandomUnderSampler
from copy import deepcopy
import impyute as impy

def split_feature_labels(tbl, c_label):
    y = tbl[c_label].copy()
    X = tbl.drop([c_label], axis=1).copy()
    return (X, y)


def merge_feature_labels(X, y, c_label):
    X[c_label] = y
    return X


class MLTable(EDATable):
    """This class prepare data for supervised learning
    - Note that the data type for c_label should be binary
    """
    def __init__(self, tbl, c_label, name='Default'):
        super(MLTable, self).__init__(tbl)
        self.name = name
        self.c_label = c_label

    def get_info(self):
        return {
            "tbl_name":self.name,
            "raw_features":len(self.tbl.columns),
            "encoded_features":len(self.tbl_e.columns),
            "train_set":len(self.train),
            "test_set":len(self.test),
            }

    def normalize(self, topk=3, methods={}):
        res = []
        for c in self.cols:
            if self.dtypes[c] in ('category', 'object'):
                bottomk = self.get_topk_vals(c, topk, False)
                res.append(self.tbl[c].replace(bottomk, np.nan))
            else:
                res.append(self.tbl[c].fillna(0))
        self.tbl_n = pd.concat(res, axis=1)
        return self.tbl_n

    def encode(self, cols=None, col_ptn=None, impute=None):
        if col_ptn:
            cols = [c for c in self.cols if re.search(col_ptn, c)]
        elif not cols:
            cols = self.cols
        if hasattr(self, "tbl_n"):
            print("Using normalized table...")
            tbl = self.tbl_n
        else:
            tbl = self.tbl
        res = []
        for c in cols:
            if c == self.c_label:
                res.append(tbl[c])                
            elif self.dtypes[c] == 'category':
                categories = tbl[c].cat.categories
                res.append(tbl[c].astype(str).replace(categories, list(range(len(categories)))).astype(str))
            elif self.dtypes[c] == 'object':
                res.append(pd.get_dummies(tbl[c], prefix=c))
            else:
                res.append(tbl[c])
        res_df = pd.concat(res, axis=1).astype(float)
        if impute=='mean':
            self.tbl_e = pd.DataFrame(impy.mean(res_df.values), columns=res_df.columns)
        elif impute=='knn':
            self.tbl_e = pd.DataFrame(impy.fast_knn(res_df.values, k=3), columns=res_df.columns)
        else: # no imputation
            self.tbl_e = res_df
        return self.tbl_e

    def split(self, test_size=0.2, random_state=None):
        if hasattr(self, "tbl_e"):
            print("Using encoded table...")
            tbl = self.tbl_e
        else:
            tbl = self.tbl
        self.train, self.test = train_test_split(tbl, test_size=test_size, random_state=random_state)
        print('Train Shape:', self.train.shape)
        print('Test Shape:', self.test.shape)

    def balance_train(self, random_state=None):
        if not hasattr(self, "train"):
            print("Split data first...")
            return

        rus = RandomUnderSampler(random_state=random_state)
        self.train = merge_feature_labels(*rus.fit_resample(*split_feature_labels(self.train, self.c_label)), self.c_label)
        print('Train Shape:', self.train.shape)
        return self.train


class MLModel:
    """This class abstract various ML models
    """
    def __init__(self, name, model, params={}):
        self.name = name
        self.model = model
        self.params = params

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y, verbose=False):
        """
        Calculates metrics and display it
        """
        # Getting the predicted values
        ypred = self.model.predict(X)
        ypred_score = self.model.predict_proba(X)
        
        # calculating metrics
        accuracy = accuracy_score(y, ypred)
        roc_auc = roc_auc_score(y, pd.DataFrame(ypred_score)[1])
        confusion = confusion_matrix(y, ypred)
        
        type1_error = confusion[0][1] / confusion[0].sum() # False Positive
        type2_error = confusion[1][0] / confusion[1].sum() # False Negative
        
        if verbose:
            print('\n### -- ' + str(type(self.model)).split('.')[-1][:-2] + ' -- ###')

            print('Confusion Matrix: \n', confusion)
            print('Accuracy: ', accuracy)
            print('ROC-AUC: ', roc_auc)
            
            print('Type 1 error: ', type1_error)
            print('Type 2 error: ', type2_error)

        return {
            "model_name":self.name,
            "accuracy":accuracy, 
            "roc_auc":roc_auc,
            "type1_error":type1_error,
            "type2_error":type2_error
            }

class MLBench:
    """This class performs supervised learning for given set of data
    """
    def __init__(self):
        self.tables = OrderedDict()
        self.models = OrderedDict()
        self.fit_models = OrderedDict()

    def add_table(self, tbl):
        self.tables[tbl.name] = tbl

    def add_model(self, mdl):
        self.models[mdl.name] = mdl

    def train_batch(self):
        for tn, tbl in self.tables.items():
            for mn, mdl in self.models.items():
                self.fit_models[(tn, mn)] = deepcopy(mdl)
                self.fit_models[(tn, mn)].train(*split_feature_labels(tbl.train, tbl.c_label))

    def cross_validate_batch(self, scoring='roc_auc', nfolds=5):
        res = []
        for tn, tbl in self.tables.items():
            for mn, mdl in self.models.items():
                eval_res = tbl.get_info()
                cv_score = cross_val_score(mdl.model, *split_feature_labels(tbl.train, tbl.c_label), scoring=scoring, cv=nfolds)
                eval_res.update({"model_name":mn, ("cv_"+scoring):cv_score.mean()})
                res.append(eval_res)
        return pd.DataFrame(res)

    def evaluate_batch(self):
        res = []
        for tn, tbl in self.tables.items():
            for mn, mdl in self.models.items():
                eval_res = tbl.get_info()
                eval_res.update(self.fit_models[(tn, mn)].evaluate(*split_feature_labels(tbl.test, tbl.c_label)))
                res.append(eval_res)
        return pd.DataFrame(res)

    def plot_partial_dependence(self, feature_set=None):
        for tn, tbl in self.tables.items():
            X = split_feature_labels(tbl.train, tbl.c_label)[0]
            if not feature_set:
                feature_range = range(len(X.columns))
            else:
                feature_range = [i for i,e in enumerate(X.columns) if e in feature_set]
            for i in feature_range:
                fig, axes = plt.subplots(1, len(self.models), figsize=(5*len(self.models), 3), sharey=True)
                for j, (mn, mdl) in enumerate(self.models.items()):
                    axes[j].set_title("PDP for %s \non %s" % (mdl.name, tbl.name))
                    plot_partial_dependence(self.fit_models[(tn, mn)].model, X , [i], ax=axes[j])

    def plot_learning_curve(self, random_state=None):
        for tn, tbl in self.tables.items():
            fig, axes = plt.subplots(1, len(self.models), figsize=(5*len(self.models), 4), sharey=False)
            for j, (mn, mdl) in enumerate(self.models.items()):
                axes[j].set_title("Learning Curve for %s \non %s" % (mdl.name, tbl.name))
                train_sizes = [int(float(len(tbl.train)) * i/10) for i  in range(1, 9)]
                _, train_scores, test_scores = \
                    learning_curve(mdl.model, *split_feature_labels(tbl.train, tbl.c_label), 
                        train_sizes=train_sizes, random_state=random_state)
                train_scores_mean = np.mean(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                axes[j].plot(train_sizes, train_scores_mean, 'o-', color="r",
                             label="Training score")
                axes[j].plot(train_sizes, test_scores_mean, 'o-', color="g",
                             label="Cross-validation score")
                axes[j].legend(loc="best")
