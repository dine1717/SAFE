# Import Libraries
import os
import math
import time
import random
# random.seed(42)

import numpy as np 
import pandas as pd
import seaborn as sns
import ppscore as pps

import itertools
import functools
from itertools import combinations
from functools import reduce
# Scikit learn 
import sklearn
from sklearn import tree
from sklearn import svm
from sklearn.base import clone
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve

# Stats
from scipy import stats

# GBDT
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier




from distutils.version import LooseVersion
if LooseVersion(sklearn.__version__) >= LooseVersion("0.24"):
    # In sklearn version 0.24, forest module changed to be private.
    from sklearn.ensemble._forest import _generate_unsampled_indices
    from sklearn.ensemble import _forest as forest
else:
    # Before sklearn version 0.24, forest was public, supporting this.
    from sklearn.ensemble.forest import _generate_unsampled_indices
    from sklearn.ensemble import forest

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import figure
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
plt.style.use('ggplot')

# Import skope-rules
from skrules import SkopeRules
# Import Rulefit
from rulefit import RuleFit



# Partial Dependecy packages
from pdpbox.pdp_calc_utils import _calc_ice_lines_inter
from pdpbox.pdp import pdp_isolate, PDPInteract
from pdpbox.utils import (_check_model, _check_dataset, _check_percentile_range, _check_feature,
                    _check_grid_type, _check_memory_limit, _make_list,
                    _calc_memory_usage, _get_grids, _get_grid_combos, _check_classes)
from joblib import Parallel, delayed




from IPython.display import Latex
from pandas.api.types import is_numeric_dtype


import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None) 
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth',None)


rs = {'random_state': 42}

import sklearn.metrics as skm

# Classification - Model Pipeline
def modelPipeline(X_train, X_test, y_train, y_test):

    log_reg = LogisticRegression(**rs)
    nb = BernoulliNB()
    knn = KNeighborsClassifier()
    svm = SVC(**rs)
    mlp = MLPClassifier(max_iter=500, **rs)
    dt = DecisionTreeClassifier(**rs)
    et = ExtraTreesClassifier(**rs)
    rf = RandomForestClassifier(**rs)
    xgb = XGBClassifier(**rs, verbosity=0)

    # Voting Classifier
    # Bagging Classifier
    # scorer = make_scorer(f1_score)

    clfs = [
            ('Logistic Regression', log_reg), 
            ('Naive Bayes', nb),
            ('K-Nearest Neighbors', knn), 
            ('SVM', svm), 
            ('MLP', mlp), 
            ('Decision Tree', dt), 
            ('Extra Trees', et), 
            ('Random Forest', rf), 
            ('XGBoost', xgb)
            ]


    pipelines = []

    scores_df = pd.DataFrame(columns=['Model', 'F1_Score', 'Precision', 'Recall', 'Accuracy', 'ROC_AUC'])


    for clf_name, clf in clfs:

        pipeline = Pipeline(steps=[
                                   ('scaler', StandardScaler()),
                                   ('classifier', clf)
                                   ]
                            )
        pipeline.fit(X_train, y_train)


        y_pred = pipeline.predict(X_test)
        # F1-Score
        fscore = skm.f1_score(y_test, y_pred)
        # Precision
        pres = skm.precision_score(y_test, y_pred)
        # Recall
        rcall = skm.recall_score(y_test, y_pred)
        # Accuracy
        accu = skm.accuracy_score(y_test, y_pred)
        # ROC_AUC
        roc_auc = skm.roc_auc_score(y_test, y_pred)


        pipelines.append(pipeline)

        scores_df = scores_df.append({
                                      'Model' : clf_name, 
                                      'F1_Score' : fscore,
                                      'Precision' : pres,
                                      'Recall' : rcall,
                                      'Accuracy' : accu,
                                      'ROC_AUC' : roc_auc
                                      
                                      }, 
                                     ignore_index=True)
        
    return pipelines, scores_df



from IPython.display import display_html
def display_side_by_side(*args):
    """
    Displays dataframe Side-by-Side

    Params
    ===============================
    df


    """
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

def display_side_by_side_cap(dfs:list, captions:list):
    """
    Display tables side by side to save vertical space

    Params
    ===========================================
        dfs        : list of pandas.DataFrame
        captions   : list of table captions
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))

# display_side_by_side([chi_df, p_df, sam], ['Chi-Square', 'p-value', 'PBS_Corr'])


class bcolors:
    """
    Prints colored Text
    """

    ResetAll = "\033[0m"

    Bold       = "\033[1m"
    Dim        = "\033[2m"
    Underlined = "\033[4m"
    Blink      = "\033[5m"
    Reverse    = "\033[7m"
    Hidden     = "\033[8m"

    ResetBold       = "\033[21m"
    ResetDim        = "\033[22m"
    ResetUnderlined = "\033[24m"
    ResetBlink      = "\033[25m"
    ResetReverse    = "\033[27m"
    ResetHidden     = "\033[28m"

    Default      = "\033[39m"
    Black        = "\033[30m"
    Red          = "\033[31m"
    Green        = "\033[32m"
    Yellow       = "\033[33m"
    Blue         = "\033[34m"
    Magenta      = "\033[35m"
    Cyan         = "\033[36m"
    LightGray    = "\033[37m"
    DarkGray     = "\033[90m"
    LightRed     = "\033[91m"
    LightGreen   = "\033[92m"
    LightYellow  = "\033[93m"
    LightBlue    = "\033[94m"
    LightMagenta = "\033[95m"
    LightCyan    = "\033[96m"
    White        = "\033[97m"

    BackgroundDefault      = "\033[49m"
    BackgroundBlack        = "\033[40m"
    BackgroundRed          = "\033[41m"
    BackgroundGreen        = "\033[42m"
    BackgroundYellow       = "\033[43m"
    BackgroundBlue         = "\033[44m"
    BackgroundMagenta      = "\033[45m"
    BackgroundCyan         = "\033[46m"
    BackgroundLightGray    = "\033[47m"
    BackgroundDarkGray     = "\033[100m"
    BackgroundLightRed     = "\033[101m"
    BackgroundLightGreen   = "\033[102m"
    BackgroundLightYellow  = "\033[103m"
    BackgroundLightBlue    = "\033[104m"
    BackgroundLightMagenta = "\033[105m"
    BackgroundLightCyan    = "\033[106m"
    BackgroundWhite        = "\033[107m"


# print(f"{bcolors.LightBlue}Warning: No active frommets remain. Continue?{bcolors.ResetAll}")





def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Finished {} in {} secs".format(repr(func.__name__), round(run_time, 3)))
        return value

    return wrapper


# Code Optimization
# https://ipython-books.github.io/43-profiling-your-code-line-by-line-with-line_profiler/


@timer
def variance(df, var_threshold):

    """
    Drops Low variance columns on the given Threshold

    Params
    ====================================
    df              : Dataframe
    var_threshold   : Variance Threshold(float)

    Returns
    ====================================
    df              : Dataframe dropping low variance columns
    constant_columns: list of columns dropped
    """

    # throws away all features with variance below var_threshold * (1-var_threshold)
    constant_filter = VarianceThreshold(var_threshold * (1-var_threshold))
    constant_filter.fit(df)
    constant_columns = [column for column in df.columns
                        if column not in df.columns[constant_filter.get_support()]]

    df_var = df.drop(columns = list(constant_columns))

    print(f'Columns dropped on Low Variance : {constant_columns}')

    return df_var, constant_columns


def correlation(df, corr_threshold):
    """
    Drops High Correlation columns on the given Threshold

    Params
    ====================================
    df              : Dataframe
    corr_threshold  : Correlation Threshold(float)

    Returns
    ====================================
    df              : Dataframe dropping High Correlation columns
    col_corr        : list of columns dropped
    """
    
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = df.corr()

    del_cols = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= corr_threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)

    
    df_mcr = df.drop(columns = list(col_corr)) # deleting the column from the dataset
    print(f'Columns dropped on Correlation Threshold : {list(col_corr)}')               

    return df_mcr, list(col_corr)

@timer
def minimalPreprocessing(df, 
                         target, 
                         corr_threshold=0.95, 
                         var_threshold=0.95, 
                         drop_tresh = 0.3, 
                        #  cardinality_thresh=30, 
                         drop_cat=False):
    
    """
    Preprocessing

    Params
    ====================================
    df              : Dataframe
    target          : target column name(str)
    corr_threshold  : Correlation Threshold(float)
    var_threshold   : Variance Threshold(float)
    drop_tresh      : Threshold to drop columns having null values(float)
    drop_cat        : Drop Categorical columns(Bool)

    Return
    ==========================================
    df_v            : Cleaned Dataframe
    object_cols_f   : list of object columns
    numeric_cols_f  : list of numeric columns
    """
    
    # Add Encoders for cat

    print(f"{bcolors.LightBlue}=={bcolors.ResetAll}"*40)
    print(f"{bcolors.Magenta}Data Preproccesing Initiated{bcolors.ResetAll}")
    print(f"{bcolors.LightBlue}=={bcolors.ResetAll}"*40)
    print(f"{bcolors.Magenta}Shape of the Dataframe : {df.shape}{bcolors.ResetAll}")

    # remove columns having null values more than 30%
    drop_pct = df.shape[0]*drop_tresh
    df1 = df.dropna(thresh=drop_pct ,how='all', axis=1)
    
    print(f"{bcolors.Blue}Shape of the Dataframe dropping Cols having NaN > 30%: {df1.shape}{bcolors.ResetAll}")
    
    df = df1.copy()

    # Rename columns with "space" to "_"
    df.columns = [i.replace(' ','_') for i in df.columns]
    
    # Separate object & numeric cols
    object_cols = list(df.select_dtypes(exclude=np.number).columns)
    numeric_cols = list(df.select_dtypes(include=np.number).columns)

    # Fill missing values with arbitary number
    d1 = dict.fromkeys(df.select_dtypes(np.number).columns, -999)
    d2 = dict.fromkeys(df.columns.difference(d1.keys()), 'Missing')
    # Merge both dicts
    d = {**d1, **d2}

    df = df.fillna(d)

    # Cardinality check
    # cardinal = [col for col in object_cols if df[col].nunique() <= cardinality_thresh]
    # high_cardinality_cols = list(set(object_cols)-set(cardinal))
    # print(f'{bcolors.Red}High Cardinality colums > 30 : {high_cardinality_cols}{bcolors.ResetAll}')
    

    # Encoding Cat
    if drop_cat:
        data = df.drop(columns = object_cols)

    elif len(object_cols) >= 1:

        df[object_cols].astype(object)

        # One-Hot Encoder
        # data = pd.get_dummies(df, columns = cardinal)

# One-Hot
# Encoding

        # Ordinal Encoder
        enc = OrdinalEncoder()
        df[object_cols] = enc.fit_transform(df[object_cols])

        # df.drop(columns=high_cardinality_cols, inplace=True)
        print(f"{bcolors.Cyan}Columns Encoding : {object_cols}{bcolors.ResetAll}")
        print(f"{bcolors.Cyan}Shape of the Dataframe Encoding Cat : {df.shape}{bcolors.ResetAll}")
        data = df.copy()
        
    else:
        print(f"{bcolors.Cyan}No Object Columns{bcolors.ResetAll}")
        data = df.copy()

    
    # Feature Eliminations
    # Remove highly correlation features
    df_c, corr_feats = correlation(data, corr_threshold)

    # Remove high variance features
    df_v, const_feats = variance(df_c, var_threshold)

    print(f"{bcolors.Yellow}Shape of the Dataframe dropping Corr & Var Threshold : {df_v.shape}{bcolors.ResetAll}")
    print(f"{bcolors.LightBlue}=={bcolors.ResetAll}"*40)

    removed_feats = corr_feats + const_feats


    object_cols_f = list(set(object_cols)-set(removed_feats))
    numeric_cols_f = list(set(numeric_cols)-set(removed_feats))

    # Remove target col
    numeric_cols_f = [x for x in numeric_cols_f if x != target]

    print(object_cols_f)
    print(numeric_cols_f)

    # Value Count & Percentage for object columns
    c = df[object_cols_f].apply(lambda x: x.value_counts(dropna=False)).T.stack().astype(int)
    p = (df[object_cols_f].apply(lambda x: x.value_counts(normalize=True, 
                                                          dropna=False)).T.stack() * 100).round(2)

    cp = pd.concat([c,p], axis=1, keys=['Count', 'Percentage %'])

    display(cp)

    return df_v, object_cols_f, numeric_cols_f



# Variance Inflation Factor (VIF)
def calculateVIF(exogs, data):
    """
    Calculates Variance inflation factor (VIF) is a technique to estimate the 
    severity of multicollinearity among independent variables within the context
    of a regression. It is calculated as the ratio of all the variances in a 
    model with multiple terms, divided by the variance of a model with one term 
    alone.

    $$ V.I.F. = 1 / (1 - R^2). $$

    Params
    ===================================
    exogs         : list of columns(exculding target)
    data          : Dataframe

    Returns
    ====================================
    df_vif        : Dataframe with VIF & Tolerance(1 - R_square)
    """
    # initialize dictionaries
    vif_dict, tolerance_dict = {}, {}

    # form input data for each exogenous variable
    for exog in exogs:
        not_exog = [i for i in exogs if i != exog]
        X, y = data[not_exog], data[exog]

        # extract r-squared from the fit
        r_squared = LinearRegression().fit(X, y).score(X, y)

        # calculate VIF
        vif = 1/(1 - r_squared)
        vif_dict[exog] = vif

        # calculate tolerance
        tolerance = 1 - r_squared
        tolerance_dict[exog] = tolerance

    # return VIF DataFrame
    df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})

    return df_vif


# Point Bi-Serial Correlation
def pbsCorrelation(df, target, numeric_cols):

    """
    Assumption - Target must be a Binary
    Point-Biserial Correlation Coefficient is a correlation measure of the 
    strength of association between a continuous-level variable (ratio or 
    interval data) and a binary variable.

    Params
    =================================
    df               : Dataframe
    target           : target column name(str)
    numeric_cols     : list of numeric columns(Continuous variables)

    Return
    ===================================
    pbsc_df          : A dataframe with features and pbsc coefficients
    """


    pbsc_dict = {}

    for i in numeric_cols:
        pbsc_val = list(stats.pointbiserialr(df[i].values, df[target].values))
        pbsc_dict[i] = pbsc_val
    pbsc_df =pd.DataFrame.from_dict(pbsc_dict, orient="index", columns=["PBS_Corr", "pval_pbs"])
    pbsc_df.reset_index(inplace=True)
    pbsc_df.rename(columns={"index": "x"}, inplace=True)
    pbsc_df.drop("pval_pbs", axis=1, inplace=True)

    # pbsc_df.round(decimals=4)
    return pbsc_df


# ChiSquare
def calculateChisq(df, object_cols, target):

    """
    The chi-squared statistic is a single number that tells you how much 
    difference exists between your observed counts and the counts you would 
    expect if there were no relationship at all in the population

    Params
    ================================
    df              : Dataframe
    object_cols     : list of object columns(Categorical columns)
    target          : target column name(str)

    Return
    ================================
    chi_s           : Dataframe with features & chisquared statistic
    """

    for i in object_cols:
        df[i] = df[i].astype("category")
        df[i] = df[i].cat.codes
    cx = df[object_cols]
    chi_scores = chi2(cx, df[target])

    chi_s = pd.DataFrame(chi_scores).T
    chi_s.rename(columns={0:"Chi-Sq", 1:"chisq_pval"}, inplace=True)
    chi_s["x"] = list(cx.columns)

    chi_s.drop("Chi-Sq", axis=1, inplace=True)

    return chi_s


# get a list of all columns in the dataframe without the target column
# column_list = [x for x in df.columns if x != 'target']

def calculate_tStatistic(df, target, numeric_cols):

    """
    target - Binary

    Calculate the t-test on TWO RELATED samples of scores, a and b.

    This is a two-sided test for the null hypothesis that 2 related or 
    repeated samples have identical average (expected) values.

    Params
    =====================================
    df               : dataframe
    target           : target column name(str)
    numeric_cols     : list of numeric columns

    Return
    =======================================
    results_df       : A dataframe with features and t-statistic p-val
    """

    # create an empty dictionary
    t_test_results = {}
    # loop over column_list and execute code 
    for column in numeric_cols:
        group1 = df.where(df[target] == 0).dropna()[column]
        group2 = df.where(df[target] == 1).dropna()[column]
        # add the output to the dictionary 
        t_test_results[column] = stats.ttest_ind(group1,group2)
    results_df = pd.DataFrame.from_dict(t_test_results,orient='Index')
    results_df.columns = ['t-statistic','t_stat_pval']

    results_df.reset_index(inplace=True)
    results_df.rename(columns = {"index":"x"}, inplace=True)

    results_df.drop('t-statistic', axis=1, inplace=True)

    return results_df

def change_in_odds(X, y):
    """
    Calculates Change in odds(LogisticRegression)

    Params
    ==============================
    X          : features
    y          : target

    Return
    ===============================
    uni_log_df : A dataframe with features and "change_in_odd"
    """

    # Logistic Coefficients
    logreg = LogisticRegression(max_iter=1000)

    uni_logreg = {}

    for i in X.columns:

        # Normalization
        # X_norm = StandardScaler().fit_transform(leads_c[i])
        logreg.fit(X[[i]].values, y.values)
        coef = logreg.coef_
        # print(coef[0][0])

        exp_coef = math.exp(coef[0][0])

        uni_logreg[i] = exp_coef

    uni_log_df = pd.DataFrame(uni_logreg.items(), columns=['x', 'Change_in_Odds(LogReg)'])
    
    return uni_log_df



############################################################################
#                            Univariate Feature Selection
############################################################################

def univariateFS(df, target, numeric_cols, object_cols, target_type="binary"):

    """
                   Univariate Feature Selection
                   ============================
    Calculates ppscore, PBS_Correlation, Change_in_odds, mutual_info_classif,
    t-Statistic, Chi-Square statistic

    Params
    ==================================
    df             : A dataframe
    target         : target column name(str)
    numeric_cols   : list of numeric columns(excluding target)
    object_cols    : list of object columns


    Return
    ================================
    unifs         : A dataframe with uni-variants of FS
    """

    X = df.drop(target, axis=1)
    y = df[target]

    feature_cols = X.columns.to_list()
    # ppscore
    predict_power_score = pps.predictors(df, target)
    ppscore = predict_power_score[["x", "y", "ppscore"]]
    # display(ppscore)

    #                          Correlation against Target
    ############################################################################
    # # Pearson
    # pearson_series = df.corrwith(df[target], method='pearson')
    # pearson_corr = pd.DataFrame({'x':pearson_series.index, 'Pearson_Corr':pearson_series.values})

    # # kendall
    # kendall_series = df.corrwith(df[target], method='kendall')
    # kendall_corr = pd.DataFrame({'x':kendall_series.index, 'Kendall_Corr':kendall_series.values})

    # # spearman
    # spearman_series = df.corrwith(df[target], method='spearman')
    # spearman_corr = pd.DataFrame({'x':spearman_series.index, 'Spearman_Corr':spearman_series.values})

    # pearson_corr = pearson_corr[~pearson_corr['x'].isin([target])]
    # kendall_corr = kendall_corr[~kendall_corr['x'].isin([target])]
    # spearman_corr = spearman_corr[~spearman_corr['x'].isin([target])]
    
    # corr_scores = reduce(lambda x,y: pd.merge(x,y, on='x', how='outer'), [pearson_corr, kendall_corr, spearman_corr])

    # corr_scores["Mean_Abs_Corr"] = (abs(corr_scores["Pearson_Corr"]) +
    #                                 abs(corr_scores["Kendall_Corr"]) + 
    #                                 abs(corr_scores["Spearman_Corr"])) / 3

    # corr_scores.drop(["Pearson_Corr", "Kendall_Corr", "Spearman_Corr"], axis=1, inplace=True)


    # Univariate Logistic Regression Coefficients
    # One vs Target
    uni_log_df = change_in_odds(X, y)


    # display(corr_scores)
    mutual_info = mutual_info_classif(X, y)

    mutual_info_lst = mutual_info.tolist()
    # print(mutual_info_lst)

    # Zip columns & Coeff
    mutual_tup = list(zip(feature_cols, mutual_info_lst))
    # Covert to DataFrame
    mutual_coef = pd.DataFrame(mutual_tup, columns=['x','MutualInfo_Coef'])

    # display(mutual_coef)
    if target_type == "binary":

        pbsc_df = pbsCorrelation(df, target, numeric_cols)
        # display(pbsc_df)
        
        # T test
        tstat = calculate_tStatistic(df, target, numeric_cols)
        # display(tstat)

    # Chi- Square Test
    chisq_df = calculateChisq(df, object_cols, target)
    # display(chisq_df)

    # Combine all---------------------------------------------------------------------------
    unifs = reduce(lambda x,y: pd.merge(x,y, on='x', how='outer'), [ppscore, 
                                                                    pbsc_df,
                                                                    tstat,
                                                                    # corr_scores,
                                                                    chisq_df,
                                                                    uni_log_df,
                                                                    mutual_coef])
    
    # round to 4
    unifs = unifs.round(decimals=4)
    # Fill na
    unifs.fillna("-", inplace=True)

    return unifs



##########################################################################
#                                  Utilities - FeatureDependency
##########################################################################

## rrpart
def importances(model, X_valid, y_valid, features=None, n_samples=5000, sort=True, metric=None, sample_weights = None):
    """
    Compute permutation feature importances for scikit-learn models using
    a validation set.
    Given a Classifier or Regressor in model
    and validation X and y data, return a data frame with columns
    Feature and Importance sorted in reverse order by importance.
    The validation data is needed to compute model performance
    measures (accuracy or R^2). The model is not retrained.
    You can pass in a list with a subset of features interesting to you.
    All unmentioned features will be grouped together into a single meta-feature
    on the graph. You can also pass in a list that has sublists like:
    [['latitude', 'longitude'], 'price', 'bedrooms']. Each string or sublist
    will be permuted together as a feature or meta-feature; the drop in
    overall accuracy of the model is the relative importance.
    The model.score() method is called to measure accuracy drops.
    This version that computes accuracy drops with the validation set
    is much faster than the OOB, cross validation, or drop column
    versions. The OOB version is a less vectorized because it needs to dig
    into the trees to get out of examples. The cross validation and drop column
    versions need to do retraining and are necessarily much slower.
    This function used OOB not validation sets in 1.0.5; switched to faster
    test set version for 1.0.6. (breaking API change)
    :param model: The scikit model fit to training data
    :param X_valid: Data frame with feature vectors of the validation set
    :param y_valid: Series with target variable of validation set
    :param features: The list of features to show in importance graph.
                     These can be strings (column names) or lists of column
                     names. E.g., features = ['bathrooms', ['latitude', 'longitude']].
                     Feature groups can overlap, with features appearing in multiple.
    :param n_samples: How many records of the validation set to use
                      to compute permutation importance. The default is
                      5000, which we arrived at by experiment over a few data sets.
                      As we cannot be sure how all data sets will react,
                      you can pass in whatever sample size you want. Pass in -1
                      to mean entire validation set. Our experiments show that
                      not too many records are needed to get an accurate picture of
                      feature importance.
    :param sort: Whether to sort the resulting importances
    :param metric: Metric in the form of callable(model, X_valid, y_valid, sample_weights) to evaluate for,
                    if not set default's to model.score()
    :param sample_weights: set if a different weighting is required for the validation samples
    return: A data frame with Feature, Importance columns
    SAMPLE CODE
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    X_train, y_train = ..., ...
    X_valid, y_valid = ..., ...
    rf.fit(X_train, y_train)
    imp = importances(rf, X_valid, y_valid)
    """
    def flatten(features):
        all_features = set()
        for sublist in features:
            if isinstance(sublist, str):
                all_features.add(sublist)
            else:
                for item in sublist:
                    all_features.add(item)
        return all_features

    if features is None:
        # each feature in its own group
        features = X_valid.columns.values
    else:
        req_feature_set = flatten(features)
        model_feature_set = set(X_valid.columns.values)
        # any features left over?
        other_feature_set = model_feature_set.difference(req_feature_set)
        if len(other_feature_set) > 0:
            # if leftovers, we need group together as single new feature
            features.append(list(other_feature_set))

    X_valid, y_valid, sample_weights = sample(X_valid, y_valid, n_samples, sample_weights=sample_weights)
    X_valid = X_valid.copy(deep=False)  # we're modifying columns

    if callable(metric):
        baseline = metric(model, X_valid, y_valid, sample_weights)
    else:
        baseline = model.score(X_valid, y_valid, sample_weights)

    imp = []
    for group in features:
        if isinstance(group, str):
            save = X_valid[group].copy()
            X_valid[group] = np.random.permutation(X_valid[group])
            if callable(metric):
                m = metric(model, X_valid, y_valid, sample_weights)
            else:
                m = model.score(X_valid, y_valid, sample_weights)
            X_valid[group] = save
        else:
            save = {}
            for col in group:
                save[col] = X_valid[col].copy()
            for col in group:
                X_valid[col] = np.random.permutation(X_valid[col])

            if callable(metric):
                m = metric(model, X_valid, y_valid, sample_weights)
            else:
                m = model.score(X_valid, y_valid, sample_weights)
            for col in group:
                X_valid[col] = save[col]
        imp.append(baseline - m)

    # Convert and groups/lists into string column names
    labels = []
    for col in features:
        if isinstance(col, list):
            labels.append('\n'.join(col))
        else:
            labels.append(col)

    I = pd.DataFrame(data={'Feature': labels, 'Importance': np.array(imp)})
    I = I.set_index('Feature')
    if sort:
        I = I.sort_values('Importance', ascending=False)
    return I

def sample(X_valid, y_valid, n_samples, sample_weights=None):
    if n_samples < 0: n_samples = len(X_valid)
    n_samples = min(n_samples, len(X_valid))
    if n_samples < len(X_valid):
        ix = np.random.choice(len(X_valid), n_samples)
        X_valid = X_valid.iloc[ix].copy(deep=False)  # shallow copy
        y_valid = y_valid.iloc[ix].copy(deep=False)
        if sample_weights is not None: sample_weights = sample_weights.iloc[ix].copy(deep=False)
    return X_valid, y_valid, sample_weights


def sample(X_valid, y_valid, n_samples, sample_weights=None):
    if n_samples < 0: n_samples = len(X_valid)
    n_samples = min(n_samples, len(X_valid))
    if n_samples < len(X_valid):
        ix = np.random.choice(len(X_valid), n_samples)
        X_valid = X_valid.iloc[ix].copy(deep=False)  # shallow copy
        y_valid = y_valid.iloc[ix].copy(deep=False)
        if sample_weights is not None: sample_weights = sample_weights.iloc[ix].copy(deep=False)
    return X_valid, y_valid, sample_weights

def sample_rows(X, n_samples):
    if n_samples < 0: n_samples = len(X)
    n_samples = min(n_samples, len(X))
    if n_samples < len(X):
        ix = np.random.choice(len(X), n_samples)
        X = X.iloc[ix].copy(deep=False)  # shallow copy
    return X

def oob_regression_r2_score(rf, X_train, y_train):
    """
    Compute out-of-bag (OOB) R^2 for a scikit-learn random forest
    regressor. We learned the guts of scikit's RF from the BSD licensed
    code:
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L702
    """
    X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y = y_train.values if isinstance(y_train, pd.Series) else y_train

    n_samples = len(X)
    predictions = np.zeros(n_samples)
    n_predictions = np.zeros(n_samples)
    for tree in rf.estimators_:
        unsampled_indices = _get_unsampled_indices(tree, n_samples)
        tree_preds = tree.predict(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds
        n_predictions[unsampled_indices] += 1

    if (n_predictions == 0).any():
        warnings.warn("Too few trees; some variables do not have OOB scores.")
        n_predictions[n_predictions == 0] = 1

    predictions /= n_predictions

    oob_score = r2_score(y, predictions)
    return oob_score

def _get_unsampled_indices(tree, n_samples):
    """
    An interface to get unsampled indices regardless of sklearn version.
    """
    if LooseVersion(sklearn.__version__) >= LooseVersion("0.24"):
        # Version 0.24 moved forest package name
        from sklearn.ensemble._forest import _get_n_samples_bootstrap
        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, n_samples)
        return _generate_unsampled_indices(tree.random_state, n_samples, n_samples_bootstrap)
    elif LooseVersion(sklearn.__version__) >= LooseVersion("0.22"):
        # Version 0.22 or newer uses 3 arguments.
        from sklearn.ensemble.forest import _get_n_samples_bootstrap
        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, n_samples)
        return _generate_unsampled_indices(tree.random_state, n_samples, n_samples_bootstrap)
    else:
        # Version 0.21 or older uses only two arguments.
        return _generate_unsampled_indices(tree.random_state, n_samples)


def permutation_importances_raw(rf, X_train, y_train, metric, n_samples=5000):
    """
    Return array of importances from pre-fit rf; metric is function
    that measures accuracy or R^2 or similar. This function
    works for regressors and classifiers.
    """
    X_sample, y_sample, _ = sample(X_train, y_train, n_samples)

    if not hasattr(rf, 'estimators_'):
        rf.fit(X_sample, y_sample)

    baseline = metric(rf, X_sample, y_sample)
    X_train = X_sample.copy(deep=False) # shallow copy
    y_train = y_sample
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = metric(rf, X_train, y_train)
        X_train[col] = save
        drop_in_metric = baseline - m
        imp.append(drop_in_metric)
    return np.array(imp)

def oob_classifier_f1_score(rf, X_train, y_train):
    """
    Compute out-of-bag (OOB) f1 score for a scikit-learn random forest
    classifier. We learned the guts of scikit's RF from the BSD licensed
    code:
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L425
    """
    X = X_train.values
    y = y_train.values

    n_samples = len(X)
    n_classes = len(np.unique(y))
    predictions = np.zeros((n_samples, n_classes))
    for tree in rf.estimators_:
        unsampled_indices = _get_unsampled_indices(tree, n_samples)
        tree_preds = tree.predict_proba(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds

    predicted_class_indexes = np.argmax(predictions, axis=1)
    predicted_classes = [rf.classes_[i] for i in predicted_class_indexes]

    oob_score = f1_score(y, predicted_classes, average='macro')
    return oob_score

def feature_dependence_matrix(X_train,
                              rfrmodel=RandomForestRegressor(n_estimators=50, oob_score=True),
                              rfcmodel=RandomForestClassifier(n_estimators=50, oob_score=True),
                              cat_count=20,
                              zero=0.001,
                              sort_by_dependence=False,
                              n_samples=5000):
    """
    Given training observation independent variables in X_train (a dataframe),
    compute the feature importance using each var as a dependent variable using
    a RandomForestRegressor or RandomForestClassifier. A RandomForestClassifer is 
    used when the number of the unique values for the dependent variable is less or 
    equal to the cat_count arg. We retrain a random forest for each var as target 
    using the others as independent vars. Only numeric columns are considered.
    By default, sample up to 5000 observations to compute feature dependencies.
    If feature importance is less than zero arg, force to 0. Force all negatives to 0.0.
    Clip to 1.0 max. (Some importances could come back > 1.0 because removing that
    feature sends R^2 very negative.)
    :return: a non-symmetric data frame with the dependence matrix where each row is the importance of each var to the row's var used as a model target.
    """
    numeric_cols = [col for col in X_train if is_numeric_dtype(X_train[col])]

    cat_cols = [col for col in numeric_cols if X_train[col].value_counts().count() <= cat_count]
    cat_cols_le = [col for col in cat_cols if X_train[col].dtypes == 'float' ]
    for col in cat_cols_le:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])

    X_train = sample_rows(X_train, n_samples)

    df_dep = pd.DataFrame(index=X_train.columns, columns=['Dependence']+X_train.columns.tolist())
    for i,col in enumerate(numeric_cols):
        X, y = X_train.drop(col, axis=1), X_train[col]
        if col in cat_cols:
            rf = clone(rfcmodel)
            rf.fit(X,y)
            imp = permutation_importances_raw(rf, X, y, oob_classifier_f1_score, n_samples)
        else:
            rf = clone(rfrmodel)
            rf.fit(X,y)
            imp = permutation_importances_raw(rf, X, y, oob_regression_r2_score, n_samples)
        """
        Some RandomForestRegressor importances could come back > 1.0 because removing
        that feature sends R^2 very negative. Clip them at 1.0.  Also, features with 
        negative importance means that taking them out helps predict but we don't care
        about that here. We want to know which features are collinear/predictive. Clip
        at 0.0.
        """
        imp = np.clip(imp, a_min=0.0, a_max=1.0)
        imp[imp<zero] = 0.0
        imp = np.insert(imp, i, 1.0)
        df_dep.iloc[i] = np.insert(imp, 0, rf.oob_score_) # add overall dependence

    if sort_by_dependence:
        return df_dep.sort_values('Dependence', ascending=False)
    return df_dep

############################################
#    # Leaderboard for all categorical Encoding techniques
#########################

################################################################################
#                                  Feature Coeff & Imp
################################################################################

def featureCoefImp(df, target, encode="onehot"):

    # Pending -  encode="onehot"


    # If cat exists
    #-----------------------------------Encode and pass

    """
                Multi-variant Feature Selection
                ===============================

    Calculates Logictic Regression Coeffectients, lasso coefficients,
    random forest feature importances, random forest permutation feature
    importances, Extra-Tree Classifier coefficients, XGB importances

    Params
    =============================
    df           : A dataframe
    target       : target column name(str)



    Return
    ===============================
    feat_scores  : A dataframe with multi variant FS scores
    """


    
    X = df.drop(target, axis=1)
    y = df[target]

    # Normalization
    X_norm = StandardScaler().fit_transform(X)

    # Train-test Split
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.33, 
                                                        random_state = 42)

    # Logistic Coefficients
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_norm, y)
    # The estimated coefficients will all be around 1:
    coef = logreg.coef_
    # np.array to list
    log_coef = coef[0].tolist()
    # Columns to list
    feature_cols = X.columns.to_list()

    # Zip columns & Coeff
    logreg_tup = list(zip(feature_cols, log_coef))
    # Covert to DataFrame
    logreg_coef = pd.DataFrame(logreg_tup, columns=['Feature','LogRegCoeff'])

    #####################################################################
    #             Random Forrest Clssifier Permutation Importances
    #####################################################################

    # Random Forrest Importance
    # https://github.com/parrt/random-forest-importances/blob/5e038f6e93abf1fc52f28907a0e9f1ff8a7fbc3d/src/rfpimp.py#L73
    rfp = RandomForestClassifier(n_estimators=100,
                                min_samples_leaf=5,
                                n_jobs=-1,
                                oob_score=True)
    rfp.fit(X_train, y_train)
    rf_imp = importances(rfp, X_test, y_test, n_samples=-1)
    rf_imp.reset_index(inplace=True)
    rf_imp.rename(columns={'Feature': 'Feature', 'Importance': 'Permutation_Imp'}, inplace=True)




    # Fit the Random Forest Classifier with 100 Decision Trees   
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    rfm = (rf.feature_importances_).tolist()

    # Zip columns & Coeff
    rfm_tup = list(zip(feature_cols, rfm))
    # Covert to DataFrame
    rfm_imp = pd.DataFrame(rfm_tup, columns=['Feature','RF_Importance'])

    # Extra Tree Classifier for Feature Selection
    # Building the model 
    extra_tree_forest = ExtraTreesClassifier(n_estimators = 100, 
                                            criterion ='entropy') 
    # Training the model 
    extra_tree_forest.fit(X_norm, y) 

    # Computing the importance of each feature 
    feature_importance = extra_tree_forest.feature_importances_ 

    # Normalizing the individual importances 
    feature_importance_normalized = np.std([tree.feature_importances_ for tree in
                                            extra_tree_forest.estimators_], 
                                            axis = 0) 

    extratree_imp = feature_importance_normalized.tolist()

    tree_tup = list(zip(feature_cols, extratree_imp))

    etc_Feature_Selection = pd.DataFrame(tree_tup, columns=['Feature','ETC_Importance'])

    # XGB- GBDT Importances
    xgb = XGBClassifier()
    xgb.fit(X_norm, y)
    # feature importance
    xgb_imp = (xgb.feature_importances_).tolist()

    xgb_tup = list(zip(feature_cols, xgb_imp))

    xgb_df = pd.DataFrame(xgb_tup, columns=['Feature','Xgb_Importance'])
    
    # Lasso
    lasso = Lasso(max_iter=1000)

    # Perform lasso CV to get the best parameter alpha for regulation
    lassocv = LassoCV(alphas=None, cv=10, max_iter=1000)
    lassocv.fit(X_norm, y.values.ravel())

    # Fit lasso using the best alpha
    lasso.set_params(alpha=lassocv.alpha_)
    lasso.fit(X_norm, y)

    l_coef = (lasso.coef_).tolist()

    # Zip columns & Coeff
    lasso_tup = list(zip(feature_cols, l_coef))
    # Covert to DataFrame
    lasso_coef = pd.DataFrame(lasso_tup, columns=['Feature','LassoCoeff'])

    # Combine all-----------------------------------------------------------------------------
    feat_scores = reduce(lambda x,y: pd.merge(x,y, on='Feature', how='outer'), [logreg_coef,
                                                                                lasso_coef,
                                                                                rf_imp,
                                                                                rfm_imp,
                                                                                etc_Feature_Selection,
                                                                                xgb_df])
    

    # display(feat_scores)
    return feat_scores.round(decimals=4)



############################################################################
#                                   Feature Independence
############################################################################
def featureIndependency(df, target):

    """
                    Feature Independency
                    ====================

    Calculates VIF & Dependency Matrix

    Params
    ==============================
    df           : A dataframe
    target       : target column name(str)

    Return
    =============================
    dep_df       : A dataframe with VIF & NLD score for each feature
    """
    
    X = df.drop(target, axis=1)
    y = df[target]

    # Normalization
    X_norm = StandardScaler().fit_transform(X)

    # Train-test Split
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.33, 
                                                        random_state = 42)

    dep_matrix = feature_dependence_matrix(X_train,
                                           rfrmodel=RandomForestRegressor(n_estimators=50, oob_score=True),
                                           rfcmodel=RandomForestClassifier(n_estimators=50, oob_score=True),
                                           cat_count=20,
                                           zero=0.001,
                                           sort_by_dependence=False,
                                           n_samples=5000)
    dep_matrix.reset_index(inplace=True) 
    depm = dep_matrix[["index","Dependence"]]

    vifdf = calculateVIF(X.columns, X)
    vifdf.reset_index(inplace=True)

    dep_df = reduce(lambda x,y: pd.merge(x,y, on='index', how='outer'), [vifdf,
                                                                         depm])
    dep_df.rename(columns={'index': 'Feature', 'Dependence': 'NLD_Score'}, inplace=True)
    
    return dep_df.round(decimals=4)





def pdp_multi_interact(model, dataset, model_features, features, 
                    num_grid_points=None, grid_types=None, percentile_ranges=None, grid_ranges=None, cust_grid_points=None, 
                    cust_grid_combos=None, use_custom_grid_combos=False,
                    memory_limit=0.5, n_jobs=1, predict_kwds=None, data_transformer=None):

    def _expand_default(x, default, length):
        if x is None:
            return [default] * length
        return x

    def _get_grid_combos(feature_grids, feature_types):
        grids = [list(feature_grid) for feature_grid in feature_grids]
        for i in range(len(feature_types)):
            if feature_types[i] == 'onehot':
                grids[i] = np.eye(len(grids[i])).astype(int).tolist()
        return np.stack(np.meshgrid(*grids), -1).reshape(-1, len(grids))

    if predict_kwds is None:
        predict_kwds = dict()

    nr_feats = len(features)

    # check function inputs
    n_classes, predict = _check_model(model=model)
    _check_dataset(df=dataset)
    _dataset = dataset.copy()

    # prepare the grid
    pdp_isolate_outs = []
    if use_custom_grid_combos:
        grid_combos = cust_grid_combos
        feature_grids = []
        feature_types = []
    else:
        num_grid_points = _expand_default(x=num_grid_points, default=10, length=nr_feats)
        grid_types = _expand_default(x=grid_types, default='percentile', length=nr_feats)
        for i in range(nr_feats):
            _check_grid_type(grid_type=grid_types[i])

        percentile_ranges = _expand_default(x=percentile_ranges, default=None, length=nr_feats)
        for i in range(nr_feats):
            _check_percentile_range(percentile_range=percentile_ranges[i])

        grid_ranges = _expand_default(x=grid_ranges, default=None, length=nr_feats)
        cust_grid_points = _expand_default(x=cust_grid_points, default=None, length=nr_feats)

        _check_memory_limit(memory_limit=memory_limit)

        pdp_isolate_outs = []
        for idx in range(nr_feats):
            pdp_isolate_out = pdp_isolate(
                model=model, dataset=_dataset, model_features=model_features, feature=features[idx],
                num_grid_points=num_grid_points[idx], grid_type=grid_types[idx], percentile_range=percentile_ranges[idx],
                grid_range=grid_ranges[idx], cust_grid_points=cust_grid_points[idx], memory_limit=memory_limit,
                n_jobs=n_jobs, predict_kwds=predict_kwds, data_transformer=data_transformer)
            pdp_isolate_outs.append(pdp_isolate_out)

        if n_classes > 2:
            feature_grids = [pdp_isolate_outs[i][0].feature_grids for i in range(nr_feats)]
            feature_types = [pdp_isolate_outs[i][0].feature_type  for i in range(nr_feats)]
        else:
            feature_grids = [pdp_isolate_outs[i].feature_grids for i in range(nr_feats)]
            feature_types = [pdp_isolate_outs[i].feature_type  for i in range(nr_feats)]

        grid_combos = _get_grid_combos(feature_grids, feature_types)

    feature_list = []
    for i in range(nr_feats):
        feature_list.extend(_make_list(features[i]))

    # Parallel calculate ICE lines
    true_n_jobs = _calc_memory_usage(
        df=_dataset, total_units=len(grid_combos), n_jobs=n_jobs, memory_limit=memory_limit)

    grid_results = Parallel(n_jobs=true_n_jobs)(delayed(_calc_ice_lines_inter)(
        grid_combo, data=_dataset, model=model, model_features=model_features, n_classes=n_classes,
        feature_list=feature_list, predict_kwds=predict_kwds, data_transformer=data_transformer)
                                                for grid_combo in grid_combos)

    ice_lines = pd.concat(grid_results, axis=0).reset_index(drop=True)
    pdp = ice_lines.groupby(feature_list, as_index=False).mean()

    # combine the final results
    pdp_interact_params = {'n_classes': n_classes, 
                        'features': features, 
                        'feature_types': feature_types,
                        'feature_grids': feature_grids}
    if n_classes > 2:
        pdp_interact_out = []
        for n_class in range(n_classes):
            _pdp = pdp[feature_list + ['class_%d_preds' % n_class]].rename(
                columns={'class_%d_preds' % n_class: 'preds'})
            pdp_interact_out.append(
                PDPInteract(which_class=n_class,
                            pdp_isolate_outs=[pdp_isolate_outs[i][n_class] for i in range(nr_feats)],
                            pdp=_pdp, **pdp_interact_params))
    else:
        pdp_interact_out = PDPInteract(
            which_class=None, pdp_isolate_outs=pdp_isolate_outs, pdp=pdp, **pdp_interact_params)

    return pdp_interact_out



# function to calculate F values (i.e. partial dependence values) for (jkl…) and all subsets:

def center(arr): return arr - np.mean(arr)

def compute_f_vals(mdl, X, features, selectedfeatures, num_grid_points=10, use_data_grid=False):
    f_vals = {}
    data_grid = None
    if use_data_grid:
        data_grid = X[selectedfeatures].values
    # Calculate partial dependencies for full feature set
    p_full = pdp_multi_interact(mdl, X, features, selectedfeatures, 
                                num_grid_points=[num_grid_points] * len(selectedfeatures),
                                cust_grid_combos=data_grid,
                                use_custom_grid_combos=use_data_grid)
    f_vals[tuple(selectedfeatures)] = center(p_full.pdp.preds.values)
    grid = p_full.pdp.drop('preds', axis=1)
    # Calculate partial dependencies for [1..SFL-1]
    for n in range(1, len(selectedfeatures)):
        for subsetfeatures in itertools.combinations(selectedfeatures, n):
            if use_data_grid:
                data_grid = X[list(subsetfeatures)].values
            p_partial = pdp_multi_interact(mdl, X, features, subsetfeatures, 
                                        num_grid_points=[num_grid_points] * len(selectedfeatures),
                                        cust_grid_combos=data_grid,
                                        use_custom_grid_combos=use_data_grid)
            p_joined = pd.merge(grid, p_partial.pdp, how='left')
            f_vals[tuple(subsetfeatures)] = center(p_joined.preds.values)
    return f_vals


# the second-order H-measure:

def compute_h_val(f_vals, selectedfeatures):
    denom_els = f_vals[tuple(selectedfeatures)].copy()
    numer_els = f_vals[tuple(selectedfeatures)].copy()
    sign = -1.0
    for n in range(len(selectedfeatures)-1, 0, -1):
        for subfeatures in itertools.combinations(selectedfeatures, n):
            numer_els += sign * f_vals[tuple(subfeatures)]
        sign *= -1.0
    numer = np.sum(numer_els**2)
    denom = np.sum(denom_els**2)
    return math.sqrt(numer/denom) if numer < denom else np.nan


#  first-order H-measure as well:

def compute_h_val_any(f_vals, allfeatures, selectedfeature):
    otherfeatures = list(allfeatures)
    otherfeatures.remove(selectedfeature)
    denom_els = f_vals[tuple(allfeatures)].copy()
    numer_els = denom_els.copy()
    numer_els -= f_vals[(selectedfeature,)]
    numer_els -= f_vals[tuple(otherfeatures)]
    numer = np.sum(numer_els**2)
    denom = np.sum(denom_els**2)
    return math.sqrt(numer/denom) if numer < denom else np.nan




# Unique Combination Pairs for list of elements
def uniqueCombinations(numeric_col):
    l = list(itertools.combinations(numeric_col, 2))
    s = set(l)
    # print('actual', len(l), l)
    return list(s)


def friedmanHstatistic(df, target):

    # Add model Parameter as user input
    # ------------------------------

    X = df.drop(target, axis=1)
    y = df[target]

    # Train-test Split
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.33, 
                                                        random_state = 42)
    # Default model, Next Steps------
    gbc = GradientBoostingClassifier(n_estimators=100, 
                                     learning_rate=0.01, 
                                     max_depth=1, 
                                     random_state=42).fit(X_train, y_train)

    combos = uniqueCombinations(X.columns)

    print(f"Unique Combination pairs      :{len(combos)}")

    hstat = {}

    for i in combos:
        # print(i)
        f_val = compute_f_vals(gbc, 
                               X, 
                               X.columns, 
                               i, # Selected Features
                               num_grid_points=10, 
                               use_data_grid=False)
        # print(f_val)
        h_measure_so = compute_h_val(f_val, i)
        # print(h_measure_so)
        hstat[i] = h_measure_so
        # print(hstat)
        # break
    s1 = pd.Series(hstat)
    s = pd.Series(s1.index.values.tolist())
    h_df = pd.concat([s,s1.reset_index(drop=True)], axis=1)
    h_df.columns = ["Pairs", "h_measure"]

    h_df.fillna(0, inplace=True)
    h_stat_df = h_df.sort_values(ascending=False, by="h_measure").round(decimals=6)

    return h_stat_df





def compute_y_pred_from_query(X, rule):
    score = np.zeros(X.shape[0])
    X = X.reset_index(drop=True)
    score[list(X.query(rule).index)] = 1
    return (score)

def compute_performances_from_y_pred(y_true, y_pred, index_name='default_index'):
    df = pd.DataFrame(data=
        {
            'precision':[sum(y_true * y_pred)/sum(y_pred)],
            'recall':[sum(y_true * y_pred)/sum(y_true)]
        },
        index=[index_name],
        columns=['precision', 'recall']
    )
    return (df)

def compute_train_test_query_performances(X_train, y_train, X_test, y_test, rule):
    
    y_train_pred = compute_y_pred_from_query(X_train, rule)
    y_test_pred = compute_y_pred_from_query(X_test, rule)
    
    performances = None
    performances = pd.concat([
        performances,
        compute_performances_from_y_pred(y_train, y_train_pred, 'train_set')],
        axis=0)
    
    performances = pd.concat([
        performances,
        compute_performances_from_y_pred(y_test, y_test_pred, 'test_set')],
        axis=0)
            
    return (performances)


def plot_roc_precison_recall(y_true, 
                             scores_with_line=[], 
                             scores_with_points=[],
                             labels_with_line=['Gradient Boosting', 'Random Forest', 'Decision Tree'],
                             labels_with_points=['skope-rules']):
    
    gradient = np.linspace(0, 1, 10)
    color_list = [ cm.tab10(x) for x in gradient ]
    # print(color_list)

    # c=color.reshape(1,-1) -reshape or transpose list

    fig, axes = plt.subplots(1, 
                             2, 
                             figsize=(12, 5), 
                             sharex=True, 
                             sharey=True)
    ax = axes[0]
    n_line = 0
    for i_score, score in enumerate(scores_with_line):
        n_line = n_line + 1
        fpr, tpr, _ = roc_curve(y_true, score)
        ax.plot(fpr, 
                tpr, 
                linestyle='-.', 
                linewidth=2,
                c=color_list[i_score], 
                lw=1, 
                label=labels_with_line[i_score])
        
    for i_score, score in enumerate(scores_with_points):
        fpr, tpr, _ = roc_curve(y_true, score)
        ax.scatter(fpr[:-1], 
                   tpr[:-1], 
                   c=color_list[n_line + i_score], 
                   s=30, 
                   label=labels_with_points[i_score])
        
    ax.set_title("ROC", fontsize=18)
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=16)
    ax.legend(loc='lower center', fontsize=8)

    ax = axes[1]
    n_line = 0
    for i_score, score in enumerate(scores_with_line):
        n_line = n_line + 1
        precision, recall, _ = precision_recall_curve(y_true, score)
        ax.step(recall, 
                precision, 
                linestyle='-.', 
                linewidth=2,
                c=color_list[i_score], 
                lw=1, 
                where='post', 
                label=labels_with_line[i_score])
        
    for i_score, score in enumerate(scores_with_points):
        precision, recall, _ = precision_recall_curve(y_true, score)
        ax.scatter(recall, 
                   precision, 
                   c=color_list[n_line + i_score], 
                   s=30, 
                   label=labels_with_points[i_score])
        
    ax.set_title("Precision-Recall", fontsize=18)
    ax.set_xlabel('Recall (True Positive Rate)', fontsize=16)
    ax.set_ylabel('Precision', fontsize=16)
    ax.legend(loc='lower center', fontsize=8)
    plt.show()
    
    


def skopeRuleFit(df, target):

    # Rename columns with "space" to "_"
    df.columns = [i.replace(' ','_') for i in df.columns]

    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.33, 
                                                        random_state = 42)



    # Train a gradient boosting classifier for benchmark
    gradient_boost_clf = GradientBoostingClassifier(random_state=42, n_estimators=30, max_depth = 5)
    gradient_boost_clf.fit(X_train, y_train)

    # Train a random forest classifier for benchmark
    random_forest_clf = RandomForestClassifier(random_state=42, n_estimators=30, max_depth = 5)
    random_forest_clf.fit(X_train, y_train)

    # Train a decision tree classifier for benchmark
    decision_tree_clf = DecisionTreeClassifier(random_state=42, max_depth = 5)
    decision_tree_clf.fit(X_train, y_train)

    feature_names = X_train.columns

    # Train a skope-rules-boosting classifier
    skope_rules_clf = SkopeRules(feature_names=feature_names, random_state=42, n_estimators=30,
                                recall_min=0.05, precision_min=0.9,
                                max_samples=0.7,
                                max_depth_duplication= 4, max_depth = 5)
    skope_rules_clf.fit(X_train, y_train)


    # Compute prediction scores
    gradient_boost_scoring = gradient_boost_clf.predict_proba(X_test)[:, 1]
    random_forest_scoring = random_forest_clf.predict_proba(X_test)[:, 1]
    decision_tree_scoring = decision_tree_clf.predict_proba(X_test)[:, 1]

    skope_rules_scoring = skope_rules_clf.score_top_rules(X_test)
    # print(f"Skope rules Scoring{skope_rules_scoring}")
    # print(f"SkopeRules Classifier{skope_rules_clf}")

    # Get number of target rules created
    print(str(len(skope_rules_clf.rules_)) + ' rules have been built with ' + 'SkopeRules.\n')

    skope_rule_listuple = skope_rules_clf.rules_
    # Print SkopeRules
    for i in skope_rule_listuple:
        print(f"{bcolors.LightBlue}=={bcolors.ResetAll}"*40)
        print(f"{bcolors.Magenta}{i}{bcolors.ResetAll}")

    # Print Train Test Query Performances
    for i in range(len(skope_rule_listuple)):
        print('Rule '+str(i+1)+':')
        display(compute_train_test_query_performances(X_train, 
                                                      y_train,
                                                      X_test, 
                                                      y_test,
                                                      skope_rules_clf.rules_[i][0]
                                                      )
        )
    
    
    # Plot ROC Curve & Precison-Recall
    plot_roc_precison_recall(y_test,
                            scores_with_line = [gradient_boost_scoring, random_forest_scoring, decision_tree_scoring],
                            scores_with_points=[skope_rules_scoring]
                            )




def friedmansRuleFit(df, target):
    """
           Impliments  Friedman's RuleFit
           https://christophm.github.io/interpretable-ml-book/rulefit.html

    
    Params
    ================================
    df          : A dataframe
    target      : target column name(str)

    Returns
    =====================================
    rules       : Rules generated using RuleFit
    """
    

    X = df.drop(target, axis=1)

    features = X.columns

    X = X.to_numpy()
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.33, 
                                                        random_state = 42)
    
    
    

    # RuleFit Regressor Performance Against Other Algorithms
    model_dict = {
        'Gradient Boosted Classifier': GradientBoostingClassifier(),
        'Random Forest Classifier':RandomForestClassifier(),
        'Decision Tree Classifier':DecisionTreeClassifier(),
        'Logistic Regression':LogisticRegression(),
        'RuleFit Classifier':RuleFit(tree_size=4,
                                     sample_fract='default',
                                     max_rules=500,
                                     memory_par=0.01,
                                     tree_generator=None,
                                     rfmode='classify',
                                     lin_trim_quantile=0.025,
                                     lin_standardise=True, 
                                     exp_rand_tree_size=True,
                                     random_state=1) 
        }

    test_results = {}

    for model in model_dict.keys():
        if model == 'RuleFit Classifier':
            model_dict[model].fit(X_train,y_train, feature_names = features)
        else:
            model_dict[model].fit(X_train,y_train)
        f1_sc = f1_score(y_test, model_dict[model].predict(X_test))
        test_results.update({model:np.round(f1_sc,2)})

    display(pd.DataFrame(data = test_results.values(),
             index = test_results.keys(),
             columns = ['F1-Score']).sort_values(by='F1-Score',ascending=False))
    
    # While not the best, we can see that the RuleFit Regressor performs similarly to other algorithms.

    # RuleFit Interpretations
    rf = model_dict['RuleFit Classifier']
    rules = rf.get_rules()

    display(rules[rules.coef!=0].sort_values(by='importance', ascending=False).head(10))

    num_rules_rule = len(rules[rules.type=='rule'])
    num_rules_linear = len(rules[rules.type=='linear'])

    print(f"{bcolors.LightBlue}=={bcolors.ResetAll}"*40)
    print(f"{bcolors.Magenta}Number of Linear Rules Generated         :{num_rules_linear}{bcolors.ResetAll}")
    print(f"{bcolors.Magenta}Number of Rules Generated                :{num_rules_rule}{bcolors.ResetAll}")
    print(f"{bcolors.LightBlue}=={bcolors.ResetAll}"*40)

    return rules


