from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from datetime import datetime


import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import os

def fn_get_newest_directory(direcotryPath: str)-> str:
    now = datetime.now()
    directoriesDict = {}
    for directories in os.listdir(direcotryPath):
        dateString = directories[-17:]
        dateFormat = datetime.strptime(dateString, "%m-%d-%Y_%H%M%S")
        directoriesDict[dateFormat] = directories
        youngest = max(dt for dt in directoriesDict if dt < now)    
    root = os.path.join(direcotryPath, directoriesDict[youngest])
    return root


def plot_importance_feature(dataframe: pd.DataFrame,
                            objetive_colum: str) -> None:
    y = dataframe[objetive_colum]
    X = dataframe.drop(objetive_colum, axis=1)

    X_train, X_test, y_train, _ = train_test_split(X, y)
    _, _, fs = select_features(X_train, y_train, X_test)

    columns = list(X_train.keys())
    content = zip(columns, fs.scores_)
    grafict = pd.DataFrame(content, columns=['Variable', 'Score'])
    grafict = grafict.sort_values('Score', ascending = False).head(8)
    sb.barplot(x=grafict['Variable'], y=grafict['Score'])
    plt.show()


def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=mutual_info_regression, k='all')
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs



def calculate_sample_mutualInfo(dataFRame: pd.DataFrame,
                                x_axis: str,
                                y_axis: str,
                                col: str)->np.array:
    sample = dataFRame.loc[:, [x_axis, y_axis, col]]
    y = sample[y_axis].values.reshape(-1,1)
    X = sample[x_axis].values.reshape(-1,1)
    return mutual_info_regression(X, y)


def fn_imputate_data_data_type(dataframe: pd.DataFrame,
                            num_imputable: float = -1,
                            cat_imputable: str = 'NA'):
    numerical_columns = list(dataframe.select_dtypes(exclude='O').columns)
    dataframe.loc[:, numerical_columns] = dataframe.loc[:,numerical_columns].fillna(num_imputable)
    categorical_columns = list(dataframe.select_dtypes(include='O').columns)
    dataframe.loc[:, categorical_columns] = dataframe.loc[:,categorical_columns].fillna(cat_imputable)
    return dataframe

                
def fn_get_prediction_threshold(y_prob: np.array,
                                threshold: float = 0.5) -> list :

    classesList = []
    for pr in y_prob:
        if pr >= threshold:
            classesList.append(1)
        else:
            classesList.append(0)
    return classesList




def fn_get_precision_recall_plots(y_prob: np.array,
                                  y_true: np.array, 
                                  negative_path: str,
                                  positive_path: str,
                                  total_path: str) -> None:
    pathDict = {1: positive_path,
                0: negative_path}
    labelDict =  {1: 'Positive',
                0: 'Negative'}
    for label in labelDict:
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob, pos_label=label)
        plt.xlabel('Thresholds')
        plt.ylabel('Precision - Recall')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])

        plt.plot(thresholds, recall[:-1], label='Recall')
        plt.plot(thresholds, precision[:-1], label='Precision')
        plt.title('{} Precision Recall Curve'.format(labelDict[label]))
        plt.legend()
        plt.savefig(pathDict[label])
        plt.close()
        plt.cla()

        totalprecisionList = []
        totalrecallList = []
        for th in thresholds:
            y_pred = fn_get_prediction_threshold(y_prob, threshold=th)
            total_recall = recall_score(y_true, y_pred, average='macro')
            total_precision = precision_score(y_true, y_pred, average='macro')
            totalrecallList.append(total_recall)
            totalprecisionList.append(total_precision)
            
        plt.xlabel('Thresholds')
        plt.ylabel('Precision - Recall')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])

        plt.plot(thresholds, totalrecallList, label='Recall')
        plt.plot(thresholds, totalprecisionList, label='Precision')
        plt.title('Total Precision Recall Curve')
        plt.legend()
        plt.savefig(total_path)
        plt.cla()


def fn_get_threshold_table(y_prob: np.array,
                           y_true: np.array,
                           table_path: str) -> None:


    negativerecallList = []
    positiverecallList = []
    totalrecallList = []
    negativeprecisionList = []
    positiveprecisionList = []
    totalprecisionList = []

    thresholdlist = np.arange(0.0, 0.95, 0.05)
    for th in thresholdlist:
        y_pred = fn_get_prediction_threshold(y_prob, threshold=th)
        negative_recall = recall_score(y_true, y_pred, pos_label=0)
        positive_recall = recall_score(y_true, y_pred, pos_label=1)
        total_recall = recall_score(y_true, y_pred, average='macro')
        negative_precision = precision_score(y_true, y_pred, pos_label=0)
        positive_precision = precision_score(y_true, y_pred, pos_label=1)
        total_precision = precision_score(y_true, y_pred, average='macro')
        
        negativerecallList.append(negative_recall)
        positiverecallList.append(positive_recall)
        totalrecallList.append(total_recall)
        negativeprecisionList.append(negative_precision)
        positiveprecisionList.append(positive_precision)
        totalprecisionList.append(total_precision)



    dictResultados = {'Threshold': thresholdlist,
                    "Negative Recall": negativerecallList,
                    "Positive Recall": positiverecallList,
                    "Total Recall": totalrecallList,
                    "Negative Precision": negativeprecisionList,
                    "Positive Precision": positiveprecisionList,
                    "Total Precision": totalprecisionList
                    }


    precision_recall_table =  pd.DataFrame(dictResultados)
    precision_recall_table.to_csv(table_path,
                                index=False)



def fn_create_monitor_table(proyect_name: str,
                            model_name: str,
                            metric_indicator: int,
                            metric_name: str,
                            metric_value: float = None) -> pd.DataFrame:
    vMonitor = {}
    vMonitor['id_momento'] = datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
    vMonitor['proyecto'] = proyect_name
    vMonitor['modelo'] = model_name
    vMonitor['id_indicador'] = metric_indicator
    vMonitor['nombre_indicador'] = metric_name
    if metric_value is None:
        vMonitor['valor_indicador'] = None
        vMonitor['ejecucion'] = 0
    else:
        vMonitor['valor_indicador'] = metric_value
        vMonitor['ejecucion'] = 1
        
    return pd.DataFrame(vMonitor, index=[1])