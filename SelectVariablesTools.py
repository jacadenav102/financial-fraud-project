from sklearn.feature_selection import mutual_info_classif
from .AuxiliaryFunctions import fn_imputate_data_data_type
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from .dictionary import DictionaryDecoder
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import json

class SelectVariablesTools:

    def __init__(self, 
                 decodersPath:str = None) -> None:
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.transormedData = None,
        self.decodersPath = decodersPath
    
    
    def __select_features(self,
                          X_train: object,
                          y_train: object, 
                          X_test: object):
        fs = SelectKBest(score_func=mutual_info_classif, k='all')
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs


    def __plot_importance_feature(self,
                                  dataFrame: pd.DataFrame,
                                  objetive_colum: str) -> None:
        v = dataFrame[objetive_colum]
        X = dataFrame.drop(objetive_colum, axis=1)

        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=0)
        _, _, vFs = self.__select_features(X_train, y_train, X_test)

        vColumns = list(X_train.keys())
        vContent = zip(vColumns, vFs.scores_)
        vGrafict = pd.DataFrame(vContent, columns=['Variable', 'Score'])
        vGrafict = vGrafict.sort_values('Score', ascending = False).head(8)
        sb.barplot(x=vGrafict['Variable'], y=vGrafict['Score'])
        plt.show()

    def fn_preprocess_data(self,
                           dataModel: pd.DataFrame) -> None:
        dataModel = fn_imputate_data_data_type(dataModel)
        vDictDecoder = DictionaryDecoder()
        self.transormedData = vDictDecoder.fn_train_transform(dataModel)
        self.vDictDecoder = vDictDecoder
        vDictDecoder.fn_save_decoder_dictionary(dictionaryPath=self.decodersPath)
    def fn_split_train_test(self,
                            objetiveVariable: str,
                            balance: bool = False) -> None:

        X = self.transormedData.drop([objetiveVariable], axis=1)
        y = self.transormedData.loc[:,[objetiveVariable]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        if balance:
            X_train, y_train  = TomekLinks().fit_resample(X_train, y_train)
        
        self.X_train, self.X_test, self.y_train, self.y_test =  X_train, X_test, y_train, y_test    
        
    def fn_get_mutual_info_score(self,
                                 numberFeatures: int = 20,
                                 featurePath: str = None, 
                                 imagePath: str = None) -> list:
            
        _, _, fs = self.__select_features(self.X_train, self.y_train, self.X_test)
        vColumns = list(self.X_train.keys())
        vContent = zip(vColumns, fs.scores_)
        vGrafict = pd.DataFrame(vContent, columns=['Variable', 'Score'])
        self.vGrafict = vGrafict.sort_values('Score', ascending = False).head(numberFeatures)
        vListFeatures = self.vGrafict['Variable'].values.tolist()
        if featurePath is not None:
            self.vGrafict.to_csv( path_or_buf=featurePath, index=False)
        if imagePath is not None:
            pngPath = imagePath
            sb.barplot(x=self.grafictP['Variable'], y=self.grafictP['Score'])
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(pngPath)
            plt.cla()
        return vListFeatures

    def fn_make_selection_process(self,
                                  dataModel: pd.DataFrame,
                                  objetiveVariable: str,
                                  balance: bool = False,
                                  numberFeatures: int = 20,
                                  featurePath: str = None,
                                  imagePath: str = None) -> list:
        
        self.fn_preprocess_data(dataModel)
        self.fn_split_train_test(objetiveVariable,
                                 balance)
        
        self.vListVaribles = self.fn_get_mutual_info_score(numberFeatures,
                                      featurePath,
                                      imagePath)
        return self.vListVaribles
        


    def fn_get_final_result(self,
                         dataModel: pd.DataFrame,
                         objetiveVariable: str,
                         balance: bool = True,
                         numberFeatures: int = 20,
                         featurePath: str = None,
                         imagePath: str = None):
        
        
        _ = self.fn_make_selection_process(dataModel=dataModel,
                                    objetiveVariable=objetiveVariable,
                                    balance=balance,
                                    numberFeatures=numberFeatures,
                                    featurePath=featurePath,
                                    imagePath=imagePath)
    
    
        
        X = self.transormedData.loc[:, self.vListVaribles]
        y = self.transormedData[objetiveVariable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        X_train, y_train = TomekLinks().fit_resample(X_train, y_train)
        
        return X_train, X_test, y_train, y_test 