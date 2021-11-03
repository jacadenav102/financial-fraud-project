from .AuxiliaryFunctions import fn_get_prediction_threshold
from .AuxiliaryFunctions import  fn_imputate_data_data_type
from .dictionary import DictionaryDecoder

import pandas as pd
import pickle as pk
import os

class PredictModel:

    @staticmethod 
    def fn_predict(vDataModel: pd.DataFrame,
                   configDict: dict) -> pd.DataFrame:
        vRootPath =  os.path.dirname(__file__)
        vFilePath = os.path.join(vRootPath, 'TrainResult')  
        vModelPath = os.path.join(vFilePath,  'fitted_pipeline.sav')
        vDencodersPath = os.path.join(vFilePath,  'encoders_dictionary.pk')
        vDictDecoder = DictionaryDecoder()
        vDictDecoder.fn_load_decoder_dictionary(vDencodersPath)
        vDataModel = vDataModel.drop_duplicates()
        vDataModel = vDataModel.drop(configDict.get("variable_objetivo"), axis=1)
        vDataModel = vDataModel.set_index(configDict.get("campo_id"))
        vDataModel = fn_imputate_data_data_type(vDataModel)
        vTransormedDataModel = vDictDecoder.fn_train_transform(vDataModel)

        with open(vModelPath, 'rb') as fp:
            vModel = pk.load(fp)
            
        vProbad = vModel.predict_proba(vTransormedDataModel)
        vPredictions = fn_get_prediction_threshold(y_prob=vProbad[:,1], 
                                                   threshold=configDict.get("threshold_seleccionado"))
        vFinalTable = pd.DataFrame(vPredictions)
        vFinalTable.loc[:,[configDict.get("campo_id")]] = list(vDataModel.index)
        vFinalTable.loc[:,['prediccion']] = vPredictions
        vFinalTable.loc[:,['Probabilidad No Fraude', 'Probabilidad Fraude']] = vProbad
        return vFinalTable
