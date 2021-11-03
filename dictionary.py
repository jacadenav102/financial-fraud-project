from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle



class DictionaryDecoder:
    __slots__=['vDictDecoders', 'vListColumns']
    def __init__(self):
        self.vDictDecoders = {}

    def fn_train_decoders(self, DataFrame,
                          listColumns = None):
        
        if listColumns is None:
            vListColumnsNames = list(DataFrame.keys())
            self.vListColumns = []
            for i, variable in enumerate(DataFrame.dtypes):
                if str(variable) == 'object':
                    self.vListColumns.append(vListColumnsNames[i])
                else:
                    pass          
              
        else:
            self.vListColumns = listColumns
        
        for column in self.vListColumns:
            vData = DataFrame[column].astype(str).str.strip()
            le = LabelEncoder().fit(vData)
            self.vDictDecoders[column] =  le


    def fn_save_decoder_dictionary(self, dictionaryPath = 'diccionarioDecoders.p'):
        with open(dictionaryPath, 'wb') as fp:
            pickle.dump(self.vDictDecoders, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def fn_load_decoder_dictionary(self,dictionaryPath = 'diccionarioDecoders.p'):
        with open(dictionaryPath, 'rb') as fp:
            self.vDictDecoders = pickle.load(fp)

    def fn_transfron_data(self,DataFrame, listColumns = None):
        vDataFrame = DataFrame.copy()
        if listColumns is None:
            vListColumns = self.vListColumns
        else:
            vListColumns = listColumns
        for column in vListColumns:
            vData = vDataFrame[column].astype(str).str.strip()
            vDataFrame[column] = self.vDictDecoders[column].transform(vData)
        return vDataFrame


    def fn_train_transform(self, DataFrame,
                           listColumns = None):
        
        self.fn_train_decoders(DataFrame, listColumns=listColumns)
        transormedData = self.fn_transfron_data(DataFrame,listColumns=self.vListColumns)
        return transormedData
