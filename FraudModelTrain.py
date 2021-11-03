from .ImbalanceInvestigation import ImbalanceInvestigation
from .AuxiliaryFunctions import fn_create_monitor_table
from .SelectVariablesTools import SelectVariablesTools
from tpot import TPOTClassifier


import pandas as pd
import pickle
import os



class TrainModel:

    @staticmethod 
    def fn_train(dataModel:pd.DataFrame,
                 configDict: dict) -> None:
        try:
            vRootPath =  os.path.dirname(__file__)
            vFilePath = os.path.join(vRootPath, 'TrainResult')  
            vModelPath = os.path.join(vFilePath,  'fitted_pipeline.sav')
            vDencodersPath = os.path.join(vFilePath,  'encoders_dictionary.pk')
            vAuclerPath = os.path.join(vFilePath,  'auc_table.csv')
            vThresholdPath = os.path.join(vFilePath,  'threshold_table.csv')
            vVariablesPath = os.path.join(vFilePath,  "variables_table.csv")

            vDataModel = dataModel.set_index(configDict.get("campo_id"))
            selectTools = SelectVariablesTools(decodersPath=vDencodersPath)

            X_train, _, y_train, _ = selectTools.fn_get_final_result(dataModel=vDataModel,
                                            featurePath=vVariablesPath,
                                            objetiveVariable=configDict.get("variable_objetivo"),
                                            numberFeatures=configDict.get("numero_variables"))

            VPipeline = TPOTClassifier(random_state=0,
                                        verbosity=1, 
                                        scoring=configDict.get("tpot_scoring_function"),
                                        config_dict="TPOT light")
            
            VPipeline.fit(X_train, y_train)
            vFittedModel = VPipeline.fitted_pipeline_

            vTransormeddataModel= selectTools.transormedData 
            y = vTransormeddataModel.loc[:, [configDict.get("variable_objetivo")]]
            X = vTransormeddataModel.loc[:, selectTools.vListVaribles]
                
            ImbalanceTools = ImbalanceInvestigation()
            ImbalanceTools.fn_get_probabilities_samplers(X, y, vFittedModel)
            ImbalanceTools.fn_get_metric_tables(aucPath=vAuclerPath,
                                                thresholdPath=vThresholdPath)
            vMetric, vModel = ImbalanceTools.fn_get_best_model()
            with open(vModelPath, 'wb') as fp:
                pickle.dump(vModel, fp)


            vMonitor = fn_create_monitor_table(proyect_name='cmf_fraudes',
                                            model_name='clasificador_fraudes',
                                            metric_name='Recall',
                                            metric_indicator=13,
                                            metric_value=vMetric)
            
            
            return vMonitor
            
        except:
            
                vMonitor = fn_create_monitor_table(proyect_name='cmf_fraudes',
                                        model_name='clasificador_fraudes',
                                        metric_name='Recall',
                                        metric_indicator=13)   
        
    
                return vMonitor
    