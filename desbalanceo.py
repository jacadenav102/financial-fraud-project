
from AuxiliaryFunctions import fn_imputate_data_data_type
from ImbalanceInvestigation import ImbalanceInvestigation
from dictionary import DictionaryDecoder
from datetime import datetime
from sparky_bc import Sparky
from pandas.io import pickle

import pickle
import json
import os

def main() -> None:
    now = datetime.now()
    fecha = now.strftime("%m-%d-%Y_%H%M%S")   

    rootPath = os.path.dirname(__file__)
    filePath = os.path.join(rootPath, 'desbalanceo/ejecución_desbalanceo_{}'.format(fecha))

    os.makedirs(filePath)   
    rocPath = os.path.join(filePath, 'roc_curve_{}.png'.format(fecha))
    detPath = os.path.join(filePath, 'det_curve_{}.png'.format(fecha))
    perPath = os.path.join(filePath, 'per_curve_{}.png'.format(fecha))
    reportPath = os.path.join(filePath, 'clasification_report_{}'.format(fecha))
    aucPath = os.path.join(filePath, 'auc_table_{}.csv'.format(fecha))
    thresholdPath = os.path.join(filePath,  'trhehold_table_{}.csv'.format(fecha))
    dropPath = os.path.join(rootPath, 'sinTira.json')
    sqlPath= os.path.join(rootPath, 'consulta1.sql')
    negativePath = os.path.join(filePath,  'negative_prerecall_{}.png'.format(fecha))
    positivePath = os.path.join(filePath,  'positive_prerecall_{}.png'.format(fecha))
    totalPath = os.path.join(filePath,  'total_prerecall_{}.png'.format(fecha))
    confusionPath = os.path.join(filePath,  'confusion_matrix_{}.png'.format(fecha))
    conexion = Sparky(username='jcadena',
                    ask_pwd=False,
                    password='*Ig3r3nc142020*')

    with open(dropPath, 'r') as fp:
        drop_list = json.load(fp)['drop']  
    dataModel = conexion.helper.obtener_dataframe(open(sqlPath).read()).drop(labels=drop_list, axis=1)

    variablePath = r'ejecuciones\ejecución_selecciones_06-10-2021 153736\listVariables_06-10-2021 153736.json'
    with open(variablePath, 'r') as fp:
        variables = json.load(fp)

    dataModel = dataModel.drop_duplicates()
    dataModel = dataModel.set_index('numero_de_radicado_c')
    dataModel = fn_imputate_data_data_type(dataModel)
    vDictDecoder = DictionaryDecoder()
    transormeddataModel = vDictDecoder.fn_train_transform(dataModel)
    y = transormeddataModel.loc[:, ['sospecha_fraude']]
    X = transormeddataModel.loc[:, variables]
    model_path = r'ejecuciones\ejecución_selecciones_06-10-2021 153736\fitted_pipeline_06-10-2021 153736.sav'
    with open(model_path, 'rb') as fp:
        exported_pipeline = pickle.load(fp)

    Imbalancetools = ImbalanceInvestigation()
    Imbalancetools.fn_get_probabilities_samplers(X, y, exported_pipeline)
    Imbalancetools.fn_generate_per_curve(X, y, exported_pipeline, filePath=perPath)
    Imbalancetools.fn_generate_roc_curve(X, y, exported_pipeline, filePath=rocPath)
    Imbalancetools.fn_generate_det_curve(X, y, exported_pipeline, filePath=detPath)
    Imbalancetools.fn_get_clasification_report(X, y, exported_pipeline, filepath=reportPath)
    Imbalancetools.fn_get_metric_tables(aucPath=aucPath, thresholdPath=thresholdPath)
    Imbalancetools.fn_get_precion_recall_plot_best(negativePath=negativePath,
                                                   positivePath=positivePath,
                                                   totalPath=totalPath)
    Imbalancetools.fn_plot_confusion_matrix_best(confusionPath=confusionPath)
if __name__ == '__main__':
    main()
