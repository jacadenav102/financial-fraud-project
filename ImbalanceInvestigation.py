
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import InstanceHardnessThreshold
from .AuxiliaryFunctions import fn_get_precision_recall_plots
from imblearn.under_sampling import EditedNearestNeighbours


from imblearn.under_sampling import RandomUnderSampler
from .AuxiliaryFunctions import fn_get_threshold_table
from imblearn.under_sampling import OneSidedSelection
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import AllKNN
from sklearn.metrics import  roc_auc_score
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from sklearn.metrics import roc_curve
from sklearn.metrics import det_curve
from progress.bar import Bar

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json


class ImbalanceInvestigation:

    def __init__(self) -> None:
        iht = InstanceHardnessThreshold(random_state=0, 
                                    estimator=LogisticRegression(solver='lbfgs', multi_class='auto'))
        self.vDictsampler = {'SMOTEENN': SMOTEENN(random_state=0),
                'SMOTETomek':SMOTETomek(random_state=0),
                'SMOTE': SMOTE(random_state=0),
                'RandomOverSampler': RandomOverSampler(random_state=0),
                'ADASYN': ADASYN(random_state=0), 
                'ClusterCentroids': ClusterCentroids(random_state=0),
                'RandomUnderSampler': RandomUnderSampler(random_state=0), 
                'EditedNearestNeighbours': EditedNearestNeighbours(),
                'AllKNN': AllKNN(), 
                'OneSidedSelection': OneSidedSelection(random_state=0),
                'NeighbourhoodCleaningRule': NeighbourhoodCleaningRule(),
                'InstanceHardnessThreshold':iht}
        
        self.vDictprob = None
        self.vDictTrue = None
        self.vTrainedSamplers = None
        self.vDictroc = None
        self.bestMethod = None
        
    def fn_train_samplers(self,
                          X_data: object,
                          y_data: object) -> None:

        
        self.vTrainedSamplers = {}
        for sampler in self.vDictsampler.keys():
            vSamplerModel =  self.vDictsampler[sampler]
            vSamplerModel.fit(X_data, y_data)
            self.vTrainedSamplers[sampler] = vSamplerModel

    def fn_get_probabilities_samplers(self,
                                      X_data: object,
                                      y_data: object,
                                      estimator: object)->None:
        

        self.vDictprob = {}
        self.vDictTrue = {}
        self.trainedModel = {}
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=123)
        with Bar('Processing', max = len(self.vDictsampler.keys())) as bar:
                for sampler in self.vDictsampler.keys():
                    X_train, y_train  = self.vDictsampler[sampler].fit_resample(X_train, y_train)

                    vModel = estimator
                    vModel.fit(X_train, y_train)
                    y_prob = estimator.predict_proba(X_test)
                    self.trainedModel[sampler] = vModel
                    self.vDictprob[sampler] = y_prob
                    self.vDictTrue[sampler] = y_test
                    print('Resample {} done'.format(sampler))
                    bar.next()
            
            
    def fn_generate_per_curve(self,
                              X_data: object = None,
                              y_data: object = None,
                              estimator: object = None,
                              showPlot: bool = False,
                              filePath: str= 'PER_cureve.png')->None:
        
        if  self.vDictprob is None:
            self.fn_get_probabilities_samplers(X_data,
                                                y_data,
                                                estimator)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        with Bar('Processing', max = len(self.vDictsampler.keys())) as bar:
            for sampler in self.vDictprob.keys():
                y_prob = self.vDictprob[sampler][:, 1]
                y_true = self.vDictTrue[sampler]
                vPrecision, vRecall, _ = precision_recall_curve(y_true,
                                                                    y_prob)
                plt.plot(vPrecision, vRecall, label=sampler)
                bar.next()
            
        plt.legend()
        plt.title('PER Curve')
        plt.savefig(filePath)
        if showPlot is True:
            plt.show()
        plt.cla()

            
    def fn_return_prediction_threshold(self,
                                        y_prob: np.array,
                                        threshold: float = 0.5) -> list:
        y_pred = []
        for prob in y_prob:
            if prob >= threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)

        return y_pred
   
   
    def fn_generate_roc_curve(self,
                            X_data: object = None,
                            y_data: object = None,
                            estimator: object = None,
                            showPlot: bool = False,
                            filePath: str = 'ROC_cureve.png') -> None:
        
        if  self.vDictprob is None:
            self.fn_get_probabilities_samplers(X_data,
                                               y_data,
                                               estimator)

        self.vDictroc = {}
        plt.xlabel('True Positive Rate')
        plt.ylabel('False Positive Rate')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        with Bar('Processing', max = len(self.vDictsampler.keys())) as bar:    

            for sampler in self.vDictprob.keys():
                y_prob = self.vDictprob[sampler][:, 1]
                y_true = self.vDictTrue[sampler]
                vListFpr, vListTpr, _ = roc_curve(y_true,
                                                y_prob)
                vAuc = roc_auc_score(y_true, y_prob)
                print(vAuc)
                self.vDictroc[sampler] = vAuc
                plt.plot(vListFpr, vListTpr, label='{}: '.format(sampler, vAuc))
                bar.next()

            
        plt.legend()
        plt.title('ROC Curve')
        plt.savefig(filePath)

        if showPlot:
            plt.show()
        plt.cla()
   

    def fn_generate_det_curve(self,
                            X_data: object = None,
                            y_data: object = None,
                            estimator: object = None,
                            showPlot: bool = False,
                            filePath: str = 'DET_cureve.png')->None:
        
        if  self.vDictprob is None:
            self.fn_get_probabilities_samplers(X_data,
                                                y_data,
                                                estimator)

        plt.xlabel('False Negative Rate')
        plt.ylabel('False Positive Rate')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        with Bar('Processing',max = len(self.vDictprob.keys())) as bar:
            
            for sampler in self.vDictprob.keys():
                y_prob = self.vDictprob[sampler][:, 1]
                y_true = self.vDictTrue[sampler]
                vFpr, vFnr, _ = det_curve(y_true,
                                                y_prob)
                plt.plot(vFpr, vFnr, label=sampler)
                bar.next()
        plt.legend()
        plt.title('DET Curve')
        plt.savefig(filePath)

        if showPlot is True:
            plt.show()
        plt.cla()

    def fn_get_clasification_report(self, X_data: object,
                                    y_data: object,
                                    estimator: object = None,
                                    filepath: str = None) -> None:
        if  self.vDictprob is None:
            self.fn_get_probabilities_samplers(X_data,
                                                y_data,
                                                estimator)
        with Bar('Processing', max = len(self.vDictsampler.keys())) as bar:
            for sampler in self.vDictprob:
                y_prob = self.vDictprob[sampler]
                y_pred = self.fn_return_prediction_threshold(y_prob[:,1])
                y_test = self.vDictTrue[sampler]
                vDictReporte = classification_report(y_test, y_pred, output_dict=True)
                file_path = filepath + '_{}'.format(sampler) + '.json'
                with open(file_path, 'w') as fp:
                    json.dump(vDictReporte, fp)
                bar.next()


    def fn_get_metric_tables(self, 
                             aucPath: str,
                             thresholdPath: str)->None:
        
        if self.vDictroc is None: 
            self.fn_calculate_roc_score()
        AucTable = pd.DataFrame( self.vDictroc, index=['AUC']).transpose().sort_values('AUC', ascending=False)
        AucTable.to_csv(path_or_buf= aucPath, index=True,
                sep=';')
        vMaxIndex = np.argmax(AucTable['AUC'])
        self.bestMethod = AucTable.index[vMaxIndex]
        y_prob = self.vDictprob[self.bestMethod][:,1]
        y_test = self.vDictTrue[self.bestMethod]

        fn_get_threshold_table(y_prob=y_prob,
                            y_true= y_test,
                        table_path=thresholdPath)


    def fn_get_precion_recall_plot_best(self,
                                        negativePath: str,
                                        positivePath: str,
                                        totalPath: str)->None:
        fn_get_precision_recall_plots(self.vDictprob[self.bestMethod][:, 1],
                                      self.vDictTrue[self.bestMethod],
                                      negative_path=negativePath,
                                      positive_path=positivePath,
                                      total_path=totalPath)
        
    def fn_plot_confusion_matrix_best(self,
                                      confusionPath:str)-> None :
        y_pred = self.fn_return_prediction_threshold(self.vDictprob[self.bestMethod][:, 1])
        y_true = self.vDictTrue[self.bestMethod]
        vConfusionMat = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(vConfusionMat)
        disp.plot()
        plt.savefig(confusionPath)
        plt.cla()
        
        
    def fn_calculate_roc_score(self,
                               X_data: object = None,
                               y_data: object = None,
                               estimator: object = None) -> None:
        
        
        if  self.vDictprob is None:
            self.fn_get_probabilities_samplers(X_data,
                                                y_data,
                                                estimator)
        self.vDictroc = {}    
        for sampler in self.vDictprob.keys():
            y_prob = self.vDictprob[sampler][:, 1]
            y_true = self.vDictTrue[sampler]
            vAuc = roc_auc_score(y_true, y_prob)
            self.vDictroc[sampler] = vAuc



    def fn_get_best_model(self):

        
        vModel = self.trainedModel.get(self.bestMethod )
        vMetric = self.vDictroc.get(self.bestMethod )
        
        return vMetric, vModel