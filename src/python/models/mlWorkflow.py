"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-10-04
Copyright (c) 2020-2027, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 

public functions:
    evaluate( diagnosticBundle : DiagnosticBundle,
             dataSet : str,
             outDirArg : str = None
             ) -> None :
             
    def loadAndSplit(
            realDataPath : str,
            realMetaDataPath : str,
            simulatedDataPath : str,
            simulatedMetaDataPath : str,        
            categories : list[str] 
        ) -> DiagnosticBundle:
        
    def shuffleAndSplist(xDF : pd.DataFrame, 
                     ySeries : pd.Series,
                     seed : int = theMeaningOfLife
                     )  ->  tuple[pd.DataFrame, pd.Series,  # Train sets
                                  pd.DataFrame, pd.Series,  # Validation sets
                                  pd.DataFrame, pd.Series]: # Test sets
    
    trainRandomForest( dataDictArg : dict, ) -> None :
    
    
"""

import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

# import os
import pandas as pd
# import pprint as pp


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import LabelEncoder

from stats.confusionMatrixUtilities import statistics
from typing import Any


# import sys

from archive.dataTypes import DiagnosticBundle, ModelBundle
from archive.archiveUtilities import is_empty, saveFigure #urlify
from data.loadCounts import loadGenesBySamplesData

from plots.plotConfusionMatrix import plotConfusionMatrix
from plots.plotROCCurve import plotROCCurve
from plots.volcanoPlotWithBaseMean import volcanoPlotWithBaseMean
from utilities import isOrderedCategoricalSeries

theMeaningOfLife = 42

###############################################################################
def binaryLoadAndSplit(
        realDataPath : str,
        realMetaDataPath : str,
        simulatedDataPath : str,
        diagnosis : list[str] = ["Colorectal", "Esophagus", "Liver", "Lung", "Stomach"]
    ) -> dict[str, DiagnosticBundle]:
    '''
    TODO 
    
    copied from binary.vs.multclassifier.ipynb. move this to a module and rename
    this is generic nothing specific to binary classifier. assume simulated data
    '''
    retDict = dict()
    for d in diagnosis :
        readCategories = [f'{d} Cancer', 'Healthy donor']
        logger.debug(f'AEDWIP readCategories: {readCategories}')
        
        # simulated Data
        XPath = f'{simulatedDataPath}/{d}/formatSynthetics.py.out/X.csv'
        yPath = f'{simulatedDataPath}/{d}/formatSynthetics.py.out/y.csv'
        logger.debug(f'\n\n\nAEDWIP XPath :\n{XPath}')
        logger.debug(f'\nAEDWIP yPath :\n{yPath}') 
        logger.debug(f'\nAEDWIP realDataPath :\n{realDataPath}')
        logger.debug(f'\nAEDWIP realMetaDataPath :\n{realMetaDataPath}') 
        
        #
        # simulated data categories do not have spaces
        #
        # ROC, sensitivity, specificity, confusion plots expect 
        # the positive condition to be labeled 1, and the negative
        # test results labeled 0.
        # make sure simCategories[0] is the negative diagnostic test class
        #
        diagnosis = f'{d}_Cancer' 
        simCategories = ['Healthy_donor', diagnosis]
        logger.debug(f'AEDWIP simCategories: {simCategories}')
         
        retDb = loadAndSplit(
                    realDataPath,
                    realMetaDataPath,
                    simulatedDataPath=XPath,
                    simulatedMetaDataPath=yPath,        
                    categories=simCategories  
                )
         
        retDb.diagnosis = diagnosis 
         
        retDict[d] = retDb
        
    return retDict

###############################################################################
def evaluate( diagnosticBundle : DiagnosticBundle,
             dataSet : str,
             outDirArg : str = None
             ) -> None :
    '''
    TODO
    
    creates plots
        confusion matix, ROC, volcano, stats, reports  ...
        calls plt.close() to prevent figures from being automatically displayed.
        
        you can save closed figure
        
        to display them (This only works in jupyter notebooks, IPython )
            - from IPython.display import display
            - from plots.plotPanel import make_panel
    
    arguments:
        dataSet : should be one of the items in  ["simTrain", "simVal", "simTest", "real"]
        
        figures can not be saved after they have been closed.
        if outDir is defined :
            will save results to outdir
            modelBuild.resultsDict will paths to the saved objects
        else :
            returns None
    '''    
    validDataSets = ["simTrain", "simVal", "simTest", "real"]
    if not dataSet in validDataSets :
        logger.error(f"invalide dataSet : {dataSet} should be key with value {validDataSets}")
        return
    
    modelBundle = getattr(diagnosticBundle, dataSet)
    
    save = False
    if  not is_empty(outDirArg) :
        save = True
        resultsDict = dict()
        modelBundle.resultsDict = resultsDict

    XDF = modelBundle.X
    model = modelBundle.model
    logger.debug(f'AEDWIP X.shape {XDF.shape}')
    logger.debug(f'AEDWIP model {type(model)}')
    predictions = model.predict( XDF )
    
    #labelEncoder = modelBundle.labelEncoder

    # ySeries value type is an ordered pd.Categorical 
    ySeries = modelBundle.y
    if not isOrderedCategoricalSeries(ySeries):
        errMsg = f'argument ySeries {type(ySeries)} expected pandas series with an ordered categorical dtype'
        logger.error(errMsg)
        raise TypeError(errMsg)
    
    #encodeYSeries = labelEncoder.transform(y)
    encodeYSeries = ySeries.cat.codes

    #
    # plot confusion matrix
    #
    title = f'{diagnosticBundle.diagnosis} {dataSet}'
    # confusionMatrixNP are precentages not counts
    confusionMatrixPrecentageNP, disp = plotConfusionMatrix(ySeries.cat.categories, 
                                            # aedwip encodeYSeries, 
                                            ySeries,
                                            predictions,
                                            title,
                                            displayAsPrecentage=True)
    
    
    # save the confusionMatrix counts 
    # modelBundle.confusionMatrix = confusion_matrix(encodeYSeries, predictions, labels=range(len(labelEncoder.classes_)))
    # modelBundle.confusionMatrix = confusion_matrix(encodeYSeries, predictions, labels=range(len(ySeries.cat.categories)))
    # modelBundle.confusionMatrix = confusion_matrix(ySeries, predictions, labels=range(len(ySeries.cat.categories)))
    modelBundle.confusionMatrix = confusion_matrix(ySeries, predictions, labels=ySeries.cat.categories)
    modelBundle.cmDisplay = disp
    # modelBundle.cmTitle = title
    
    # stats contains, tp, fp, fn, tn, sensitify, and specificity
    # statsDict = statistics(confusionMatrixNP, labelEncoder)
    confusionMatrixNP = confusion_matrix(ySeries, predictions, labels=ySeries.cat.categories)
    statsDict = statistics(confusionMatrixNP, ySeries.cat.categories)
    modelBundle.statsDict = statsDict
    
    if save:
        #fileName = urlify(title)
        prefix = "confusionMatrix"
        cmPlotPath = saveFigure(disp.figure_, title, outDirArg, prefix)
        resultsDict['cmPlotPath'] = cmPlotPath
        resultsDict['stats'] = statsDict


    # prevent plotConfusionMatrix plot from being automatically displayed
    plt.close(disp.figure_) 

    #
    # create statistic report, tp, tn, fp, ....
    #
    report = classification_report(#encodeYSeries, 
                                   ySeries, 
                                   predictions,
                                #    target_names=labelEncoder.classes_,
                                   target_names=ySeries.cat.categories, 
                                   output_dict=True)
    modelBundle.report = report
    if save:
        resultsDict['report'] = report

    #
    # plot ROC Curve
    #
    fig, ax, aucDict = plotROCCurve(
        model=model,
        #aedwip labelEncoder=labelEncoder,
        XDF=XDF ,
        #y=y,
        ySeries=ySeries,
        title=title,
        #binaryPositiveLabel='Healthy_donor'
        )
    
    # modelBundle.rocTite = title
    modelBundle.rocFig = fig
    modelBundle.rocAx = ax 
    plt.close(fig)
    
    if save:
        prefix = "roc"
        rocPlotPath = saveFigure(fig,  title, outDirArg, prefix)
        resultsDict['rocFig'] = rocPlotPath
        
        
    # add roc auc to the stats dict
    for key, auc in aucDict.items():
        classStatDict = statsDict[key] 
        classStatDict['auc'] = auc

    #
    # plot Volcano Plot
    #
    vPath = modelBundle.deseqResultsPath
    if vPath :
        fig = volcanoPlotWithBaseMean(vPath,  title)
        modelBundle.volcanoFig = fig

        if save:
            prefix = "volcano"
            volcanoPlotPath = saveFigure(fig,  title, outDirArg, prefix)
            resultsDict['volcanoFig'] = volcanoPlotPath
            
        plt.close(fig) 

            
    # prevent plotConfusionMatrix plot from being automatically displayed
    #plt.close(fig) 
    

###############################################################################
def loadAndSplit(
        realDataPath : str,
        realMetaDataPath : str,
        simulatedDataPath : str,
        simulatedMetaDataPath : str,        
        categories : list[str] 
    ) -> DiagnosticBundle:
    '''
    TODO : 
    
    use to load binary classifiers data. eg 'Lung_cancer vs Healthy donor' or multiclassifier data
    
    simulated data is split into train/val/test sets
    
    real data is not split
    
    arguments:
        categories:
            ROC, sensitivity, specificity, confusion plots expect 
            the positive condition to be labeled 1, and the negative
            test results labeled 0.
            make sure categories[0] is the negative diagnostic test class
            
    '''
    logger.debug(f'AEDWIP simulatedMetaDataPath:\n{simulatedMetaDataPath}')
    logger.debug(f'AEDWIP check for spaces and order categories : {categories}')
    simulatedXDF, tmpYSeries = loadGenesBySamplesData( categories, simulatedDataPath , simulatedMetaDataPath)
    
    # create a series with categorical values
    # reset() creates a strictly positional index
    # and guaranteed to match the categorical array length.
    tmpY = tmpYSeries.reset_index(drop=True)
    simulatedYSeries = pd.Series(
                             pd.Categorical( tmpY, 
                                   categories, 
                                   ordered=True),
                             index=simulatedXDF.index
                        )
    logger.debug(f'AEDWIP tmpYSeries[0:4] : {tmpYSeries[0:4]}')
    logger.debug(f'AEDWIP simulatedYSeries[0:4] : {simulatedYSeries[0:4]}')

    logger.info( f'{ simulatedYSeries.groupby(simulatedYSeries, observed=False).size() }' )
    
    stratifyByCol = "Classes"
    t = shuffleAndSplist(simulatedXDF, simulatedYSeries, seed=theMeaningOfLife)
    XTrainDF, YTrainSeries, XValidationDF, YValidationSeries, XTestDF, YTestSeries = t

    retMbTrain = ModelBundle(XTrainDF, YTrainSeries)
    retMbVal = ModelBundle(XValidationDF, YValidationSeries)
    retMbTest = ModelBundle(XTestDF, YTestSeries)
    
    #
    # load real. do not split into train/val/test
    # select the features from the training set
    #
    realXDF, realTmpYSeries = loadGenesBySamplesData( categories, 
                                                    realDataPath , 
                                                    realMetaDataPath,
                                                    yCategoryColName = 'diagnosis',
                                                    yRowIdName = 'sample_id'
                                                    )
    # create a series with categorical values
    # reset() creates a strictly positional index
    # and guaranteed to match the categorical array length.
    tmpRealY = realTmpYSeries.reset_index(drop=True)
    realYSeries = pd.Series(
                    pd.Categorical(
                        tmpRealY,
                        categories, 
                        ordered=True),
                        index=realXDF.index
                    )    
    realXDF = realXDF.loc[:, XTrainDF.columns]
    retMbReal = ModelBundle(realXDF, realYSeries)
        
    diagnosis = 'AEDWIP_NOT_SET'
    retDB = DiagnosticBundle(diagnosis, 
                                realDataPath,
                                realMetaDataPath,
                                simulatedDataPath,
                                simulatedMetaDataPath,
                                categories,
                                retMbTrain, retMbVal, retMbTest, retMbReal)

    return retDB

###############################################################################
def shuffleAndSplist(xDF : pd.DataFrame, 
                     ySeries : pd.Series,
                     seed : int = theMeaningOfLife
                     )  ->  tuple[pd.DataFrame, pd.Series,  # Train sets
                                  pd.DataFrame, pd.Series,  # Validation sets
                                  pd.DataFrame, pd.Series]: # Test sets
    '''
    TODO
    '''
    dataSetTuple  = train_test_split(xDF, ySeries, train_size=0.6, 
                        random_state=seed, shuffle=True, stratify=ySeries)
    
    XTrainDF, XTmpDF, YTrainSeries, YTmpSeries = dataSetTuple
    
    logger.info(f'XTrainDF.shape : {XTrainDF.shape}')
    logger.info(f'YTrainSeries.shape : {YTrainSeries.shape}')

    dataSetTuple  = train_test_split(XTmpDF, YTmpSeries, train_size=0.5, 
                        random_state=seed, shuffle=True, 
                        stratify=YTmpSeries)
    
    XValidationDF, XTestDF, YValidationSeries, YTestSeries = dataSetTuple
    
    logger.info(f'XValidationDF.shape : {XValidationDF.shape}')
    logger.info(f'YValidationSeries.shape : {YValidationSeries.shape}')
    
    logger.info(f'XTestDF.shape : {XTestDF.shape}')
    logger.info(f'YTestSeries.shape : {YTestSeries.shape}')
    
    return (XTrainDF, YTrainSeries, XValidationDF, YValidationSeries, XTestDF, YTestSeries)

###############################################################################
def trainRandomForest( dataDictArg : dict, 
                      randomState : int = theMeaningOfLife,
                      **kwargs: Any,
                      ) -> None :
    '''    
    
    for each key in dataDictArg train a a Random Forest Classifiler
        - using the simulated training data set
        - for model bundles simTrain, simVal, simTest, and real
            * modelBundle.model to the trained random forest
            
    2/6/26
        - added kwags:
        - removed n_estimators=1000,from RandomForestClassifier
        pass it using kwags in future
        
    example of use:
    
        trainRandomForest(runConfig, n_estimators=1000, max_depth=10)
        
    '''
    for diagnosis in dataDictArg.keys():
        print(f'{diagnosis}')
        
        trainDiagnosticBundle = dataDictArg[diagnosis]
        
        modelBundle = trainDiagnosticBundle.simTrain
        XDF = modelBundle.X
        ySeries = modelBundle.y
        logger.warning(f'XDF.shape : {XDF.shape} y.shape : {ySeries.shape}')
        
        #labelEncoder = LabelEncoder().fit( y )
        
        rf = RandomForestClassifier( random_state=randomState, **kwargs)
        # trainDict[RF_MODEL_KEY] = rf
        
        # if we train the model using cat.codes
        # model.classes_ == [0,1,2]
        # this make debugging harder
        # 
        # encodedY = ySeries.cat.codes # y is a pd.series the value type is and ordered pd.Categorical, 
        # rf.fit(XDF, encodedY)
        rf.fit(XDF, ySeries)
        
        # set the model and label encoder on all the model bundles
        # this will make it easier to implement evaluate()
        for attributeName in ["simTrain", "simVal", "simTest", "real"]:
            modelBundle = getattr(trainDiagnosticBundle, attributeName)
            modelBundle.model = rf
            # modelBundle.labelEncoder = labelEncoder
            modelBundle.orderedCategories = ySeries.cat.categories

