"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-10-11
Copyright (c) 2020-2023, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 
    configBinaryLoadAndSplit(config : ModelConfig, diagnosis : list[str]) -> None:
    
    configBinaryCalculateSensitivityAtSpecificity(
                modelConfig : ModelConfig,
                specifityTargets : list[float ]= [0.9]
                ) -> pd.DataFrame:
                
    configMultiLoad(runConfig : ModelConfig) -> None

    configMultiCalculateSensitivityAtSpecificity
    
    configConfusionPanel(config : ModelConfig, dataSet : str ) -> None :
    
    configEvaluate( config : ModelConfig) -> None:
    
    configROCPanel(config : ModelConfig, dataSet : str ) -> None :
    
    configTrain(config : ModelConfig, archive : dict=None ) -> None :
    
    configVolcanoPanel(config : ModelConfig ) -> None :


"""
import logging
logger = logging.getLogger(__name__)
from joblib import load as joblibLoad
import json
import os
import numpy as np
import pandas as pd
import shutil
from sklearn.preprocessing import label_binarize

# local imports
from archive.dataTypes import  DiagnosticBundle, ModelConfig
from archive.archiveUtilities import  saveFigure 
from models.mlWorkflow import  binaryLoadAndSplit, evaluate, loadAndSplit, trainRandomForest
from plots.plotConfusionMatrix import plotConfusionMatrixPanel
from plots.plotROCCurve import plotROCPanel
from plots.sensitivity_at_specificity import computeSensitivitiesAtTargets
from plots.volcanoPlotWithBaseMean import plotVolcanoPanel

################################################################################
def configBinaryLoadAndSplit(config : ModelConfig, 
                             diagnosis : list[str],
                             localCachePath : str = None) -> None:
    ''' 
    assume the simulated training data was create using 'binary' generation pipeline
    
    assumes the location of the training data set is
    XPath = f'{simulatedDataPath}/Liver/formatSynthetics.py.out/X.csv'
    
    if localCachePath is not none:
        check if data is in cache, 
        
        else copy to cache
            config.realDataPath
            config.realMetaDataPath
            config.simulatedDataPath
    '''
    logger.info("BEGIN")

    # init with true data source path
    realDataPath      = config.realDataPath
    realMetaDataPath  = config.realMetaDataPath
    simulatedDataPath = config.simulatedDataPath
    
    if localCachePath is not None and localCachePath != "" :
        if os.path.isdir(localCachePath):
            logger.warning(f'loading data from local cache : {localCachePath}')
        else :
            os.makedirs(localCachePath)
            logger.warning(f'copying {realDataPath} to {localCachePath}')
            shutil.copy(realDataPath, localCachePath)
            
            logger.warning(f'copying {realMetaDataPath} to {localCachePath}')
            shutil.copy(realMetaDataPath, localCachePath)

            # copy directory
            logger.warning(f'copying {simulatedDataPath} to {localCachePath}')
            simulatedDirName = os.path.basename(simulatedDataPath)
            dstDir = f'{localCachePath}/{simulatedDirName}'
            shutil.copytree(simulatedDataPath, dstDir)

        #use local cache path
        realDataFileName = os.path.basename(realDataPath)
        realDataPath     = f'{localCachePath}/{realDataFileName}'
        
        realMetaFileName = os.path.basename(realMetaDataPath)
        realMetaDataPath  = f'{localCachePath}/{realMetaFileName}'
        
        simulatedDirName = os.path.basename(simulatedDataPath)
        simulatedDataPath = f'{localCachePath}/{simulatedDirName}'
        
        
    logger.warning(f'loading {realDataPath}')
    logger.warning(f'loading {realMetaDataPath}')
    logger.warning(f'loading {simulatedDataPath}')

    
    d= binaryLoadAndSplit(realDataPath, 
                        realMetaDataPath,
                        simulatedDataPath,
                        diagnosis)
    config.dataDict = d
    
    logger.info("END")

    
    
################################################################################
def configBinaryCalculateSensitivityAtSpecificity(
            modelConfig : ModelConfig,
            specifityTargets : list[float ]= [0.9]
            ) -> pd.DataFrame:
    ''' 
    Sensitivity is the True Positive Rate
    
    Specificity is the True Negative Rate
    '''
    
    retDF = pd.DataFrame()
    for key in modelConfig.dataDict.keys():
        diagnosticBundle = modelConfig.dataDict[key]
        
        realModelBundle = diagnosticBundle.real
        XDF = realModelBundle.X 
        ySeries = realModelBundle.y
        model = realModelBundle.model

        # get the encode labels
        encodedYSeries = ySeries.cat.codes
    
        # we want the probability of a positive test results. ie cancer
        #yScoreNP = model.predict_proba(XDF)[:, 1]
        positiveLabelStr = ySeries.cat.categories[1]
        # model.classes_ is in sort order
        # we need to map series order to model.classes
        posIdx = list(model.classes_).index(positiveLabelStr)
        yScoreNP = model.predict_proba(XDF)[:, posIdx]
        
        # specifity == True Negative Rate
        # specifityTargets = [0.9]
        sDF, points = computeSensitivitiesAtTargets(yTrue=encodedYSeries, yScore=yScoreNP, 
                                                targets=specifityTargets,
                                                interpolate=True)
        

        sDF = _ExtraSensitivityStats(diagnosticBundle, key, sDF)
        
        retDF = pd.concat([retDF, sDF])
        retDF = retDF.reset_index(drop=True)
        
    return retDF

################################################################################
def configMultiLoad(runConfig : ModelConfig) -> None:
    ''' 
    load training data
    splits into train/validate/test data sets
    initializes the runConfig dataDict
    '''
 
    XPath = f"{runConfig.simulatedDataPath}/formatSynthetics.py.out/X.csv"
    yPath = f"{runConfig.simulatedDataPath}/formatSynthetics.py.out/y.csv"
        
    diagnosticBundle = loadAndSplit(
        runConfig.realDataPath,
        runConfig.realMetaDataPath,
        simulatedDataPath=XPath,
        simulatedMetaDataPath=yPath,
        # categories=diagnosis_G,
        categories=runConfig.categories #runConfig_G.categories
    )
    
    diagnosticBundle.diagnosis = "multi"
    dataDict = dict()
    dataDict['multi'] = diagnosticBundle
    runConfig.dataDict = dataDict
    
################################################################################
def configMultiCalculateSensitivityAtSpecificity(
            modelConfig : ModelConfig,
            specifityTargets : list[float ]= [0.9]
            ) -> pd.DataFrame:
    ''' 
    TODO
    
    use real data not simulated data sets
    
    assume all models are multi classifiers
    
    returns a data frame with columns
    ['diagnosis', 'Sensitivity (TPR)', 'Target Specificity', 'Achieved Specificity (TNR)',	'Approx. Threshold','auc']
    '''
    retDF = pd.DataFrame()
    
    for dataKey in modelConfig.dataDict.keys():
        diagnosticBundle = modelConfig.dataDict[dataKey]
        realModelBundler = diagnosticBundle.real
        model = realModelBundler.model
        XDF = realModelBundler.X
        ySeries = realModelBundler.y
        
        #encodedYSeries = ySeries.cat.codes
        yStr = ySeries.to_numpy()
        predictionNP = model.predict_proba( XDF )

        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.label_binarize.html
        # LabelBinarizer is generally for target variables, while OneHotEncoder is more versatile and commonly used for feature encoding.
        # yBinaryNP.shape = numSamples x numCategories
        # predictionNP.shape: (224, 6)
        # yBinaryNP.shape : (224, 6)
        # [[0 1 0 0 0 0]
        #  [0 1 0 0 0 0]
        #  [0 1 0 0 0 0]
        #  ...
        #  [0 0 0 0 0 1]
        #  [0 0 0 0 0 1]
        #  [0 0 0 0 0 1]]
        # nClasses= len(ySeries.cat.categories)
        # yBinaryNP = label_binarize(encodedYSeries, classes=np.arange(nClasses))
        mClasses = model.classes_
        yBinaryNP = label_binarize(yStr, classes=mClasses)
        
        # for i in range(nClasses):
        for classIdx, diagnosis in enumerate(mClasses):
            yTrue = yBinaryNP[:, classIdx]
            yScore   = predictionNP[:, classIdx]  

            t = computeSensitivitiesAtTargets(
                                # yTrue=yBinaryNP[:,i], 
                                yTrue,
                                # yScore=predictionNP[:,i],
                                yScore,
                                targets=specifityTargets,
                                interpolate=True
                                )

            senitivityAtSpecificityDF, points = t
            
            #diagnosis = ySeries.cat.categories[i]
            # diagnosis = model.classes_[i]
            senitivityAtSpecificityDF = _ExtraSensitivityStats(diagnosticBundle, diagnosis, senitivityAtSpecificityDF)


            retDF = pd.concat([retDF, senitivityAtSpecificityDF]).reset_index(drop=True)
        
    return retDF

################################################################################
def configConfusionPanel(config : ModelConfig, dataSet : str ) -> None :
    '''
    arguments :
        dataSet : valid values {'simTrain', 'simVal', 'simTest', 'real'}
    '''
    validDataSets = {'simTrain', 'simVal', 'simTest', 'real'} 
    if dataSet not in validDataSets:
        emsg = f'dataSet : {dataSet} not in {validDataSets}'
        logger.error(emsg)
        raise ValueError(emsg)
    
    title = f'{config.dataSetName} {dataSet} data set'
    panelFig = plotConfusionMatrixPanel(config.dataDict, 
                                                dataSet=dataSet, 
                                                title=title) 
    panelFig.suptitle(title)
    imgPath = saveFigure(panelFig, title, config.imgPanelOutDir, prefix="confusionMatrix")
    key = f"{config.dataSetName}_{dataSet}'ConfusionMatrixPanel'"
    config.savePanelDict[key] = imgPath

################################################################################
def configEvaluate( config : ModelConfig) -> None:
    ''' 
    TODO
    '''
    # calculate peformance on all data sets and all cancer types
    for  dataSet in ["simTrain", "simVal", "simTest", "real"]:
        for diganosis, diagnosticBundle in config.dataDict.items():
            print(f'diganosis : {diganosis} dataSet : {dataSet}')
            evaluate(diagnosticBundle, dataSet=dataSet, outDirArg=config.imgOutDir)

################################################################################   
def configROCPanel(config : ModelConfig, dataSet : str ) -> None :
    '''
    arguments :
        dataSet : valid values {'simTrain', 'simVal', 'simTest', 'real'}
    '''
    validDataSets = {'simTrain', 'simVal', 'simTest', 'real'} 
    if dataSet not in validDataSets:
        emsg = f'dataSet : {dataSet} not in {validDataSets}'
        logger.error(emsg)
        raise ValueError(emsg)
    
    title = f'{config.dataSetName} {dataSet} data set'
    panelFig =  plotROCPanel(config.dataDict, 
                                                dataSet=dataSet, 
                                                title=title) 
    panelFig.suptitle(title)
    imgPath = saveFigure(panelFig, title, config.imgPanelOutDir, prefix="ROC")
    key = f"{config.dataSetName}_{dataSet}'ROCPanel'"
    config.savePanelDict[key] = imgPath
  
################################################################################
def configVolcanoPanel(config : ModelConfig ) -> None :
    '''
    only the simTrain bundle has volcano plots
    '''    
    dataSet = "simTrain"
    title = f'{config.dataSetName} {dataSet} data set'
    panelFig = plotVolcanoPanel(config.dataDict, 
                                                dataSet=dataSet, 
                                                title=title) 
    panelFig.suptitle(title)
    imgPath = saveFigure(panelFig, title, config.imgPanelOutDir, prefix="volcano")
    key = f"{config.dataSetName}_{dataSet}'VolcanoPanel'"
    config.savePanelDict[key] = imgPath

################################################################################
def configTrain(
        config : ModelConfig,
        archive : dict=None 
        ) -> bool: 
    ''' 
    if archive json file is defined load model from cache
    else runs training
    
    example of archive file path
    src/jupyterNotebooks/binaryClassifiersXIA.out/shrunkBest20/archive.json
    
    returns true if models was trained (you should save models)
    '''
    dataDict = config.dataDict
    
    needToSave = False
    if archive :
        logger.warning('loading previously trained model')
        dbBundlePaths = archive['diagnosticBundles']
        
        # loadData() create the diagnosticBundles and initialized the ModelBundles
        for key, dbBundle in dataDict.items():
            diagnosis = dbBundle.diagnosis
            
            # we know the diagnosis will be part of the path
            dbBundlePath = None
            for tmpPath in dbBundlePaths:
                if diagnosis in tmpPath:
                    dbBundlePath = tmpPath
                    break
                
            dbBundle = _loadModelInToDiagnosticBundle(dbBundle, dbBundlePath)
        
        
    else :
        logger.warning('training models')
        needToSave = True
        trainRandomForest( dataDict )

    return needToSave
    
#########
######### private
#########

################################################################################
def _ExtraSensitivityStats(diagnosticBundle : DiagnosticBundle,
           diagnosis : str,
           sensitivityDF : pd.DataFrame) -> pd.DataFrame:
    '''
    private
    
    adds number of simulated and test samples + diagnosis
    
    
    returns sensitivityDF with extra columns
    '''
    # 
    # size of training 
    #
    nRows = sensitivityDF.shape[0]

    simTrainModelBundle = diagnosticBundle.simTrain
    nTrainingSamples = simTrainModelBundle.X.shape[0]
    sensitivityDF['nSimTrainSamples'] = [nTrainingSamples]*nRows
    
    nTrainFeatures = simTrainModelBundle.X.shape[1]
    sensitivityDF['nSimTrainFeatures'] = [nTrainFeatures]*nRows

    #
    # size of real test set
    #
    realModelBundle = diagnosticBundle.real
    XDF = realModelBundle.X 
    nRealTestSamples = XDF.shape[0]
    sensitivityDF['nRealTestSamples'] = [nRealTestSamples]*nRows

    nRealTestFeatures = XDF.shape[1]
    sensitivityDF['nRealTestFeatures'] = [nRealTestFeatures]*nRows

    # we want 'diagnosis to be the first row
    colOrderedList = list(sensitivityDF.columns)
    sensitivityDF['diagnosis'] = [diagnosis]*nRows
    sensitivityDF = sensitivityDF.loc[:, ['diagnosis'] + colOrderedList]
    
    return sensitivityDF

################################################################################
def _loadModel( modelPath : str ) -> None :
    ''' 
    TODO
    
    save state generated by trainRandomForest() 
    
    arguments:
        outDir : location to save model and labelEncoder in joblib format
            saveDict has path to model and encoder files

    '''
    if not os.path.exists(modelPath) :
        emsg = f'not found {modelPath}'
        logger.error(emsg)
        return None
    
    trainedModel = joblibLoad(modelPath)
    
    return trainedModel

################################################################################
def _loadModelInToDiagnosticBundle(
                        diagnosticeBundle : DiagnosticBundle, 
                         dbPath : str ) -> DiagnosticBundle :
    ''' 
    TODO
    diagnosticeBundle argument will be update
    
    returns the diagnosticeBundle argument. (enables builder pattern)
    
    '''
    retDB = None
    if os.path.exists(dbPath):
        with open(dbPath, 'r') as dbFD:
            dbDict = json.load(dbFD)
            
        #printDictTree(dbDict)
            
        diagnosis = dbDict['diagnosis'] 
        logger.info(f'diagnosis : {diagnosis}')
        
        trainedModelDict = dbDict['trainedModel']
        trainedModelPath = trainedModelDict['modelPath']
        
        trainedModel = _loadModel(trainedModelPath)
        
        orderedCategories = dbDict['trainedModel']['orderedCategories']

        for attributeName in ["simTrain", "simVal", "simTest", "real"]:
            modelBundle = getattr(diagnosticeBundle, attributeName)
            modelBundle.model = trainedModel
            y = modelBundle.y
            
            # sanity check
            # we made an assumption that some version of LoadAndSplit() was called
            # and set model.y It should be the same ordering as the list in the archive
            tEmsg = f'diagnosis y : { y.cat.categories} not equal {orderedCategories}'
            assertEmgs = tEmsg + ' bug discovered 12/12/2025,orderedCategories was added to archive dict as string not a list. '
            
            assert list(y.cat.categories) == orderedCategories , assertEmgs
            
            modelBundle.orderedCategories = orderedCategories
    else :
        emsg = f'missing {dbPath}'
        logger.error(emsg)
        raise FileNotFoundError(emsg)

        
    return retDB
