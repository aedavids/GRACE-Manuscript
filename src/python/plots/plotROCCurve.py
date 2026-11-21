"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-09-08
Copyright (c) 2020-2027, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 
    public functions:
        plotROCCurve()
        plotROCPanel()
        calculateBinary()
        calculateMultiClass()
        
"""

# import array
from itertools import cycle
import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_curve, auc #, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder

# local import
from archive.dataTypes import DiagnosticBundle
from plots.plotPanel import make_panel
from utilities import isOrderedCategoricalSeries

################################################################################
def plotROCCurve(
        model : BaseEstimator,
        #labelEncoder : LabelEncoder,
        XDF : pd.DataFrame, 
        ySeries : pd.Series,
        figsize: tuple[float, float] = (8.0, 6.0),
        title : str = 'Real Data: ROC Curve for Each Class (One-vs-Rest)',
        #binaryPositiveLabel : str = None
    ) -> tuple[plt.figure, plt.axes, dict[str, float]]:
    '''
    TODO:
    
    works for binary or multi classification 
   
    arguments:
        model must implement predict_proba()
        
        XDF:
            dataframe of sample
            
        y :
            the labels for each sample.
            a series with ordered categorical value type
        
        # deprecated binaryPositiveLabel:
        #     default None
        #     if there are only 2 classes specify positive label as a string
            
        figsize:
            (width, height)
            
    returns:
        (fig, ax, aucDict)
    '''
    
    retAUCDict = dict()
    if not isOrderedCategoricalSeries(ySeries):
        errMsg = f'argument ySeries {type(ySeries)} expected pandas series with an ordered categorical dtype'
        logger.error(errMsg)
        raise TypeError(errMsg)

    
    fig, ax = plt.subplots(figsize=figsize)
    
    colorCycle = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])

    # classLabelStr =  labelEncoder.inverse_transform(model.classes_)
    classLabelStr =  ySeries.cat.categories
    
    nClasses = len( model.classes_ )
    # encodedY = ySeries.cat.codes
    
    if nClasses == 2:
        # if binaryPositiveLabel == None:
        #     binaryPositiveLabel =  labelEncoder.classes_[1]
        #     logger.warning(f'binaryPositiveLabel was not defined using default value {binaryPositiveLabel}')
        
        
        # # positiveLabel = labelEncoder.transform( [binaryPositiveLabel] )[0]
        # positiveLabel = 1 # ySeries is ordered categorical dtype, the second class is the positive class
        
        # the classes of the model seems to be in alphabetic order
        # make sure we know which one is the positive class
        positiveLabelStr=ySeries.cat.categories[1]
        positiveLabelInt = 1 # ySeries is ordered categorical dtype, the second class is the positive class


        logger.info(f'positiveLabel  {positiveLabelStr} ')
        # fpr, tpr, threshold, roc_auc = calculateBinary(model, XDF, encodedY, positiveLabelStr)
        fpr, tpr, threshold, roc_auc = calculateBinary(model, XDF, ySeries, positiveLabelStr)
        retAUCDict[positiveLabelStr] = roc_auc 
        ax.plot(fpr, tpr, color="red", lw=2,
                label=f'{positiveLabelStr} (AUC = {roc_auc:0.2f})')
                # label=f'Class {i} (AUC = {roc_auc[i]:0.2f})')        
    else:        
        # fpr, tpr, threshold, roc_auc = calculateMultiClass(model, XDF, encodedY, nClasses)    
        fpr, tpr, threshold, roc_auc = calculateMultiClass(model, XDF, ySeries)    
        for i, color in zip(range(nClasses), colorCycle):
            logger.info(f'\n\nfpr[i] : {fpr[i]}')
            logger.info(f'tpr[i] : {tpr[i]}')
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{classLabelStr[i]} (AUC = {roc_auc[i]:0.2f})')
                    # label=f'Class {i} (AUC = {roc_auc[i]:0.2f})')
            retAUCDict[ classLabelStr[i] ] = roc_auc[i]

    ax.plot([0, 1], [0, 1], 'k--', lw=2)  # diagonal
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")

    return (fig, ax, retAUCDict)

################################################################################
def plotROCPanel(
                dataDict : dict[str,DiagnosticBundle],
                dataSet : str,
                title: str = "") -> plt.Figure:
    """ 
    TODO
    
    arguments
        dataSets should a one of the items in ["simTrain", "simVal", "simTest", "real"]
    """
    validDataSets = ["simTrain", "simVal", "simTest", "real"]
    if not dataSet in validDataSets :
        logger.error(f"invalide dataSet : {dataSet} should be key with value {validDataSets}")
        return

    rocFigs = []
    for diganosis, diagnosticBundle in dataDict.items():
        mb = getattr(diagnosticBundle, dataSet)
        rocFigs.append( mb.rocFig )    
        
    titles = None #[ 'aaaa', 'bbbb', 'Ccccccccccc', 'd', 'TTTTTTTTTTTTT']
    panel = make_panel(rocFigs, ncols=3, mode="uniform", cm_shared_scale=True, hide_colorbars=True, 
                       titles=titles,
                       displayAsPrecentage=True
                       )
    # figTitle = "dataSet"
    panel.suptitle( title )
    plt.show()    
    
    return panel
 
    
################################################################################
def calculateBinary(
            model : BaseEstimator,
            XDF : pd.DataFrame, 
            #encodedY : pd.Series,
            ySeries : pd.Series,
            positiveLabelStr : str,
            # positiveLabelInt : int,
    ) -> tuple[ np.array, np.array, np.array, float]:
    '''
    TODO
    
    positiveLabelStr
        12/9/2025
        we change training() so that random forest is fit using a 
        and orderd categorical series instead of an integer encoding
        
        We made this change to make it easier to debug SHAP
        
    returns :
        tuple of dictionarys fpr, trp, roc_auc
            fpr is false positive rate
            tpr is true positive rate
            threshold
            roc_auc is the receiver operating characteristics 
    '''
    
    if not isOrderedCategoricalSeries(ySeries):
        errMsg = f'argument ySeries {type(ySeries)} expected pandas series with an ordered categorical dtype'
        logger.error(errMsg)
        raise TypeError(errMsg)

    fpr = dict()
    tpr = dict()
    threshold = dict()
    #roc_auc = dict()

    realPredictedProbablity = model.predict_proba(XDF)
    
    # 1) Make y 1-D
    # yTrue = np.asarray(encodedY).ravel()         # from (224, 1) -> (224,)
    yTrue = ySeries.to_numpy()
    
    # 2) Get scores for the positive class only
    # proba = model.predict_proba(X_test)    # shape (224, 2)
    # choose which label is "positive"
    
    #posIdx = np.where(model.classes_ == positiveLabel)[0][0]
    # model.classes_ is in sort order
    # we need to map series ordering to the modle's ordering
    posIdx = list(model.classes_).index(positiveLabelStr)

    y_score = realPredictedProbablity[:, posIdx]            # shape (224,)

    # 3) Compute ROC
    fpr, tpr, threshold = roc_curve(yTrue, y_score, pos_label=positiveLabelStr)
    roc_auc = auc(fpr, tpr)
    
    return (fpr, tpr, threshold, roc_auc)

################################################################################
def calculateMultiClass(
            model : BaseEstimator,
            XDF : pd.DataFrame, 
            ySeries : pd.Series,
            #nClasses : int,
    ) -> tuple[ dict[int, np.array], dict[int, np.array], dict[int, np.array], dict[int, float],]:
    '''
    ROC/AUC requires binary labels, so for multiclass classification each class
    is evaluated one-vs-rest: the target class vs. all other classes combined.
    
    returns :
        tuple of dictionarys fpr, trp, roc_auc
            fpr is false positive rate
            tpr is true positive rate
            threshold
            roc_auc is the receiver operating characteristics 
    '''
    
    if not isOrderedCategoricalSeries(ySeries):
        errMsg = f'argument ySeries {type(ySeries)} expected pandas series with an ordered categorical dtype'
        logger.error(errMsg)
        raise TypeError(errMsg)

    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()

    realPredictedProbablity = model.predict_proba(XDF)
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.label_binarize.html
    # LabelBinarizer is generally for target variables, while OneHotEncoder 
    # is more versatile and commonly used for feature encoding.
    # yBinaryNP.shape = numSamples x numCategories
    # example output
    # predictionNP.shape: (224, 6)
    # yBinaryNP.shape : (224, 6)
    # [[0 1 0 0 0 0]
    #  [0 1 0 0 0 0]
    #  [0 1 0 0 0 0]
    #  ...
    #  [0 0 0 0 0 1]
    #  [0 0 0 0 0 1]
    #  [0 0 0 0 0 1]]
    # yBinary = label_binarize(encodedY, classes=np.arange(nClasses))
    yBinary = label_binarize(ySeries, classes=ySeries.cat.categories)


    # for i in range(nClasses):
    for seriesIdx, className in enumerate(ySeries.cat.categories) :
        # model.classes_ is in sort order
        # we need to map series ordering to the model's ordering
        modelIdx = list(model.classes_).index(className)
        
        # we ignore the threshold return argument
        fpr[modelIdx], tpr[modelIdx],threshold[modelIdx] = roc_curve(yBinary[:, seriesIdx], realPredictedProbablity[:, modelIdx])
        roc_auc[modelIdx] = auc(fpr[modelIdx], tpr[modelIdx])

    return (fpr, tpr, threshold, roc_auc)