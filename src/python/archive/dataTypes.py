"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-10-04
Copyright (c) 2020-2027, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 

see 
    ModelConfig
    ModelBundle 
    DiagnosticBundle
"""

from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.base import BaseEstimator  # base class for all sklearn models
from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.preprocessing import LabelEncoder
from typing import Optional

# from dataclasses import dataclass
# from typing import Optional
from utilities import urlify
# @dataclass
################################################################################
class ModelConfig :
    '''
    captures all the input and configuration parameters for mlWorkflow.py
    
    see configMlWorkflows.py
    '''
    ################################################################################
    def __init__(self, 
                 dataSetName : str,
                 rootOutDir : str,
                 simulatedDataPath : str, 
                 realDataPath : str,
                 realMetaDataPath : str,
                 deseqResultsType : str,
                 #deseqResultsRootPath : Optional[str] = None,
                categories : list[str] = None,
                 ) :
        ''' 
        TODO
        '''
        self.dataSetName = urlify(dataSetName)
        self.rootOutDir = rootOutDir
        self.simulatedDataPath : str = simulatedDataPath
        self.realDataPath : str = realDataPath
        self.realMetaDataPath : str = realMetaDataPath
        
        # categories is used in multiCl;assfier.ipynb
        self.categories = categories
        
        validTypeSet = {'results', 'results.lfcShrink'}
        if not deseqResultsType in validTypeSet :
            emsg = f'deseqResultsType : {deseqResultsType} is not valid must be in set {validTypeSet}'
            logger.error(emsg)
            raise ValueError( emsg)
        
        self.deseqResultsType : str = deseqResultsType
        #self.deseqResultsRootPath = deseqResultsRootPath
    
        self.dataDict : dict[str, DiagnosticBundle] = dict()
        self.outDirPath : str = f'{self.rootOutDir}/{self.dataSetName}'
        self.imgOutDir : str = f"{self.outDirPath}/img"
        self.imgPanelOutDir : str = f"{self.outDirPath}/img/panel"
        os.makedirs(self.imgPanelOutDir, exist_ok=True)
    
        # savePanelDict
        #   key
        #   item is the panel img's filePath
        self.savePanelDict : dict[str, str] = dict()
    
        self.abstract : Optional[str] = None
        
        self.resultsDict : dict[str,str] = dict()


################################################################################
@dataclass
class ModelBundle :
    '''
    TODO
    '''
    X : pd.DataFrame
    y : pd.Series
    model : Optional[BaseEstimator] = None 
    
    # 
    # do not use LabelEncoder. create bugs
    # to many functions assume the positive diagnostic test results is labeled 1
    # and the healthy, negative, control, ... is labeled 0
    # Optional[LabelEncoder] = None
    #
    # orderedCategories[0] is the negative diagnostic test class. i.e healthy_doner
    #
    orderedCategories : Optional[list[str]] = None
    
    confusionMatrix : Optional[np.array] = None
    
    # Optional[type] = None means value is not required in __init__ we can fill it in later
    
    #  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
    # has fig, axes, ...
    cmDisplay : Optional[ConfusionMatrixDisplay] = None
    #cmTitle: Optional[str] = None
    
    # stats contains, tp, fp, fn, tn, sensitify, and specificity
    statsDict : Optional[dict] = None
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    report : Optional[dict] = None
    
    #rocTitle : Optional[str] = None
    rocFig : Optional[plt.Figure] = None
    rocAx : Optional[plt.axes] = None
    
    # if deseqResultsPath is set volcano plot will be be crated
    deseqResultsPath : Optional[str] = None
    volcanoFig : Optional[plt.Figure] = None

    
    # if eval(outDir not None) 
    #   resultsDict will have paths to the saved objects
    resultsDict : Optional[dict] = None
    
################################################################################
@dataclass
class DiagnosticBundle :
    '''
    TODO
    
    diagnosis : str
    
    path to source data before being split into train/validate/test sets
        realCountPath : str
        realMetaPath : str
        simulatedCountPath: str
        simulatedMetaPath : str
    '''
    diagnosis : str
    realCountPath : str
    realMetaPath : str
    simulatedCountPath: str
    simulatedMetaPath : str
    categories : list[str]
    simTrain : ModelBundle
    simVal   : ModelBundle
    simTest  : ModelBundle
    real     : Optional[ModelBundle] = None
