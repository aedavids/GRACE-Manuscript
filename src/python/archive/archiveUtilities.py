"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-10-04
Copyright (c) 2020-2027, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 

public functions: 
    getModel(config : ModelConfig, diagnosis:str ) -> BaseEstimator
    
    getOrderedCategories(config : ModelConfig, diagnosis:str) -> list[str]:
    
    getRealData(config : ModelConfig, diagnosis:str ) -> tuple[pd.DataFrame, pd.Series]:
    
    getTrainingData(config : ModelConfig, diagnosis:str ) -> tuple[pd.DataFrame, pd.Series]:
    
    is_empty(s: str) -> bool:
    
    def loadArchive(archivePath : str) -> dict[str, Any]
    
    printDictTree(d : dict, 
                  indent : str="",
                  displayType :bool=True) -> None:
                  
    saveFigure(fig : plt.figure, title: str, outDirArg : str,  prefix : str = "") -> str
    
    urlify(s : str) -> str:

"""
from sklearn.base import BaseEstimator  # base class for all sklearn models

import logging
logger = logging.getLogger(__name__)

import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from typing import Any

# local imports
from archive.dataTypes import  ModelConfig


###############################################################################
def urlify(s : str) -> str :
    '''
    useful function for converting plot titles to strings
    https://stackoverflow.com/a/1007615/4586180

    print(urlify("I can't get no satisfaction!"))
    Prints: I-cant-get-no-satisfaction"
    '''

    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)

    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", '-', s)
    
    return s

###############################################################################
def getModel(config : ModelConfig, diagnosis:str ) -> BaseEstimator:
    ''' 
    returns the simulateTrain model bundle's model
    '''
    diagnosticBundle = config.dataDict[diagnosis]
    modelBundle = diagnosticBundle.simTrain 
    model = modelBundle.model
    
    return model

###############################################################################
def getOrderedCategories(config : ModelConfig, diagnosis:str) -> list[str]:
    ''' 
    returns the orderedCategories list for the real model bundle
    
    example :        
        config=runConfig_G, 
        diagnosis="Liver"
        
        return = Index(['Healthy_donor', 'Liver_Cancer'], dtype='object')
    '''
    diagnosticBundle = config.dataDict[diagnosis]
    modelBundle = diagnosticBundle.real 
    return( modelBundle.orderedCategories)

###############################################################################
def getRealData(config : ModelConfig, diagnosis:str ) -> tuple[pd.DataFrame, pd.Series]:
    ''' 
    returns real data set (X, y)
    '''
    diagnosticBundle = config.dataDict[diagnosis]
    modelBundle = diagnosticBundle.real 
    return( modelBundle.X, modelBundle.y)

###############################################################################
def getTrainingData(config : ModelConfig, diagnosis:str ) -> tuple[pd.DataFrame, pd.Series]:
    ''' 
    returns simulated training data set (X, y)
    '''
    diagnosticBundle = config.dataDict[diagnosis]
    modelBundle = diagnosticBundle.simTrain 
    return( modelBundle.X, modelBundle.y)

###############################################################################
def is_empty(s: str) -> bool :
    """
    Check if a string is None, empty, or only whitespace.
    
    Parameters
    ----------
    s : str
        The string to check. Can also be None.
    
    Returns
    -------
    bool
        True if the string is None, empty, or only whitespace. 
        False otherwise.
    """
    return s is None or s.strip() == ""

###############################################################################
def loadArchive(archivePath : str) -> dict[str, Any] :
    ''' 
    loads a json file created by from archive.saveDataTypes import archive
    
    raises FileNotFoundError
    '''
    retArchiveDict = None
    if os.path.exists(archivePath) and os.path.isfile(archivePath) : 
        print(f"'{archivePath}' exists .")
        with open(archivePath, 'r') as fd:
            retArchiveDict = json.load(fd)
    else :
        emsg = f'archive path does not exist : {archivePath}'
        logger.error(emsg)
        raise FileNotFoundError(emsg)
    
    return retArchiveDict
    
###############################################################################
def printDictTree(d : dict, 
                  indent : str="",
                  displayType :bool=False) -> None:
    """
    Recursively print nested dict keys as a tree.
    """
    last_key = list(d.keys())[-1] if d else None
    
    for key in d:
        is_last = (key == last_key)
        connector = "└── " if is_last else "├── "
        
        v = d[key]
        if displayType and not isinstance(v, dict):
            print( indent + connector + str(key) + ":" + v.__class__.__name__ ) 
        else :
            print(indent + connector + str(key))
        
        # Recursively print children if value is a dict
        if isinstance(d[key], dict):
            new_indent = indent + ("    " if is_last else "│   ")
            printDictTree(d[key], new_indent)


###############################################################################
def saveFigure(fig : plt.figure, title: str, outDirArg : str,  prefix : str = "") -> str:
    ''' 
    TODO 
    save PNG and PDF versions files in a consistent way
    
    
    arguments:
        title:
            there are several ways the title could be set
            It is not clear how to get the title in a consistent way
            the output file name will be a URL safe version of the title  
            
            
    returns the png output file path
    '''
    os.makedirs(outDirArg, exist_ok=True)
    fileName = urlify(prefix + "_" + title).strip()
    outPathPNG = f'{outDirArg}/{fileName}.png'
    outPathPDF = f'{outDirArg}/{fileName}.pdf'

    # bbox_inches="tight" reduces extra whitespace
    #  if clipping try adding pad_inches=0.1 to leave a bit of breathing room:
    # facecolor="white", sets the background color
    # transparent=True, 
    #   In a PNG, this means the alpha channel is set to 0 (clear).
    #   In a PDF/SVG, it means no opaque rectangle is drawn behind your plot.
    fig.savefig(outPathPNG, bbox_inches="tight",  dpi=600 )
    logger.warning(f"saved : {outPathPNG}")
    
    fig.savefig(outPathPDF, bbox_inches="tight",  dpi=600 )
    logger.warning(f"saved : {outPathPDF}")

    
    # to maintain backwards compatiblity only the PNG path is returned
    return outPathPNG
