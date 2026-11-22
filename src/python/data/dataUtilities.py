"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-12-10
Copyright (c) 2020-2027, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 

    getBinaryConfusionDataSets(
        model : BaseEstimator,
        XDF : pd.DataFrame, 
        ySeries : pd.Series
        ) -> dict[str, pd.DataFrame] :
        
    printCancerKeywordsAndRelevanceScore(
        annotationsDF : pd.DataFrame,
        features : list[str], 
        cols : list[str] = ['hgnc_symbol', 'biotype',	'gene_name', 'summary'],
    ) -> None:
    
    saveList(listPath : str, l : list, isSingleItemLine : bool=False)
    
"""

import logging
logger = logging.getLogger(__name__)

import os
import pandas as pd
import pprint as pp
from sklearn.base import BaseEstimator
import sys

# local imports
from data.cancerRelevanceScoring import addCancerRelevanceScore
from data.termLibrary import getDefaultTermLibrary
from data.textAnnotator import annotateManySummaries, compileTermLibrary
from explainability.plotSHAP import formatTopContributors
from utilities import smartDisplay
from utilities import isOrderedCategoricalSeries


################################################################################
def printCancerKeywordsAndRelevanceScore(
        annotationsDF : pd.DataFrame,
        features : list[str], 
        cols : list[str] = ['hgnc_symbol', 'biotype',	'gene_name', 'summary'],
    ) -> None:
    '''
    1) search the 'summary field in the annotaions dataframe for cancer 
    releated keywords and prints them
    
    2) prints keyword cancer relevance score
    
    arguments:
        
        annotationsDF :
                a data frame create by data.geneAnnotator
                
        features:
            list of features of interest

        cols :
            annotation columns to display
            default == ['hgnc_symbol', 'biotype',	'gene_name', 'summary']
    '''
    
    featureAnnotationsDF = annotationsDF.loc[features, cols]
    if 'summary' in cols:
        # run simple regx/NLP for key words
        termLibrary = getDefaultTermLibrary()
        compiledLibrary = compileTermLibrary(termLibrary) 
        cleanDF = featureAnnotationsDF.loc[:, 'summary'].dropna()
        summariesByGeneIdDict = cleanDF.to_dict()
        nlpDict = annotateManySummaries(summariesByGeneIdDict , compiledLibrary)
        #smartDisplay(nlpDict)
        print(f'regx/NLP keyword search of summary')
        for key, item in nlpDict.items():
            print(f"\n{key}")
            print(pp.pformat(item))

        
        # compute cancer relevance score
        for key, annotation in nlpDict.items():
            # smartDisplay("\n" + "=" * 60)
            print(f"\nGene: {key}")

            scoredAnnotation = addCancerRelevanceScore(annotation)
            crScore = scoredAnnotation["cancerRelevanceScore"]

            print(f"Cancer Relevance Score: {crScore:.2f}")

            if crScore <= 0:
                print("Top Contributors: None")
                continue

            print("Top Contributors:")
            print(formatTopContributors(scoredAnnotation["topContributors"]))

            
    # display all the annotation information
    # smartDisplay(interactionAnnotationsDF)
    
################################################################################
def getBinaryConfusionDataSets(
    model : BaseEstimator,
    XDF : pd.DataFrame, 
    ySeries : pd.Series
    ) -> dict[str, pd.DataFrame] :
    '''
    calls model.predict() then splits the predictions into 4 dataframes
    TP, TN, FP, FN
    
    assumes model binary classifier
    
    returns a dictionary with keys 
        'TP', 'TN', 'FP', 'FN'
        the values are data frames with columns 'yTrue' and 'yPred'
    '''
    
    if not isOrderedCategoricalSeries(ySeries):
        errMsg = f'argument ySeries {type(ySeries)} expected pandas series with an ordered categorical dtype'
        logger.error(errMsg)
        raise TypeError(errMsg)


    retDict = dict()
    
    # predictions are string values
    # encode predictions
    # TODO AEDWIP should we leave predictions as strings?
    # we know that ySeries.cat.categories[0] is the control class?
    predictionsNP = model.predict( XDF )
    
    categories = ySeries.cat.categories
    yPredictionSeries = pd.Series(
                             pd.Categorical( predictionsNP, 
                                   categories, 
                                   ordered=True),
                             index=ySeries.index
                        )
    encodedPredictionsNP = yPredictionSeries.cat.codes.values

    df = pd.DataFrame(data={
                            "yTrue": ySeries.cat.codes.values,
                            "yPred": encodedPredictionsNP
                            },
                    index=XDF.index
                )
    
    TP_DF = df[ (df['yTrue'] == 1) &  (df['yPred'] == 1)]
    retDict['TP'] = TP_DF
    
    TN_DF = df[ (df['yTrue'] == 0) &  (df['yPred'] == 0)]
    retDict['TN'] = TN_DF
    
    FP_DF = df[ (df['yTrue'] == 0) &  (df['yPred'] == 1)]
    retDict['FP'] = FP_DF
    
    FN_DF = df[ (df['yTrue'] == 1) &  (df['yPred'] == 0)]
    retDict['FN'] = FN_DF
    
    return retDict


################################################################################
def saveList(listPath : str, l : list, isSingleItemLine : bool=False):
    '''
    TODO
    /private/home/aedavids/extraCellularRNA/deconvolutionAnalysis/python/analysis/utilities.py
    '''
    with open(listPath, "w") as f:
        if not isSingleItemLine:
            f.write(str(l))
        else:
            for value in l:
                f.write(str(value) +"\n")
                
    print(f'saved: {listPath}')
