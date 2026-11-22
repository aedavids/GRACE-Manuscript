"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-12-10
Copyright (c) 2020-2023, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 

    describeExplainerStr(explainer : shap.Explanation) -> str:

    displayAnnotationExplinationsForSample(
        annotationDF : pd.DataFrame,
        shapValues : shap._explanation.Explanation,
        sampleId : str,
        classIdx : int, 
        XDF,
        cols : str = ['hgnc_symbol', 'biotype',	'gene_name', 'summary'],
        topN : int,
    ) -> None:
    
    getExplain(
        config : ModelConfig, 
        diagnosis : str, 
        useTraining : bool = True
    ) -> tuple[ shap.Explanation, pd.DataFrame, pd.Series, BaseEstimator]:
    
    getExplainTreeInteractions(
        config : ModelConfig, 
        diagnosis : str, 
    ) -> tuple[ shap.Explanation, BaseEstimator]:      
    
    getExplainTreeInterventional(
        config : ModelConfig, 
        diagnosis : str, 
        useTraining : bool = True
    ) -> tuple[ shap.Explanation, pd.DataFrame, pd.Series, BaseEstimator]:    
    
    getSHAPValues(
        pklPath : str,
        XDF : pd.DataFrame,
        explainer : shap.Explanation
    ) -> shap._explanation.Explanation:
    
    selectShapBySampleIds(XDF : pd.DataFrame, 
                          cmDF : pd.DataFrame, 
                          shapValues : shap._explanation.Explanation, 
                          classIdx : Optional[int] = None
                          ) -> shap._explanation.Explanation
                          
    validateData(
        model: Any,
        df: pd.DataFrame,
    )  -> tuple[pd.DataFrame, list[str]] 
"""

import logging
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

import numpy as np
import os
import pandas as pd
import pickle
import shap
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
# import sys
from typing import Any, Optional

# local imports
from archive.dataTypes import  ModelConfig
from archive.archiveUtilities import getModel, getRealData, getTrainingData
from utilities import smartDisplay

###############################################################################
def displayAnnotationExplinationsForSample(
        annotationDF : pd.DataFrame,
        shapValues : shap._explanation.Explanation,
        sampleId : str,
        classIdx : int, 
        XDF,
        cols : str = ['hgnc_symbol', 'biotype',	'gene_name', 'summary'],
        topN : int=10,
    ) -> None:
    '''
    provides biologic explinations for sample's shap values
    
    If called from notebook uses IPython display() else print
    arguments:
    
         annotationDF :
            a data frame create by data.geneAnnotator
            
        shapValues :
        
        sampleId:
            example 'SRR14506690'
                
        classIdx :
            shap value class index
            
            model.classes_ is in sort order
            this may not be the same order as 
            shap values use same order realLung_ySeries

            example :
                shapHealthyLungClassIdx = list(lungRFModel.classes_).index("Healthy_donor")
                shapCancerLungClassIdx = list(lungRFModel.classes_).index("Lung_Cancer")
                
        XDF:
            used to map sample id to shap value row indices
            
        cols:
            annotation columns to display
            default ['hgnc_symbol', 'biotype',	'gene_name', 'summary']
            
        topN:
            number of features to display

    
    '''
    
    # shapValues=realLungShapValues
    # classIdx=shapHealthyLungClassIdx 
    # sampleId = 'SRR14506690'
    # XDF=realLungXDF
    
    dataIndexInShapValue = XDF.index.get_loc(sampleId)
    
    logger.warning(f'AEDWIP TODO refactor see shapUtilities.rankTopFeaturesByShap()')
    
    # sv is a shape.Explanation
    # https://shap.readthedocs.io/en/latest/generated/shap.Explanation.html
    sv = shapValues [dataIndexInShapValue,:, classIdx]
    # print(f'sv.shape : {sv.shape}')
    # print(f'sv.values.shape : {sv.values.shape}')
    # print(f'sv.values[0:5] : {sv.values[0:5]}')

    
    # sort
    idx = np.argsort(np.abs(sv.values))[::-1]
    # print(f'idx[:10] : {idx[:10]}')
    
    # sv.features_names is a list
    # convert to numpy array so that we can use fancy indexing
    featureNamesNP = np.array(sv.feature_names)
    topFeatures = featureNamesNP[idx[:topN]]
    # print(f'topFeatures : {topFeatures}')
    topValues   = sv.values[idx[:10]]

    # try:
    #     from IPython.display import display
    #     IN_IPYTHON = True
    # except ImportError:
    #     IN_IPYTHON = False
    
    # # aedwipAannotationDF = xaiDict['Lung']['annotations'].set_index('input_id')
    # # cols = ['hgnc_symbol', 'biotype',	'gene_name', 'summary']
    
    # if IN_IPYTHON:
    #     with pd.option_context("display.max_colwidth", None) :
    #         display(annotationDF.loc [topFeatures, cols])
    # else :
    #     print(annotationDF.loc [topFeatures, cols])
    
    smartDisplay(annotationDF.loc [topFeatures, cols])
        

###############################################################################
def describeExplainerStr(explainer : shap.Explanation) -> str:
    '''
    useful for debugging
    
    returns a string
    '''
    
    a = f'type : {type(explainer)} {type(explainer).__name__}'
    b = f'\nfeaturePerturbation : {getattr(explainer, "feature_perturbation", None)}'
    c = f'\nmasker : {type(explainer.masker)} {type(explainer.masker).__name__}'
    hasData = hasattr(explainer, "data")
    d = f'\nhas data: {hasData}'
    
    return a + b + c + d


###############################################################################
def getExplain(
        config : ModelConfig, 
        diagnosis : str, 
        useTraining : bool = True
    ) -> tuple[ shap.Explanation, pd.DataFrame, pd.Series, BaseEstimator]:
    ''' 
    creates and explainer based on sklearn model type.
    
    if the Explainer() can not figure out what the model type is
    it defaults to permuation. this is very computationaly expensive and time consuming
    
    see getExplainTreeInterventional() and aedwip
    for tree, ie random forest classifier  explainer
    
    argumenents:
    
        useTraining :
            if true 
                training data set is used as the background dataset
                for integrating out features.
            else :
                the 'real' data set is used
                
    notes:
    
        check if feature_perturbation=independent
        
        SHAP samples it independently from its marginal distribution
        ignores correlations with other features
        This is fast but breaks biological correlations, creates unrealistic molecular states
        and can not define interactions. 
        
    see describeExplainerStr()
    
    returns (explainer, XDF, ySeries, model)
    ''' 
    model = getModel(config, diagnosis)
    if useTraining :
        XDF, ySeries = getRealData(config, diagnosis)

    else : 
        XDF, ySeries = getTrainingData(config, diagnosis)
            
    explainer = shap.Explainer(model, XDF)

    return (explainer, XDF, ySeries, model)

###############################################################################
def getExplainTreeInteractions(
        config : ModelConfig, 
        diagnosis : str, 
    ) -> tuple[ shap.Explanation, BaseEstimator]:  
    '''
    creat a tree explainer by setting feature_perturbation = 'tree_path_dependent'
    
    Since SHAP values rely on conditional expectations, we need to decide how to 
    handle correlated (or otherwise dependent) input features. No background 
    dataset is required and the approach is to just follow the trees 
    and use the number of training examples that went down each leaf to 
    represent the background distribution.
    
    ref: 
        https://github.com/shap/shap/blob/master/shap/explainers/_tree.py
        https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html#shap.TreeExplainer 
        
    '''
    
    model = getModel(config, diagnosis)
    isRandomForest = isinstance(model, RandomForestClassifier)
    if isRandomForest :
        fp = "tree_path_dependent"
        explainer = shap.Explainer(model, feature_perturbation=fp)
    else:  
        emsg = f'TODO model is not a RandomForestClassifier. got : {type(model).__name__}tree explainer would probably still work'
        logger.error(emsg)
        raise TypeError(emsg)
    
    return (explainer,  model)

###############################################################################
def getExplainTreeInterventional(
        config : ModelConfig, 
        diagnosis : str, 
        useTraining : bool = True
    ) -> tuple[ shap.Explanation, pd.DataFrame, pd.Series, BaseEstimator]:  
    '''
    create a tree explainer by setting feature_perturbation = "interventional".
    SHAP values rely on conditional expectations. The dependencies between features 
    are handled according to the rules dictated by causal inference
    
    Use this explainer to calculate main effect, feature importance,
    waterfall plots, beeswarms, ranking features, and per-sample explinations
    
    Do not use to calculate interactions. It uses class 'shap.maskers._tabular.Independent
    FEATURE_DEPENDENCE::independent does not support interactions!
    
    arguments:
        useTraining :
                if true 
                    training data set is used as the background dataset
                    for integrating out features.
                else :
                    the 'real' data set is used
                    
    ref: 
        https://github.com/shap/shap/blob/master/shap/explainers/_tree.py
        https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html#shap.TreeExplainer 
        
        Janzing, Dominik, Lenon Minorics, and Patrick Blöbaum. 
        “Feature relevance quantification in explainable AI: A causal problem.” 
        International Conference on artificial intelligence and statistics. PMLR, 2020.
        https://proceedings.mlr.press/v108/janzing20a/janzing20a.pdf 
    '''
    
    model = getModel(config, diagnosis)

    if useTraining :
        XDF, ySeries = getTrainingData(config, diagnosis)

    else : 
        XDF, ySeries = getRealData(config, diagnosis)

    
    isRandomForest = isinstance(model, RandomForestClassifier)
    
    if isRandomForest :
        fp = "interventional"
        explainer = shap.Explainer(model, XDF, feature_perturbation=fp)
        
    else:
        emsg = f'TODO model is not a RandomForestClassifier. got : {type(model).__name__} tree explainer would probably still work'
        logger.error(emsg)
        raise TypeError(emsg)
    
    return (explainer, XDF, ySeries, model)

    
###############################################################################
def getSHAPValues(
        pklPath : str,
        XDF : pd.DataFrame,
        explainer : shap.Explanation
    ) -> shap._explanation.Explanation:
    '''
    check if shap values are stored in local cache
    else calls explainer, and save to 
    
    arguments 
    
    pklPath :
        if not None:
            load and save file path
        
    XDF : 
        the sample/examples you want to compute SHAP values for
        
    explainer :
        a shap explaintion that was fit on background data
        example: 
            lungExplainer = shap.Explainer(lungRFModel.predict_proba, lungTrainingXDF)
        
    ref: attributes section  https://shap.readthedocs.io/en/latest/generated/shap.Explanation.html
    
    values (the actual SHAP numbers)
    sv.values.shape == (nSamples, nFeatures)
    
    base_values = E[f(X)] under the background (reference) distribution
    data (feature values) i.e the original data
    feature_names
    output_names
    clustering, bounds, etc.
    
    '''
    retShapValues = None
    if pklPath and os.path.exists(pklPath) and os.path.isfile(pklPath) : 
        # load
        logger.warning(f'loading : {pklPath}')
        with open(pklPath, 'rb') as fd:
            retShapValues = pickle.load(fd)
    else :
        # print(f'XDF.shape : {XDF.shape}')
        retShapValues = explainer(XDF)
        # save
        if pklPath :
            with open(pklPath, 'wb') as fd:
                pickle.dump(retShapValues, fd)
            logger.warning(f'saved : {pklPath}')
        
    return retShapValues

###############################################################################
def selectShapBySampleIds(XDF : pd.DataFrame, 
                          cmDF : pd.DataFrame, 
                          shapValues : shap._explanation.Explanation, 
                          classIdx : Optional[int] = None
                          ) -> shap._explanation.Explanation :
    '''
    selects sample shap values for samples in cmDF
    
    example of use:
        svTP = selectShapBySampleIds(realLungXDF, realLungCFDict['TP'],
                             realLungShapValues, shapCancerLungClassIdx)
   
    arguments:
        XDF :
            used to map cmDF sample ids to shap value row indices
        
        cmDF:
            dataframe returned by  data.utilities.getBinaryConfusionDataSets()
            
        
        shapeValues :
        
        classIdx :
            shap value class index
            
            model.classes_ is in sort order
            this may not be the same order as 
            shap values use same order realLung_ySeries

            example :
                shapHealthyLungClassIdx = list(lungRFModel.classes_).index("Healthy_donor")
                shapCancerLungClassIdx = list(lungRFModel.classes_).index("Lung_Cancer")
    '''
    sampleIds = list(cmDF.index)
    rowIdx = [XDF.index.get_loc(sid) for sid in sampleIds]

    sv = shapValues[rowIdx]  # selects samples (works for Explanation)

    # If multiclass, optionally pick a class
    if sv.values.ndim >= 3:
        if classIdx is None:
            raise ValueError("shapValues are multiclass; pass classIdx.")
        # slice by (samples, features, classes)
        sv = sv[:, :, classIdx]  # now (samples, features)

    return sv

###############################################################################
def validateData(
        model: Any,
        df: pd.DataFrame,
    )  -> tuple[pd.DataFrame, list[str]] :
    ''' 
    make sure sample data matches the expected model format
    ie. the same columns in the same order
    
    Validate and align a dataframe to a fitted sklearn-style model.

    This utility:
    1. Confirms the model exposes feature_names_in_
    2. Confirms all required model features are present in df
    3. Optionally rejects unexpected extra columns
    4. Reorders df columns to exactly match model.feature_names_in_

    The returned dataframe is safe to use with model.predict(),
    model.predict_proba(), and SHAP so that feature ordering stays consistent.

    Args:
        model:
            A fitted model with a feature_names_in_ attribute.
        df:
            Input dataframe to validate and align.
            
    Returns:
        alignedDf:
            A dataframe containing only the model features, in model order.
        featureNames:
            The aligned feature names in model order.

    Raises:
        TypeError:
            If df is not a pandas DataFrame.
        ValueError:
            If the model lacks feature_names_in_, if required features are
            mising, or if unexpected extra columns are present.
    '''
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Model does not have feature_names_in_. "
            "Cannot validate or align feature order."
        )
        
    # make sure feature names and column names are strings not integers
    retModelFeatureNames: list[str] = [str(x) for x in model.feature_names_in_]
    dfColumnNames: list[str] = [str(x) for x in df.columns]

    modelFeatureSet: set[str] = set(retModelFeatureNames)
    dfColumnSet: set[str] = set(dfColumnNames)

    missingFeatures: list[str] = sorted(modelFeatureSet - dfColumnSet)
    extraFeatures: list[str] = sorted(dfColumnSet - modelFeatureSet)

    if missingFeatures:
        raise ValueError(
            "Dataframe is missing required features.\n"
            f"Missing features ({len(missingFeatures)}): {missingFeatures}"
        )

    if extraFeatures:
        raise ValueError(
            "Dataframe contains unexpected extra features.\n"
            f"Extra features ({len(extraFeatures)}): {extraFeatures}"
        )

    # make sure the features are in the correct order
    retAlignedDF: pd.DataFrame = df.loc[:, retModelFeatureNames].copy()
    alignedColumns: list[str] = [str(x) for x in retAlignedDF.columns]
    if alignedColumns != retModelFeatureNames:
        raise ValueError(
            "Feature alignment failed. "
            "Aligned dataframe columns do not match model.feature_names_in_."
        )

    return retAlignedDF, retModelFeatureNames