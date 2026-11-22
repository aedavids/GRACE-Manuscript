"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-12-10
Copyright (c) 2020-2023, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 

public functions 

    createWaterfallTitle(
                config : ModelConfig, 
                # diagnosis:str, 
                classIdx : int,
                confusionDataSetDict : dict[str, pd.DataFrame],
                confusionKey : str,
                ) -> str :
    
    plotInteraction(sv : shap.Explanation,
                    intractionTup : tuple[str, str, float],
                    classIdx : int,
                    width : Optional[float] = None,
                    title : Optional[str]=None
                ) -> plt.Figure:
                
    plotInteractionHelper(sv : shap.Explanation,
                    intractionTup : tuple[str, str, float],
                    classIdx : int,
                    annotationsDF : pd.DataFrame,
                    width : Optional[float] = None,
                    title : Optional[str]=None
                ) -> plt.Figure:
                    
    plotWaterfall(
        sv : shap._explanation.Explanation,
        width : Optional[float] = None,
        title : Optional[str] = None,
        maxNumberOfFeaturesToDisplay : int = 10
        ) -> plt.Figure :
        
    waterfallPlotHelper(
        config : ModelConfig, 
        confusionDataSetDict : dict[str, pd.DataFrame],
        shapValues : shap._explanation.Explanation,
        classIdx : int,
        confusionKey : str,
        #dataIndex : int,
        sampleId : str,
        XDF : pd.DataFrame,
        width : Optional[float] = None,
        # height : Optional[float] = None,
        maxNumberOfFeaturesToDisplay : int = 10, # the shap waterfall plot default
        debug : bool = False
        ) -> plt.Figure :
"""

import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint as pp
import shap
from typing import Optional


# local imports 
from archive.dataTypes import  ModelConfig
from archive.archiveUtilities import getModel, getOrderedCategories 
from data.cancerRelevanceScoring import addCancerRelevanceScore
from data.textAnnotator import annotateManySummaries, compileTermLibrary 
from data.textAnnotator import getDefaultTermLibrary
from plots.plotUtilities import calculateAspectRatio, heightForAspectRatio
from utilities import smartDisplay

################################################################################
def createWaterfallTitle(
            config : ModelConfig, 
            # diagnosis:str, 
            classIdx : int,
            confusionDataSetDict : dict[str, pd.DataFrame],
            sampleId : str, 
            confusionKey : str,
            ) -> str :
    ''' 
    
    arguments:
        config
                
        classIdx :
            classifiers train a model for each class. classIdx = 0 should be the
            control class. ie something like healthy
            
        confusionDataSetDict:
            dictionary created by getConfusionDataSets()
            
        sampleId :
            
        confusionKey :
            ['TP', 'TN', 'FP', 'FN']
    '''
    
    validKeys = ['TP', 'TN', 'FP', 'FN']
    
    if not confusionKey in validKeys :
        emsg = f'confusionKey not in {validKeys}'
        logger.error(emsg)
        raise ValueError(emsg)       
        
    confusionDF = confusionDataSetDict[confusionKey]
    if confusionDF.empty :
        emsg = f'confusionDF[{confusionKey}] dataFrame was empty'
        logger.error(emsg)
        raise ValueError(emsg)
    
    # # pick a sample we know is a true positive
    # dataIndex=0,
    # sampleIdStr is a pandas.core.series.Series
    # 'SRR14506725'
    # sampleIdStr = str(confusionDF.iloc[dataIndex].name)
    
    # liverRFModel = getModel(runConfig_G, diagnosis="Liver")
    groundTrue = confusionDF.loc[sampleId, 'yTrue']
    prediction = confusionDF.loc[sampleId, 'yPred'] #liverRFModel.predict()
    
    diagnosis = confusionDataSetDict['diagnosis']
    orderedIndex = getOrderedCategories(config, diagnosis)
    groundTrueStr = orderedIndex[groundTrue]
    predictionStr = orderedIndex[prediction]
    
    # modelType will be something like RandomForestClassifier
    model = getModel(config, diagnosis )
    modelType = type(model).__name__
    modelName = f'{diagnosis} {modelType}'
    
    classNameStr = orderedIndex[classIdx]
    # print(f'AEDWIP classNameStr: {classNameStr}, classIdx : {classIdx}, {orderedIndex}')

    template = "sample = s8dkd, model = liverCancerRF, class = healthy, y = Liver Cancer, predicted = healthy, "
    # retTitle = f"sample:{sampleIdStr}, model:{modelName}, class:{classNameStr}, y:{groundTrueStr}, p:{predictionStr}"
    retTitle = f"{sampleId}, {modelName}, {classNameStr}, y:{groundTrueStr}, p:{predictionStr}"
    print(f'\n {retTitle}')
    
    return retTitle

################################################################################
def formatTopContributors(topContributors) -> str:
    if not topContributors:
        return "None"

    lines = [
        f"  • {category}: {score:.2f}"
        for category, score in topContributors
        if score > 0
    ]
    return "\n".join(lines) if lines else "None"

################################################################################
def plotInteraction(sv : shap.Explanation,
                    intractionTup : tuple[str, str, float],
                    classIdx : int,
                    width : Optional[float] = None,
                    title : Optional[str]=None
                ) -> plt.Figure:
    '''
    create a scatter plot of feature1 colored by feature 2
    arguments :
    
        interactionTup :
            tuple returned by explainability.interactions.calculateInteractions
            tup[0] = featureName1
            tup[1]=  featureName2
            tup[2] = interaction value
            
        classIdx:
            model.classes_ is in sort order
            this may not be the same order as 
            shap values use same order realLung_ySeries

            example
            shapHealthyLungClassIdx = list(lungRFModel.classes_).index("Healthy_donor")
            
        width :
            if not None, will cause figure to be resized
            the height will be calculated based on the 
            original figures aspect ratio
            
        title :
            
    return :
        Figure will not be displayed. use plt.show()
    '''
    feature1, feature2, v = intractionTup
    # cols = ['hgnc_symbol', 'biotype',	'gene_name', 'summary']
    # with pd.option_context("display.max_colwidth", None) :
    #     display(annotationsDF.loc [[feature1, feature2], cols])
    # print(f'interaction value : {v}')


    # shap.plots.scatter( sv[:, feature1, classIdx] )

    # shap.plots.scatter( sv[:, feature2, classIdx]  )

    ax = shap.plots.scatter( sv[:, feature1, classIdx], 
                        color=sv[:, feature2, classIdx],
                        show=False, )
    
    if not ax :
        logger.error(f'!!!!!! ax is None do not set show=True')
    
    if title :
        ax.set_title( title )
        
    fig = ax.get_figure()
    if width :
        # resize
        aspectRatio = calculateAspectRatio(fig)
        height = heightForAspectRatio(width, aspectRatio)
        fig.set_size_inches(width, height, forward=True)
    
    return fig    

################################################################################
def plotInteractionHelper(sv : shap.Explanation,
                    intractionTup : tuple[str, str, float],
                    classIdx : int,
                    annotationsDF : pd.DataFrame,
                    cols : list[str] = ['hgnc_symbol', 'biotype',	'gene_name', 'summary'],
                    width : Optional[float] = None,
                    title : Optional[str]=None
                ) -> plt.Figure:
    ''' 
    create a mini report. display annotation for interaction features and calls plotInteraction()
    The interaction value will be appened to the title 
    
    if 'summary' in cols include regex/nlp key word search
    
    arguments :
    
        interactionTup :
            tuple returned by explainability.interactions.calculateInteractions
            tup[0] = featureName1
            tup[1]=  featureName2
            tup[2] = interaction value
            
        classIdx:
            model.classes_ is in sort order
            this may not be the same order as 
            shap values use same order realLung_ySeries

            example
            shapHealthyLungClassIdx = list(lungRFModel.classes_).index("Healthy_donor")
            
        annotationsDF :
                a data frame create by data.geneAnnotator

        cols :
            annotation columns to display
            default == ['hgnc_symbol', 'biotype',	'gene_name', 'summary']
            
        width :
            if not None, will cause figure to be resized
            the height will be calculated based on the 
            original figures aspect ratio
            
        title :
        
    returns plot figure
    '''
    
    feature1, feature2, v = intractionTup
    
    # with pd.option_context("display.max_colwidth", None) :
    #     display(annotationsDF.loc [[feature1, feature2], cols])

    # format :.4f print upto 4 deciminals with no trailing zeros
    interactionStr = f'I={v:.4f}'.rstrip("0").rstrip(".") 
    if title :
        title = f'{title} {interactionStr}'
    else :
        title = f'{interactionStr}'
        
    retFig = plotInteraction(sv, intractionTup, classIdx, width, title)
    
    # add gene annotation information
    interactionAnnotationsDF= annotationsDF.loc [[feature1, feature2], cols]
    logger.error(f'AEDWIP this was refactored into data.dataUtilities.py')
    if 'summary' in cols:
        # run simple regx/NLP for key words
        termLibrary = getDefaultTermLibrary()
        compiledLibrary = compileTermLibrary(termLibrary) 
        cleanDF = interactionAnnotationsDF.loc[:, 'summary'].dropna()
        summariesByGeneIdDict = cleanDF.to_dict()
        nlpDict = annotateManySummaries(summariesByGeneIdDict , compiledLibrary)
        #smartDisplay(nlpDict)
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
    smartDisplay(interactionAnnotationsDF)
    
    
    return retFig
   
################################################################################ 
def plotScatterPlotWithMarkers(
    shapValues : shap.Explanation,
    ySeries : pd.Series,
    classIdx : int,
    mainFeatureName : str, 
    colorByFeatureName : str,
):
    '''
    AEDWIP : TODO
    
    Goal: can we make it easier to understand scatter plots by using 
    different markers to identify the samples class label

    create a colored by scatter plot.

    # This does not work well, 
    '''
    logger.warning(f'THIS IS AN experimental implementation, there are hard coded values')
    # fig, ax = plt.subplots(figsize=(7, 4))

    # display ref
    ax = shap.plots.scatter(
        shapValues[:, mainFeatureName, classIdx], 
        color=shapValues[:, colorByFeatureName, classIdx],
        alpha=1, #0.75, #0.5, # 0.15,
        show=True,
    )

    #
    # create plot with markers
    #
    ax = shap.plots.scatter(
        shapValues[:, mainFeatureName, classIdx], 
        color=shapValues[:, colorByFeatureName, classIdx],
        alpha=1, #0.2, #1, #0.75, #0.5, # 0.15,
        show=False,
    )

    # resize
    fig = ax.get_figure()
    width = fig.get_figwidth()
    AEDWIPWidth = 10
    width = width + AEDWIPWidth
    if width :
        # resize
        aspectRatio = calculateAspectRatio(fig)
        height = heightForAspectRatio(width, aspectRatio)
        fig.set_size_inches(width, height, forward=True)
        

    # ensure we have the real axes
    ax = plt.gca()

    #
    # overlay class markers (same x/y data, but different marker styles)
    #
    x = shapValues[:, mainFeatureName, classIdx].data
    y = shapValues[:, mainFeatureName, classIdx].values

    healthyMask = (ySeries == 'Healthy_donor')
    cancerMask  = (ySeries == 'Lung_Cancer')

    # s=28 linewidths=1.2
    lineWidth=0.5
    edgeColor = (0, 0, 0, 0.35)

    # is the area of the markers already drawn by shap
    baseS = np.median(ax.collections[0].get_sizes())
    print(f'baseS : {baseS}')

    ax.scatter(x[healthyMask], y[healthyMask], marker="o", 
                # s is the marker area
                s=baseS*1.6, #10, #28, #80, 
                facecolors="none", 
                edgecolors=edgeColor, #"black",
                linewidths=lineWidth, #0.5, #1.2, #2.0, 
                label="Healthy",
                zorder=10
                )

    # s=30 linewidths=1.2
    squareMarker = "s"
    triangleUpMarker="^"
    ax.scatter(x[cancerMask],  y[cancerMask],  marker=squareMarker, 
                # s is the marker area
                s=baseS*1.6, #10, #30, #90, 
                facecolors="none",  
                edgecolors=edgeColor, #"black",
                linewidths=lineWidth, #1.2, #2.0, 
                label="Cancer",
                zorder=11
                )

    ax.legend(frameon=False)

    ax.set_title(f"SHAP scatter: {mainFeatureName} (markers = class)")

    # print(len(ax.collections), [type(c).__name__ for c in ax.collections[-3:]])

    plt.show()
    
    
################################################################################
def plotWaterfall(
    sv : shap._explanation.Explanation,
    width : Optional[float] = None,
    title : Optional[str] = None,
    maxNumberOfFeaturesToDisplay : int = 10
    ) -> plt.Figure :
    '''
    
    arguments:
        sv: a row from object retuned by explainer(XDF)
        
        example:
            sv = shapValues[dataIndex,:, classIdx]
    
        width
            if not None, will cause figure to be resized
            the height will be calculated based on the 
            original figures aspect ratio
    
        maxNumberOfFeaturesToDisplay : 
            show maxNumberOfFeaturesToDisplay + 1 rows
            if set to 5 and there are 10 features, 
            the label for the last row will be + 4 other feaures
            
        title
        
    return :
        Figure will not be displayed. use plt.show()
        
    sanity check model prediction equal additive shap values
    model(x[i]) == sv[i].base_values + sv[i].values.sum()
    '''
    
    # sv and realLungShapValues are shap._explanation.Explanation objs
    #
    # values (the actual SHAP numbers)
    # sv.values.shape == (nSamples, nFeatures)
    #
    # base_values = E[f(X)] under the background (reference) distribution
    # data (feature values) i.e the original data
    # feature_names
    # output_names
    # clustering, bounds, etc.
    #
 
    ax = shap.plots.waterfall(sv, show=False, max_display=maxNumberOfFeaturesToDisplay)
    if not ax :
        logger.error(f'!!!!!! ax is None do not set show=True')
    
    if title :
        ax.set_title( title )
        
    fig = ax.get_figure()

    if width :
        # resize
        aspectRatio = calculateAspectRatio(fig)
        height = heightForAspectRatio(width, aspectRatio)
        fig.set_size_inches(width, height, forward=True)
    
    return fig
 

#################################################################################
def waterfallPlotHelper(
        config : ModelConfig, 
        confusionDataSetDict : dict[str, pd.DataFrame],
        shapValues : shap._explanation.Explanation,
        classIdx : int,
        confusionKey : str,
        #dataIndex : int,
        sampleId : str,
        XDF : pd.DataFrame,
        width : Optional[float] = None,
        # height : Optional[float] = None,
        maxNumberOfFeaturesToDisplay : int = 10, # the shap waterfall plot default
        debug : bool = False
        ) -> plt.Figure:
    ''' 
    easy of use class
    
    arguments:
    
    config
    
    confusionDataSetDict
    
    shapValues
    
    classIdx :
        classifiers output a probablity for each class. ie healthy, cancer, ...
        used to select the shap values.
        
    # dataIndex :
    #     specifices the row to select in the confusionDataSetDict[confusionKey] dataFrame
    
    sampleId :
        example 'SRR14506725'
        sampleId is index into XDF and confusionDataSetDict[confusionKey]
        
    width is in inches
    '''
    title = createWaterfallTitle(config,  
                                 classIdx, 
                                 confusionDataSetDict, 
                                 sampleId,
                                 confusionKey)
    
    # shapValues are not dataframes. rows are indexs using integers
    cfDF = confusionDataSetDict[confusionKey]
    #sampleIdStr = cfDF.iloc[dataIndex].name
    dataIndexInShapValue = XDF.index.get_loc(sampleId)
    sv = shapValues [dataIndexInShapValue,:, classIdx]
    
    if debug :
        debugIndexing(
            config, 
            confusionDataSetDict,
            sv, # pass values for a single sample 
            classIdx,
            confusionKey,
            sampleId,
            XDF,
        )
        

    fig = plotWaterfall(
        sv=sv, 
        width=width, 
        title=title,
        maxNumberOfFeaturesToDisplay=maxNumberOfFeaturesToDisplay
        )
    
    return fig 

#################################################################################
def debugIndexing(
        config : ModelConfig, 
        confusionDataSetDict : dict[str, pd.DataFrame],
        sv : shap._explanation.Explanation, # pass values for a single sample 
        classIdx : int,
        confusionKey : str,
        #dataIndex : int,
        sampleId : str,
        XDF : pd.DataFrame,
        # width : Optional[float] = None,
        # # height : Optional[float] = None,
        # maxNumberOfFeaturesToDisplay : int = 10, # the shap waterfall plot default
        # debug : bool = False
    ):
    '''
    check for index mismatches
    
    assumes binary classifier. TODO test for mutli
    '''
    print(f'##### BEGIN DEBUG')
    diagnosis = confusionDataSetDict['diagnosis']
    print(f'diagnosis : {diagnosis}')
    print(f'confusionKey : {confusionKey}')
    cfDF = confusionDataSetDict[confusionKey]
    print(f'cfDF[sampleId] {cfDF.loc[sampleId]}')
    
    
    #
    # check sample values match shape original data values
    #
    sampleSeries = XDF.loc[sampleId ,:]
    print(f'sample feature values should match sv.data value')
    print(f'sample featues:\n{sampleSeries}\n')
    
    svDataNP = sv.data
    print(f'sv.Data : \n{svDataNP}')
    
    sortedDataNP = np.sort( svDataNP )
    sortedSampleNP = sampleSeries.sort_values().to_numpy()
    # print(f'sv.data ie original data values should match sample features\n{svDataNP}')
    # sortedSampleNP = sampleSeries.sort_values().to_numpy()
    # sortedSvValuesNP = np.sort( sortedSampleNP )
    # valuesAreEqual = np.array_equal(sortedSampleNP, sortedSvValuesNP)
    valuesAreEqual = np.array_equal(sortedDataNP, sortedSampleNP)
    if not valuesAreEqual:
        logger.error(f'test 1 fail: sample values do not match shape original data values')
    else :
        print(f'test 1 pass: sample features values match == shape original data values')
    
    #
    # does f(x) match model probablity
    #
    model = getModel(config, diagnosis)
    sampleXDF = XDF.loc[[sampleId] ,:]
    sampleXDF.shape
    modelPrediction = model.predict(sampleXDF)
    print(f'\nmodel.predict : {modelPrediction}\n')
    modelPredictProba = model.predict_proba(sampleXDF)
    print(f'model.predict_proba : modelPredictProba\n')
    
    baseValues = sv.base_values
    print(f'sv.base_values \n{baseValues}')
    
    svValuesNP = sv.values
    print(f'\nsv.values \n{svValuesNP}')

    fxFromShape = baseValues +  svValuesNP.sum()
 
    # if fxFromShape in modelPredictProba:
    print(f'classIdx : {classIdx}')
    if np.isclose(fxFromShape, modelPredictProba[0][classIdx]):
        print(f'test 2 passed, model probaliblity == shap f(x) ')
    else :
        print(f'test 2 failed: shap f(x) {fxFromShape} != model probablities {modelPredictProba}')

    # test to see if shape value are probablities or log-odds
    # they are probablities
    #
    # from scipy.special import expit # this is sigmoid() log-odds (logits) → probabilities.
    # logit = sv.base_values + sv.values.sum()
    # print(f'\nexpit(logit) {expit(logit)}')


    print("######## DEBUG END\n")
