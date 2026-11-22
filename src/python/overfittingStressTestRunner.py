# store type annotations as strings instead of evaluating them immediately.
# Prevents forward-reference errors
from __future__ import annotations

# 
# expored overfittingStressTestRunner.ipynb as python script
#


# %% [markdown]
# # Explicit overfitting stress tests (critical)
# Andrew E. Davidson   
# aedavids@ucsc.edu  
# Date: 2026-01-27.  
# Copyright (c) 2020-2023, Regents of the University of California All rights reserved.  
# https://polyformproject.org/licenses/noncommercial/1.0.0  
# 
# 
# **Prove core claim : What happens as feature count increases?**.  
# 

# %% [markdown]
# 
# ---
# ## How random forest classifiers work.  
# assume 100 samples, 50 features, and n_estimators = T
# 
# **bootstrap**.  
# for each tree 
# - a bootstrap sample of 100 samples is drawn with replacement
#     + some samples may appear multiple times, other may not appear at all
#     + on average 63.2 % of samples appear at least once
#     + about 36.8 samples are not used. they are the out-of-bag samples for that tree
#     + <span style="color:red;background-color:yellow;">each tree has its own oob</span>
#     + <span style="color:red;background-color:yellow;">across the forest every sample is oob for about 36% of trees</span>
# - the term 'estimator' refers to a single tree
#         + n_estimators = the number of trees
# 
# **What the OOB dataset reall is**. 
# There is not single oob data set
# 
# for each sample i:   
# * collect prediction from only the treess where sample i was OOB
# * aggregate those predictions (mean, prob, marjority vote, ...)
# 
# this produces one oob prediction per sample
# 
# you end up with n_samples OOB predictions
# 
# The OOB behaves like an internal validation set, not a true extranal hold-out. <span style="color:red;background-color:yellow;">It is not a replacement for an external test set</span>
# 
# It is good enough for model comparison, not acceptable as final clinical performance
#     
# 

# %% [markdown]
# ---
# ## What did Chen, Elife, do?
# used python. My guess is used sklear. 
# ```{python}
# RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)
# fit()
# # Access the OOB score
# oob_accuracy = rf_classifier.oob_score_
# 
# calculate other metrics
# oob_probs = rf.oob_prediction_
# oob_preds = np.argmax(oob_probs, axis=1) # Get the class label from probabilities
# 
# # oob_decision_function_ is per-class probablities
# roc_auc_score(y, rf.oob_decision_function_[:, 1])
# ```
# 
# ```
# Repeat 100 times:
#   - Train RF
#   - Compute OOB ROC/AUC
# Report:
#   - Mean ± SD
# ```

# %% [markdown]
# 
# ## What did Reggiardo, Nat biomedical engineering, do?
# 
# nat biomed paper used logistic regression it k-fold cross validation
# 

# %% [markdown]
# # Random Forest experimental design
# good base line.  
# 
# RFs.  
# - robust to irrelevant features
# - resuce variance via bagging
# - still suffer when p, number of features >> n, due to 
#     + split instablity
#     + spurious correlations
#     + limited sample diversity
# 
# **fixed**. 
# - same samples
# - same labels
# - same RF hyper parameters
# 
# **sweep**.  
# n_features = [10, 25, 50, 100, 250, 500, 1000, ...]
# 1. subset or augment features
# 2. train RF
# 3 compute
#     * OOB ROC/AUC
#     * Variance across repeats
#     
# Repeat whole process N = 50–100 times
# 
# **Expected results if our claim is true**.  
# demonstrate Feature scaling failure in classical ML vs robustness with distribution-aware simulation
# - RF OOB AUC
#     * plateaus or degrades as features increase
# - simulated data models
#     * continue to improve
#     * or stays stable with lower variance

# %%
defaultNotebookName="overfittingStressTest"


# %%
from attr import dataclass
from dataclasses import dataclass
from datetime import datetime, timedelta
# import glob
from IPython.display import display
# from IPython.display import HTML
import os
import numpy as np
import pandas as pd
# import pprint as pp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import sys
import time
from typing import Any

from argparse import ArgumentParser
#https://stackoverflow.com/a/3853776/4586180
#https://stackoverflow.com/a/3853776/4586180
from argparse import RawDescriptionHelpFormatter

# global variables use by cli
__all__ = []
__version__ = 0.1
__author__ = "Andrew Davidson aedavids@ucsc.edu"
__date__ = '2026-02-16'
__updated__ = '2026-02-16'


#
# when we run from bash sometime we get tons of warnings
#  warnings.warn(
# /private/home/aedavids/miniconda3/envs/fagazi/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: 
#     UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` 
#     to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
# 
# this some from configuring random Forest to use all avalable cores
# the sys admin has some sort of max number of threads limit so we
# do not have to worry about how to config this
#
# randomforestClassifiler with n_jobs=-1
# all of the warning make our log files unusable
#
# setting this here does not work
# set the export PYTHONWARNINGS="ignore::UserWarning:sklearn.utils.parallel"
# in the bash script
#
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*joblib workers.*",
    category=UserWarning,
)


# local imports
################################################################################
def appendToPYTHONPATH(paths : list[str]) -> None:
    ''' 
    add paths to PYTHONPATH and sys.path
    '''
    # TODO refactor into utility function
    for p in paths:
        # to be able to import our local python files we need to set the sys.path
        # https://stackoverflow.com/a/50155834
        if p not in sys.path:
            sys.path.append(p)
            
        pPath = os.environ["PYTHONPATH"]
        os.environ["PYTHONPATH"] = f"{pPath}:{p}"
    

    #logger.warning( f"Updated PYTHONPATH: {os.environ["PYTHONPATH"]}" )
    print( f"Updated PYTHONPATH: {os.environ["PYTHONPATH"]}" )
        

ORIG_PYTHONPATH = os.environ['PYTHONPATH']

extraRNA_PythonPath0 = '/private/home/aedavids/fagazi/src/python'
extraRNA_PythonPath1 = '/private/home/aedavids/extraCellularRNA' 
extraRNA_PythonPath2 = '/private/home/aedavids/extraCellularRNA/src'
extraRNA_PythonPath3 = '/private/home/aedavids/fagazi/src/jupyterNotebooks'
appendToPYTHONPATH([extraRNA_PythonPath0, extraRNA_PythonPath1, extraRNA_PythonPath2, extraRNA_PythonPath3]) 

from notebookUtils import  getNotebookName

from cliUtilities import configLogging
import logging

logger = logging.getLogger(__name__)


# from archive.dataTypes import  ModelConfig
# from archive.saveDataTypes import archive 
from data.loadCounts import loadGenesBySamplesData
# from models.configMLWorkflow import configBinaryLoadAndSplit, configBinaryCalculateSensitivityAtSpecificity
# from models.configMLWorkflow import configConfusionPanel, configEvaluate
# from models.configMLWorkflow import configROCPanel, configVolcanoPanel
# from models.mlWorkflow import trainRandomForest
from plots.sensitivity_at_specificity import computeSensitivitiesAtTargets
import stats.best_fit_simulator as bestFitSimulator
import stats.createBestFitSimulatorInput as createBestFitSimulatorInput 
import stats.findBestDistribution as findBestDistribution
import stats.formatSynthetics as formatSynthetics
import stats.generateSamples as generateSamples

notebook_start_time = time.time()
print("⏱️ Notebook started.")

# %%
# notebookName, notebookDir = getNotebookName(defaultNotebookName)
notebookName = 'overfittingStressTestRunner'
# print(f'notebookName: {notebookName}')
# print(f'notebookDir: {notebookDir}')
# notebookOutDir = f'{notebookDir}/{notebookName}.out'
# print(f'notebookOutDir: {notebookOutDir}')
# os.makedirs(notebookOutDir, exist_ok=True)

# %%
# _G : global use to insure we do not pick up globals in function by accident 
# durring refactoring

simulatedDataRootDir_G = "/private/groups/kimlab/aedavids/fagazi/simulatedData"

realDataRoot_G = "/private/groups/kimlab/data/elife"
realDataPath_G = f'{realDataRoot_G}/elife_all_norm_counts_2023-05-18.csv'
realMetaDataPath_G = f'{realDataRoot_G}/elife_scaled_metaData_2023-05-18.csv'

elifeDESeqResults = "/private/groups/kimlab/data/elife/deseq/binary/binaryDeseq.sh.out"

localCachePath_G = f"/scratch/aedavids/fagazi/simulatedData/{notebookName}"

theMeaningOfLife = 42

from baseCommandLine import BBaseCommandLine
################################################################################
class overfitRunnerCommandLine( BBaseCommandLine ):
    '''
    Handle the command line, usage and help requests.
    '''
    ###############################################################################
    def __init__( self, version, author, date, update ):
        '''
        Implement a parser to interpret the command line argv string using argparse.

        arguments:
            inOpst: a list of cli arguments. pass None if you want to use the the
                    true CLI arguments. pass a list if you want to use from a juypter notebook
        '''
        super().__init__(version, author, date, update )

    ###############################################################################
    def _build( self ):
        self.parser = ArgumentParser( description=self._getLicence(), formatter_class=RawDescriptionHelpFormatter )
    
        #
        # optional arguments
        #
        self.parser.add_argument('-f', "--featureCounts", required=False,
                                         default=[5, 10],  # default list
                                        type=int,
                                        nargs="+",                 # 1 or more ints
                                        help="list of the number of genes to test. Default = 5 10",
    )
        self.parser.add_argument( '-l', '--lfcThreshold', required=False, default=2.0, metavar="",   
                                            action='store', 
                                            type=float,
                                            help="log fold change cut off. Default = 2.0"
        )  

        self.parser.add_argument( '-p', '--padjThreshold', required=False, default=0.05, metavar="",   
                                            action='store', 
                                            type=float,
                                            help="adjust p-value cut off. default = 0.05"
        )   
        
        self.parser.add_argument( '-r', '--numRepeats', required=False,
                                default=50, metavar="", 
                                action="store",
                                type=int,  
                                help="number of time to repeat simulation and training loop for each featureCount. default = 50"
        )
        
        
        self.parser.add_argument( '-specifityTarget', '--specifityTarget', required=False, default=0.9, metavar="",   
                                            action='store', 
                                            type=float,
                                            help="calculate sensitivity @ specifityTarget. Default = 0.9"
        )  
        # self.parser.add_argument( '-c', '--category', 
        #                         default=None, metavar="", 
        #                         action="store",
        #                         type=str,  
        #                         help="category name. ex 'control'"
        # )
        
        # self.parser.add_argument( '-s', '--seed', 
        #                         default=None, metavar="", 
        #                         action="store",
        #                         type=int,  
        #                         help="value used to initialized random number generator"
        # )
        

        # allow zero or more var args. ie user defined args
        #self.parser.add_argument('vargs', nargs="*", action='store')
    
        #
        # group required arguments. This will create a better help message
        # make sure to set required=True
        #
        self.requiredArg = self.parser.add_argument_group( 'required arguments' )

       

        self.requiredArg.add_argument( '-c', '--cancerType', required=True, 
                                                # default=".", 
                                                # metavar="",   
                                                action='store', 
                                                type=str,
                                                help='valid types are "Colorectal", "Esophagus", "Liver", "Lung", "Stomach" '
        ) 

        
# ################################################################################ 
# def _parseCLI(logger : logging.Logger, inCommandLineArgsList : list[str]) -> tuple[str,str,str]:
#     '''
#     TODO
#     '''

#     cli = overfitRunnerCommandLine(
#             version=__version__ , 
#             author=__author__ ,
#             date=__date__, 
#             update=__updated__
#     )
    
#     if inCommandLineArgsList is None:
#         cli.parse()
#     else:
#         cli.parse( inCommandLineArgsList )

#     logger.warning(f'command line arguments : {cli.args}')
#     # intCountsPath  = cli.args.intCountsPath
#     # realCountsPath = cli.args.realCountsPath 
#     # outFilePath    = cli.args.out
    
#     return (cli.args.diagnosis)
      

# %%
def loadElifeBinaryClassificationDataDepricated(
        XDFPath : str,
        yDFPath : str, 
        diagnosis : str
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]: 
    '''
    
    will select "Healthy donor" + diagnosis
    
    arguments:
        XDFPath
        
        yDFPath
        
        diagnosis
            valid values "Colorectal Cancer", "Esophagus Cancer", "Liver Cancer", "Lung Cancer", "Stomach Cancer"
            
    returns
        XDF with sampleId as index
        
        ySeries and ordered categorical series with class "Healthy_donor" set a the negative class.
        index will be sample Id
        
        metaDataDF
    '''
    
    # loadGenesBySamplesData fill fix diagnosis name
    # example 'Healthy donor' -> 'Healthy_donor'
    binaryClasses = ['Healthy donor', diagnosis]
    realXDF, realYSeries, realMetaDF = loadGenesBySamplesData(
        categories=binaryClasses,
        xDFPath=XDFPath,
        yDFPath=yDFPath,
        yCategoryColName="diagnosis",
        yRowIdName='sample_id',
        withFullMetadata=True
    )

    # create a series with categorical values
    # reset() creates a strictly positional index
    # and guaranteed to match the categorical array length.
    tmpRealY = realYSeries.reset_index(drop=True)
    
    # our diagnosis need to match the realYSeries value, 
    # we need to insure we label the negative class as zero
    fixedDiagnosis =  [d.replace(" ", "_") for d in binaryClasses]
    realYSeries = pd.Series(
                    pd.Categorical(
                        tmpRealY,
                        categories=fixedDiagnosis, 
                        ordered=True),
                        index=realXDF.index
                    )    
    return (realXDF, realYSeries, realMetaDF)
    
def testLoadElifeBinaryClassificationData() :
    XDF, ySeries, metaDF = loadElifeBinaryClassificationDataDepricated(
        XDFPath=realDataPath_G,
        yDFPath=realMetaDataPath_G,
        diagnosis="Colorectal Cancer"
    )
    print(f"XDFPath.shape : {XDF.shape}")
    print(f"ySeries.shape : {ySeries.shape}")
    # print(ySeries[0:5])
    print(f"metaDF.shape : {metaDF.shape}")
    # print(f"metaDF.iloc[0:5] : {metaDF.iloc[0:5, 0:5]}")
    
    return (XDF, ySeries, metaDF )



# %%
def loadElifeClassificationData(
        XDFPath : str,
        yDFPath : str, 
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]: 
    '''
    
    
    
    arguments:
        XDFPath
        
        yDFPath
        
            
    returns
        XDF with sampleId as index
        
        ySeries and ordered categorical series with class "Healthy_donor" set a the negative class.
        index will be sample Id
        
        metaDataDF
    '''
    
    # loadGenesBySamplesData fill fix diagnosis name
    # example 'Healthy donor' -> 'Healthy_donor'
    # binaryClasses = ['Healthy donor', diagnosis]
    diagnosis = ['Healthy donor', "Colorectal Cancer", "Esophagus Cancer", "Liver Cancer", "Lung Cancer", "Stomach Cancer"]

    realXDF, realYSeries, realMetaDF = loadGenesBySamplesData(
        categories=diagnosis,
        xDFPath=XDFPath,
        yDFPath=yDFPath,
        yCategoryColName="diagnosis",
        yRowIdName='sample_id',
        withFullMetadata=True
    )

    # create a series with categorical values
    # reset() creates a strictly positional index
    # and guaranteed to match the categorical array length.
    tmpRealY = realYSeries.reset_index(drop=True)
    
    # our diagnosis need to match the realYSeries value, 
    # we need to insure we label the negative class as zero
    fixedDiagnosis =  [d.replace(" ", "_") for d in diagnosis]
    realYSeries = pd.Series(
                    pd.Categorical(
                        tmpRealY,
                        categories=fixedDiagnosis, 
                        ordered=True),
                        index=realXDF.index
                    )    
    return (realXDF, realYSeries, realMetaDF)


# globals
realXDF_G, realYSeries_G, realMetaDF_G = loadElifeClassificationData(
                            XDFPath=realDataPath_G,
                            yDFPath=realMetaDataPath_G,
                        )

# %%
def selectDifferentialExpressedGenes(
        resultPath : str,
        lfcThreshold : float,
        padjThreshold : float,
        
    ) -> pd.DataFrame:
    ''' 
    
    arguments:
        resultPath path to a DESeq2 output file. Assumes there are extra header lines
        
        example of header
            # mean of normalized counts for all samples 
            # log2 fold change (MLE): diagnosis Colorectal Cancer vs Healthy donor 
            # standard error: diagnosis Colorectal Cancer vs Healthy donor 
            # Wald statistic: diagnosis Colorectal Cancer vs Healthy donor 
            # Wald test p-value: diagnosis Colorectal Cancer vs Healthy donor 
            # BH adjusted p-values 
            # design: ~gender + diagnosis 
            # "name","baseMean","log2FoldChange","lfcSE","stat","pvalue","padj"
            
        lfcThreshold :
            select genes with >= absolute( log fold change threshold )
            
        padjThreshold :
            select genes with adjust p-value <= threshold
            
    returns
        pandas data frame with columns
        "name","baseMean","log2FoldChange","lfcSE","stat","pvalue","padj"
        sorted in descending order by abs(log2FoldChange)
    '''
    print(resultPath)
    #typical header section use skip rows
    
    # mean of normalized counts for all samples 
    # log2 fold change (MLE): diagnosis Colorectal Cancer vs Healthy donor 
    # standard error: diagnosis Colorectal Cancer vs Healthy donor 
    # Wald statistic: diagnosis Colorectal Cancer vs Healthy donor 
    # Wald test p-value: diagnosis Colorectal Cancer vs Healthy donor 
    # BH adjusted p-values 
    # design: ~gender + diagnosis 
    # "name","baseMean","log2FoldChange","lfcSE","stat","pvalue","padj"
    
    deseqDF = pd.read_csv(resultPath, skiprows=7) 
    
    #
    # find statistically signifigant genes
    #
    selectSignificantRowsPS = deseqDF.loc[:,"padj"] <= padjThreshold
    # logger.info("number of genes with padj <= {} : {}".format(padjThreshold,
    #                                                 selectSignificantRowsPS.sum()))

    #
    # use absolute value of log fold change to select best 
    # biologically signifigant genes
    # 
    significantDF = deseqDF.loc[selectSignificantRowsPS, :]
    absPS = significantDF['log2FoldChange'].abs()
    significantDF2 = significantDF.assign(absLog2FoldChange=absPS)    

    selectBestUpRegulatedRows = significantDF2.loc[:, 'absLog2FoldChange'] >= lfcThreshold
    # significantDF3 = significantDF2.loc[selectBestUpRegulatedRows, :]
    retDF = significantDF2.loc[selectBestUpRegulatedRows, :]
    # significantDF3 = significantDF3.sort_values( by = ['baseMean'], ascending=False)
    retDF = retDF.sort_values( by = ['absLog2FoldChange'], ascending=False)
    
    return retDF   
    
    

def testSelectDifferentialExpressedGenes() :  
    retDict = dict()
    for diagnosis in ["Colorectal", "Esophagus", "Liver", "Lung", "Stomach"] :
        dDict = dict()
        dDict['padj'] = []
        dDict['count'] = []
        for padj in [0.001, 0.01, 0.5] :
            aedwip = f"{diagnosis}_vs_Healthy_donor.results"
            rPath = f'{elifeDESeqResults}/{diagnosis}/{aedwip}'
            retDF = selectDifferentialExpressedGenes(
                resultPath=rPath,
                lfcThreshold=2.0,
                padjThreshold=padj
                )
            dDict['padj'].append(padj)
            dDict['count'].append(retDF.shape[0])
            
        retDict[diagnosis] = dDict
    
    return retDict
    
# aedwipDict = testSelectDifferentialExpressedGenes()
# print(aedwipDF.shape)
# aedwipDF.head()

# print(f'choose a class with lots of features')
# print(f'try liver. binary classification results where good, TODO check performance against roman and chen')
# aedwipDict

# %%
def createTestGeneList() -> list[str]:
    diagnosis = "Liver"
    resultsFileName = f"{diagnosis}_vs_Healthy_donor.results"
    resultFilePath = f'{elifeDESeqResults}/{diagnosis}/{resultsFileName}'

    resultsDF = selectDifferentialExpressedGenes(
                    resultPath = resultFilePath,
                    lfcThreshold=2.0,
                    padjThreshold=0.001
    )
    retList = resultsDF["name"][0:5].to_list()

    return retList

testLiverGeneList = createTestGeneList()
# testLiverGeneList

# %%
def generateSamplesPipelineStep2(
        outDir : str,
        genes : list[str],
        XPath : str,
        yPath : str,
    ) -> tuple[pd.DataFrame, str]:
    '''
    Docstring for generateSamplesPipelineStep2
    '''
    logger.info("BEGIN")
    cbfsiOut = f'{outDir}/createBestFitSimulatorInput.py.out'
    os.makedirs(cbfsiOut, exist_ok=True)
    outFilePath = f'{cbfsiOut}/testDistribution.csv'
    if not os.path.isfile(outFilePath) :        
        # load the data exactly the same way as main()
        # to much hacking to get in memory version to work
        countDF = pd.read_csv(XPath, index_col=0) 
        metaDF = pd.read_csv(yPath) 
        metaDF = metaDF.set_index('sample_id')

        formatedDF = createBestFitSimulatorInput.run(logger, genes, countDF, metaDF)
        # print(formatedDF.head())
        
        logger.warning(f'writting to {outFilePath}')
        formatedDF.to_csv(outFilePath, index=False)
        
    else :
        logger.warning(f"skipping step 2. output file exists load from  {outFilePath}")
        formatedDF = pd.read_csv(outFilePath)
        # print(formatedDF.head())

    logger.info("END")
    
    return (formatedDF, outFilePath)

# %%
def generateSamplesPipelineStep3(
    outDir : str,
    formatedPath : str
    ) :
    '''
    Docstring for generateSamplesPipelineStep3
    '''
    logger.info("BEGIN")
    
    # formatedDF was written to disk
    # it is a lot easier to call main. 
    # the simulator is does a lot of work
    outPrefix = f'{outDir}/best_fit_simulator.py.out/'
    
    # if downstream file exist no need to recompute
    downStreamFilePath = f'{outPrefix}/best_models.csv'
    if not os.path.isfile(downStreamFilePath) :
        prefixPath = f'{outPrefix}Q-Q-Plots'
        inCommandLineArgsList = [
            '--input', formatedPath,
            '--out-prefix', outPrefix,
            '--n-synth', '0',
            '--seed',  '42',
            '--plots-prefix', prefixPath
        ]
        try :
            bestFitSimulator.main( inCommandLineArgsList )
        except SystemExit as e:
            logger.info(f"bestFitSimulator.main() called sys.exit({e.code}); ignoring in notebook")  
    
    else :
        logger.warning(f"skipping step 3. output file exists load from  {downStreamFilePath}")


    logger.info("END")


# %%
def generateSamplesPipelineStep4(
    outDir : str,
    formatedPath : str
    ) :
    '''
    Docstring for generateSamplesPipelineStep4
    '''
    logger.info("BEGIN")
    
    # formatedDF was written to disk
    # it is a lot easier to call main. 
    # the simulator is does a lot of work
    outPrefix = f'{outDir}/best_fit_simulator.py.out_ZINB_NB_int/'
    
    # if down stream files exist no need to recompuate
    downStreamFilePath = f'{outPrefix}/best_models.csv'
    if not os.path.isfile( downStreamFilePath ) :
        prefixPath = f'{outPrefix}Q-Q-Plots'
        inCommandLineArgsList = [
            '--input', formatedPath,
            '--out-prefix', outPrefix,
            '--n-synth', '0',
            '--seed',  '42',
            '--plots-prefix', prefixPath,
            '--enable-discrete'
        ]
        try :
            bestFitSimulator.main( inCommandLineArgsList )
        except SystemExit as e:
            logger.info(f"bestFitSimulator.main() called sys.exit({e.code}); ignoring in notebook")  
    
    else :
        logger.warning(f"skipping step 4. output file exists load from  {downStreamFilePath}")


    logger.info("END")

# %%
def generateSamplesPipelineStep5(
    outDir : str,
    ) :
    '''
    Docstring for generateSamplesPipelineStep5
    '''
    logger.info("BEGIN")
    
    # it is a lot easier to call main. 
    # outDir= f'{outDir}/findBestDistribution.py.out'
    outFilePath = f'{outDir}/findBestDistribution.py.out/bestPropDist.csv'
    if not os.path.isfile(outFilePath) :
        intCountPath = f'{outDir}/best_fit_simulator.py.out_ZINB_NB_int/best_models.csv'
        realCountPath = f'{outDir}/best_fit_simulator.py.out/best_models.csv'
        inCommandLineArgsList = [
            '--intCountsPath', intCountPath,
            '--realCountsPath', realCountPath,
            '--out', outFilePath,
        ]
        try :
            findBestDistribution.main( inCommandLineArgsList )
        except SystemExit as e:
            logger.info(f"findBestDistribution.main() called sys.exit({e.code}); ignoring in notebook")   
    else :
        logger.warning(f"skipping step 5. output file exists  {outFilePath}")
    

    logger.info("END")

# %%
def generateSamplesPipelineStep6(
    outDir : str,
    numSamplesPerClass : int
    ) :
    '''
    Docstring for generateSamplesPipelineStep6
    
    num 
    '''
    logger.info("BEGIN")
    
    # it is a lot easier to call main. 
    # outDir= f'{outDir}/findBestDistribution.py.out'
    outFilePath = f'{outDir}/generateSamples.py.out/simulated.csv'
    if not os.path.isfile(outFilePath) :
        bestPath = f'{outDir}/findBestDistribution.py.out/bestPropDist.csv'
        inCommandLineArgsList = [
            '--best', bestPath,
            '--outPath', outFilePath,
            '--num', str(numSamplesPerClass),
            '--seed', '42'
        ]
        try :
            generateSamples.main( inCommandLineArgsList )
        except SystemExit as e:
            logger.info(f"generateSamples.main() called sys.exit({e.code}); ignoring in notebook")  
            
    else :
        logger.warning(f"skipping step 6. output file exists  {outFilePath}")


    logger.info("END")

# %%
def generateSamplesPipelineStep7(
    outDir : str,
    ) -> tuple[str,str] :
    '''
    Docstring for generateSamplesPipelineStep6
    
    returns (pathX.csv, path.csv)
    '''
    logger.info("BEGIN")
    
    # it is a lot easier to call main. 
    tmpOutDir= f'{outDir}/formatSynthetics.py.out'
    XPath = f'{tmpOutDir}/X.csv'
    yPath = f'{tmpOutDir}/y.csv'
    if ( not os.path.isfile(XPath) ) or ( not os.path.isfile(yPath) ):
        syntheticsPath = f'{outDir}/generateSamples.py.out/simulated.csv'
        inCommandLineArgsList = [
            '--sampleIdPrefix', "s",
            '--synthetics', syntheticsPath,
            '--outDir',tmpOutDir ,
        ]
        try :
            formatSynthetics.main( inCommandLineArgsList )
        except SystemExit as e:
            logger.info(f"formatSynthetics.main() called sys.exit({e.code}); ignoring in notebook")  
            
    else :  
        logger.warning(f"skipping step 7. output file exists  {XPath}")
        logger.warning(f"skipping step 7. output file exists  {yPath}")
        
    logger.info("END")
    return( XPath, yPath)



# %%
def makeCategoricalWithReference(
    yDF: pd.DataFrame,
    negativeClass: str
    ) -> pd.Series:
    """
    Return a Series with dtype 'category' (ordered) where negativeClass is the
    first category (reference level). Preserves the original index.
    
    yDF must have columns 'sampleIds' and 'Classes'
    """
    assert 'sampleIds' in yDF.columns,  "error expected col name 'sampleIds' "
    assert 'Classes'   in yDF.columns,  "error expected col name 'Classes' "
    
    # All levels present in the data (preserve observed order)
    levels = list(yDF['Classes'].unique())

    # Move negativeClass to the front if present
    if negativeClass in levels:
        levels.remove(negativeClass)
        levels = [negativeClass] + levels

    retSeries =  pd.Series(
                    pd.Categorical(
                        yDF['Classes'], 
                        categories=levels, 
                        ordered=True),
                    
                    index=yDF['sampleIds']
                )
    
    return retSeries

# %%
def sampleGenerationPipeline(
        outDir : str,
        genes : list[str],
        XPath : str,
        yPath : str,
        numSamplesPerClass : int,
        negativeClassName="Healthy_donor",
    ) -> tuple[pd.DataFrame, pd.Series]:
    ''' 
    ref src/bin/generateSamplesPipeline.sh

    arguments :

        outDirPath 
        
    returns :
        a tuple withsimulatedXDF and ySeries
        ySeries has dtype 'category' (ordered) where negativeClass is the
        first category (reference level).
            
    '''
    # we have a list of genes no need to run step 2
    
    # if the pipeline was run
    # outXPath and outyPath contain the simulated samples
    tmpOutDir= f'{outDir}/formatSynthetics.py.out'
    outXPath = f'{tmpOutDir}/X.csv'
    outyPath = f'{tmpOutDir}/y.csv'
    
    if ( not os.path.isfile(outXPath) ) or ( not os.path.isfile(outyPath) ):
        formatedDF, formatedPath = generateSamplesPipelineStep2(outDir, genes, XPath, yPath)

        generateSamplesPipelineStep3(outDir, formatedPath, )

        generateSamplesPipelineStep4(outDir, formatedPath )

        generateSamplesPipelineStep5(outDir )

        generateSamplesPipelineStep6(outDir, numSamplesPerClass )

        #XPath, yPath = generateSamplesPipelineStep7(outDir) 
        t = generateSamplesPipelineStep7(outDir) 
        
    else :
        logger.warning(f"skipping pipeline. output file exists  {outXPath}")
        logger.warning(f"skipping pipeline. output file exists  {outyPath}")

        
    # XDF = pd.read_csv(XPath)
    # tmpYDF = pd.read_csv(yPath)
    XDF = pd.read_csv(outXPath)
    tmpYDF = pd.read_csv(outyPath)
    
    ySeries = makeCategoricalWithReference(tmpYDF, negativeClassName)
    
    return (XDF, ySeries)


def testSampleGenerationPipeline():
    testOutDir = f'{localCachePath_G}/testSampleGenerationPipeline'
    # run the pipeline to generate sample for all 6 classs
    ret = sampleGenerationPipeline(
        outDir=testOutDir,
        genes=testLiverGeneList,
        XPath=realDataPath_G,
        yPath=realMetaDataPath_G,
        numSamplesPerClass=5,
        negativeClassName="Healthy_donor"
    )
    
    testXDF, testYSeries = ret
    print(f'\n\n################')
    print(f'testXDF.shape : {testXDF.shape}')
    print(testXDF.iloc[0:5, 0:5])
    print(f' testYSeries.cat.categories : {testYSeries.cat.categories}')
    print(testYSeries[0:5])
    
# testSampleGenerationPipeline()


# %%
@dataclass(frozen=True) # frozen == read only
class StratifiedFold :
    XTrainDF : pd.DataFrame
    XTestDF : pd.DataFrame
    yTrainSeries : pd.Series
    yTestSeries : pd.Series

def splitAndStratify(XDF : pd.DataFrame,
                     ySeries: pd.Series, 
                     kFolds : int,
                     random_state : int=theMeaningOfLife
                     ) -> list[StratifiedFold] :
                            
    '''
    Docstring for splitAndStratify
    '''
    if not XDF.index.equals(ySeries.index):
        raise ValueError("XDF and ySeries must have identical indices (same order).")
    
    retList : list[StratifiedFold] = []
    skf = StratifiedKFold(n_splits=kFolds, shuffle=True, random_state=random_state)

    for _, (trainIndex, testIndex) in enumerate(skf.split(XDF, ySeries), start=1):
        # Get the data for the current train and test sets
        XTrainDF  = XDF.iloc[trainIndex] 
        XTestDF   = XDF.iloc[testIndex]
        yTrainSeries = ySeries.iloc[trainIndex]
        yTestSeries  = ySeries.iloc[testIndex]
        
        d = StratifiedFold(XTrainDF, XTestDF, yTrainSeries, yTestSeries)
        retList.append(d)
        
    return retList

# %%
def validateClasses( cancerTypes : list[str] ) -> None :
    ''' 
    raises ValueError if invalid
    '''
    validCancerTypes=["Colorectal", "Esophagus", "Liver", "Lung", "Stomach"]
    for c in cancerTypes:
        if not c in validCancerTypes :
            emsg = f'{c} is not in {validCancerTypes}'
            logger.error(emsg)
            raise ValueError(emsg)

# %%
def differentialExpressionHelper(
        diagnosis : str,
        lfcThreshold : float,
        padjThreshold : float,
    ) -> tuple[pd.DataFrame, float]:
    ''' 
    easy of use for selectDifferentialExpressedGenes()
    
    AEDWIP elifeDESeqResults is hard code global
    
    returns 
        (resultsDF, totalNumFeatures)
    '''
    resultsFileName = f"{diagnosis}_vs_Healthy_donor.results"
    resultFilePath = f'{elifeDESeqResults}/{diagnosis}/{resultsFileName}'
    logger.info(f'reading deseq results file {resultFilePath}')
    resultsDF = selectDifferentialExpressedGenes(resultFilePath, lfcThreshold, padjThreshold)
    logger.info(f'{diagnosis} : num genes : {resultsDF.shape[0]}')
    totalNumFeatures = resultsDF.shape[0]
    
    return (resultsDF, totalNumFeatures)

# %%
def selectRealDataSet(
    resultsDF : pd.DataFrame,
    numFeatures : int,
    realXDF : pd.DataFrame,
    diagnosis : str,
    realYSeries : pd.Series,
    realMetaDF : pd.Series
    ) -> tuple[list[str], pd.DataFrame, pd.Series, pd.DataFrame] :
    ''' 
    
    returns
        (geneList, tmpRealXDF, tmpRealYSeries, tmpRealMaskDF)
    '''
    # select genes of interst
    tmpDF = resultsDF.head(numFeatures)
    geneList = tmpDF["name"].to_list()
    logger.info(f'numFeatures : {numFeatures} genelist : {geneList}')
    
    # select the real samples and features for diagnosis and healthy
    diagnosisLabel = f'{diagnosis}_Cancer'
    mask = realYSeries.isin( [diagnosisLabel, "Healthy_donor"] )
    logger.info(f'mask.shape : {mask.shape}')
    tmpRealYSeries = realYSeries[mask]
    # selectMaskRows = realMetaDF.loc[:, "sample_id"].isin(mask.index)
    # tmpRealMaskDF =  realMetaDF.loc[selectMaskRows.index, :]
    tmpRealMaskDF = realMetaDF.loc[realMetaDF["sample_id"].isin(realYSeries.index[mask]), :]
    logger.info(f'aedwip diagnosis : {diagnosisLabel} tmpRealYSeries.shape : {tmpRealYSeries.shape} tmpRealMaskDF.shape : {tmpRealMaskDF.shape}')
    sampleIds = tmpRealYSeries.index
    tmpRealXDF = realXDF.loc[sampleIds, geneList]
    logger.info(f'diagnosis : {diagnosis} tmpRealXDF.shape : {tmpRealXDF.shape} tmpRealYSeries : {tmpRealYSeries.shape}')
    
    ret = (geneList, tmpRealXDF, tmpRealYSeries, tmpRealMaskDF)
    return ret 

# %%
def saveTmp(
        localCachePath : str,
        tmpRealXDF : pd.DataFrame,
        numFeatures : int,
        tmpRealMaskDF : pd.DataFrame,
        repeatIdx : int,
        foldIdx : int,
        diagnosis : str, 
        lfcThreshold : float,
        padjThreshold : float,
    ) -> tuple[str, str, str] :
    ''' 
    
    returns 
        (tmpOutPath, localCachePathRealXPath, localCachePathRealMaskPath)
    '''
    # save tmpReal to disk so that we can generate samples
    # create large numbers of files in a single directory can degrade i/o performance
    tmpOutPath = f'{localCachePath}/crossFitSimulatedToRealRF/{diagnosis}/n{numFeatures}/{diagnosis}.n:{numFeatures}.rIdx:{repeatIdx}.fIdx:{foldIdx}lfc:{lfcThreshold}.padj:{padjThreshold}'
    os.makedirs(tmpOutPath, exist_ok=True)
    
    localCachePathRealXPath = f'{tmpOutPath}/tmpRealX.csv'
    logger.debug(f'saving {localCachePathRealXPath}')
    tmpRealXDF.index.name = 'sample_id'
    tmpRealXDF = tmpRealXDF.transpose()
    
    tmpRealXDF.to_csv(localCachePathRealXPath) # , index=False
    
    localCachePathRealMaskPath = f'{tmpOutPath}/tmpRealMask.csv'
    logger.debug(f'saving : {localCachePathRealMaskPath}')
    # tmpRealMaskDF.index.name = 'sample_id'
    tmpRealMaskDF.to_csv(localCachePathRealMaskPath, index=False)
    
    ret = (tmpOutPath, localCachePathRealXPath, localCachePathRealMaskPath)
    return ret


# %%
def generateSamplesHelper(
        geneList : list[str],
        tmpOutPath : str, 
        localCachePathRealXPath : str,
        localCachePathRealMaskPath : str,
    ) -> tuple[pd.DataFrame, pd.Series]:
    '''
    
    returns 
        (simulatedXDF, simulatedYSeries)
    '''
    numClasses = 2
    numSamplesPerClass = numClasses * 10 * len(geneList)
    logger.info(f'numClasses : {numClasses} len(geneList) : {len(geneList)} numSamplesPerClass : {numSamplesPerClass}')
    simulatedXDF, simulatedYSeries = sampleGenerationPipeline(
                                            tmpOutPath,
                                            geneList,
                                            localCachePathRealXPath,
                                            localCachePathRealMaskPath,
                                            numSamplesPerClass,
                                            )
    ret = (simulatedXDF, simulatedYSeries)
    return ret

# %%
def generateUniqRepeatableSeed(
        diagnosis : str,
        ySeries : pd.Series,
        baseSeed : int,
        nFeatures : int,
        repeatIdx : int,
        # foldIdx : int,
    ) -> int :
    ''' 
    “How sensitive are results to the randomness of RF + simulation?”
    
    we need to vary the seed across (cancerType, nFeatures, repeatIndex)
      - deterministic results (rerunning gives identical numbers)
      - independent repeats (each repeat explores RF randomness)
      - honest variance estimates
      
    '''
    className = f'{diagnosis}_Cancer'
    diagnosisClassValue = ySeries.cat.categories.get_loc(className)
    
    rfSeed =  baseSeed \
                + diagnosisClassValue * 10_000 \
                + nFeatures * 1_000 \
                 + repeatIdx 
    
    return rfSeed

# %%
def trainRFClassifier(
        simulatedXDF : pd.DataFrame,
        simulatedYSeries : pd.Series,
        seed : int,
    ) -> RandomForestClassifier :
    '''
    Docstring for trainRFClassifier
    '''
    #
    # n_estimators = 1000. What I arbitrarily choose for all the other
    # binary and multiclassification test
    # reviewer-safe justification "“A large number of trees was used to stabilize probability estimates.”"
    #
    # max_features = "sqrt" This is the default
    # 
    # min_samples_leaf = 2  # or even 5. default is 1
    # minimum number of samples required to be at a leaf node
    # With n≈30–50 positives, single-sample leaves are noise
    #
    # bootstrap = True (default)
    # ob_score = False (since you’re using CV of real hold ot, not OOB)

    rfModel = RandomForestClassifier(
            n_estimators=1000,
            max_features="sqrt",
            min_samples_leaf=2,
            #class_weight="balanced", our generated samples are balanced
            bootstrap=True,
            n_jobs=-1, # use all avalible processors to run in parrallel
            random_state=seed,
            )


    tmpDF = simulatedXDF.transpose()
    rfModel.fit(tmpDF, simulatedYSeries)   
    
    return rfModel 

# %%
def makeBinaryOrderedCategorical(
    y: pd.Series,
    negativeClass: str,
    positiveClass: str,
) -> pd.Series:
    """
    Convert an ordered categorical Series (possibly with many categories) into an
    ordered *binary* categorical Series containing only {negativeClass, positiveClass}.

    Raises if any values are outside those two classes.
    """
    # Preserve index
    yStr = y.astype("string")

    allowed = {negativeClass, positiveClass}
    badMask = ~yStr.isin(allowed) & ~yStr.isna()
    if bool(badMask.any()):
        badVals = sorted(set(yStr[badMask].dropna().unique().tolist()))
        raise ValueError(f"Found values outside the two classes: {badVals}")

    return pd.Series(
        pd.Categorical(
            yStr,
            categories=[negativeClass, positiveClass],
            ordered=True,
        ),
        index=y.index,
        name=y.name,
    )

# %%
def saveFold(
    localCachePath : str,
    realMetaDF : pd.DataFrame,
    repeatIdx : int,    
    foldIdx : int,
    fold : StratifiedFold,
    numFeatures : int,
    diagnosis : str,
    lfcThreshold : float,
    padjThreshold : float,
    ) -> tuple[str, str, str]:
    '''
    Docstring for saveFold
    
    returns tmpOutPath, localCachePathFoldXTrainPath, localCachePathFoldTrainMaskPath
    '''
    foldXTrainDF = fold.XTrainDF
    selectSamplesMask = realMetaDF['sample_id'].isin( foldXTrainDF.index )
    foldTrainMaskDF = realMetaDF.loc[selectSamplesMask, :]
    t = saveTmp(localCachePath, foldXTrainDF, numFeatures, foldTrainMaskDF, 
                    repeatIdx, foldIdx, 
                    diagnosis, lfcThreshold, padjThreshold)
    tmpOutPath, localCachePathFoldXTrainPath, localCachePathFoldTrainMaskPath = t
    
    return (tmpOutPath, localCachePathFoldXTrainPath, localCachePathFoldTrainMaskPath)
                

# %%
# def calculateMetricsDeprecicated(
#         rfModel : RandomForestClassifier,
#         foldXTestDF : pd.DataFrame,
#         foldBinaryYTestSeries : pd.Series,
#         positiveClass : str,
#         specifityTargets : list[float],
#     )  -> tuple[float, pd.DataFrame]: 
#     '''
#     TODO
    
#     returns 
#         (accuracy, SensitivityDF.loc[:, ['Sensitivity (TPR)', 'Achieved Specificity (TNR)', 'Approx. Threshold', 'auc']])
#     '''
#     yTrue = foldBinaryYTestSeries           
#     yPred = rfModel.predict(foldXTestDF)
    
#     accuracy = accuracy_score(yTrue, yPred)
                
#     #
#     # calculate sensitiviy@specificity
#     #
    
#     # rfModel.classes_ will be in alphabetic order. 
#     # make sure we select the predictions for the positive class
#     posIdx = list(rfModel.classes_).index(positiveClass)
#     yScoreNP = rfModel.predict_proba(foldXTestDF)[:, posIdx]
    
#     # get the encode labels, ie integer values for labels
#     encodedYSeries = yTrue.cat.codes
#     # SensitivityDF columns = ['Target Specificity', 'Sensitivity (TPR)', 
#     #                               'Achieved Specificity (TNR)', 
#     #                               'Approx. Threshold', 'auc']
#     SensitivityDF, _ = computeSensitivitiesAtTargets(
#                             yTrue=encodedYSeries, 
#                             yScore=yScoreNP, 
#                             targets=specifityTargets,
#                             interpolate=True)    
    
#     retDF = SensitivityDF.loc[:, ['Sensitivity (TPR)', 'Achieved Specificity (TNR)', 'Approx. Threshold', 'auc']]
#     return (accuracy, retDF)

def calculateMetrics(
        rfModel : RandomForestClassifier,
        foldXDF : pd.DataFrame,
        foldBinaryYSeries : pd.Series,
        positiveClass : str,
        specifityTargets : list[float],
    )  -> tuple[float, pd.DataFrame]: 
    '''
    TODO
    
    returns 
        (accuracy, SensitivityDF.loc[:, ['Sensitivity (TPR)', 'Achieved Specificity (TNR)', 'Approx. Threshold', 'auc']])
    '''
    yTrue = foldBinaryYSeries           
    yPred = rfModel.predict(foldXDF)
    
    accuracy = accuracy_score(yTrue, yPred)
                
    #
    # calculate sensitiviy@specificity
    #
    
    # rfModel.classes_ will be in alphabetic order. 
    # make sure we select the predictions for the positive class
    posIdx = list(rfModel.classes_).index(positiveClass)
    yScoreNP = rfModel.predict_proba(foldXDF)[:, posIdx]
    
    # get the encode labels, ie integer values for labels
    encodedYSeries = yTrue.cat.codes
    # SensitivityDF columns = ['Target Specificity', 'Sensitivity (TPR)', 
    #                               'Achieved Specificity (TNR)', 
    #                               'Approx. Threshold', 'auc']
    SensitivityDF, _ = computeSensitivitiesAtTargets(
                            yTrue=encodedYSeries, 
                            yScore=yScoreNP, 
                            targets=specifityTargets,
                            interpolate=True)    
    
    retDF = SensitivityDF.loc[:, ['Sensitivity (TPR)', 'Achieved Specificity (TNR)', 'Approx. Threshold', 'auc']]
    return (accuracy, retDF)

# %%

def formatTimedelta(delta: timedelta) -> str:
    """
    Format a timedelta as HH:MM:SS
    """
    totalSeconds: int = int(delta.total_seconds())

    hours: int = totalSeconds // 3600
    minutes: int = (totalSeconds % 3600) // 60
    seconds: int = totalSeconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
def elapsedTimeStr (
    startTime : datetime,
    endTime : datetime,
    ) -> str :
    '''
    Docstring for foo
    
    :param startTime: Description
    :type endTime: datetime
    '''
    deltaTime: timedelta = endTime - startTime
    
    return formatTimedelta( deltaTime )
    

# %%
# def createFoldMetricRowDictDepreciated(
#         diagnosis : str,
#         numFeatures : int,
#         repeatIdx : int,
#         repeatSeed : int,
#         foldIdx : int, 
#         accuracy : float, 
#         sensitivityDF : pd.DataFrame,
#     ) -> list[dict[str, Any]] :
#     '''
#     refactored code to improve readablity
    
#     calculate fold metrics, replace infinite values with np.nan
#     pandas mean() and std() will ignore these valuess
#     '''          
#     foldRowMetricDict : dict[str, Any] = {
#         "diagnosis": diagnosis,
#         'numFeatures' : numFeatures,
#         "repeatId" : repeatIdx,
#         "repeatSeed" : repeatSeed,
#         "foldId": foldIdx,  # if you have it
#         "accuracy": float(accuracy),
#         "auc": float(sensitivityDF.loc[:, "auc"].iloc[0]),
#         "sensitivityAt": float(sensitivityDF.loc[:, "Sensitivity (TPR)"].iloc[0]),  
#         "threshold": float(sensitivityDF.loc[:, "Approx. Threshold"].iloc[0]),
#     }
    
#     # sanitize threshold if it can be inf
#     if not np.isfinite(foldRowMetricDict["threshold"]):
#         foldRowMetricDict["threshold"] = np.nan
        
#     return foldRowMetricDict
                    
def createFoldMetricRowDict(
        columnPrefix : str,
        diagnosis : str,
        numFeatures : int,
        repeatIdx : int,
        repeatSeed : int,
        foldIdx : int, 
        accuracy : float, 
        sensitivityDF : pd.DataFrame,
    ) -> list[dict[str, Any]] :
    '''
    refactored code to improve readablity
    
    calculate fold metrics, replace infinite values with np.nan
    pandas mean() and std() will ignore these valuess
    
    columnPrefix : 
        should be either realFit,  SimulatedTrain, realTest,
        
        realFit : 
            the data set we used to create the distributions used to generated
            simulated data
            
        SimulatedTrain :
            the simulated data set the model was trained on
            
        realTest:
            the real data hold out set.
    '''          
    foldRowMetricDict : dict[str, Any] = {
        columnPrefix + "_diagnosis": diagnosis,
        columnPrefix + '_numFeatures' : numFeatures,
        columnPrefix + "_repeatId" : repeatIdx,
        columnPrefix + "_repeatSeed" : repeatSeed,
        columnPrefix + "_foldId": foldIdx,  # if you have it
        columnPrefix + "_accuracy": float(accuracy),
        columnPrefix + "_auc": float(sensitivityDF.loc[:, "auc"].iloc[0]),
        columnPrefix + "_sensitivityAt": float(sensitivityDF.loc[:, "Sensitivity (TPR)"].iloc[0]),  
        columnPrefix + "_threshold": float(sensitivityDF.loc[:, "Approx. Threshold"].iloc[0]),
    }
    
    # sanitize threshold if it can be inf
    if not np.isfinite(foldRowMetricDict[columnPrefix + "_threshold"]):
        foldRowMetricDict[columnPrefix + "_threshold"] = np.nan
        
    return foldRowMetricDict

# %%
# def averageMetricsAcrossFoldsDepreciated(
#         foldMetricsDF : pd.DataFrame, 
#         diagnosis : str,
#         numFeatures : int,
#         repeatIdx : int,
#         repeatSeed : int,
#     ) -> pd.DataFrame:
#     ''' 
#     refactored to improve readablity
#     calculate metric averages across all the folds for a given repeat
#     '''
#     repeatRowMetricsDF: pd.DataFrame = pd.DataFrame([{
#                                 "diagnosis": diagnosis,
#                                 'numFeatures' : numFeatures,
#                                 "repeatId" : repeatIdx,
#                                 "repeatSeed" : repeatSeed,
#                                 "accMean": float(foldMetricsDF["accuracy"].mean()),
#                                 "accStd": float(foldMetricsDF["accuracy"].std(ddof=1)),
#                                 "aucMean": float(foldMetricsDF["auc"].mean()),
#                                 "aucStd": float(foldMetricsDF["auc"].std(ddof=1)),
#                                 "sensitivityAtMean": float(foldMetricsDF["sensitivityAt"].mean()),
#                                 "sensitivityAtStd": float(foldMetricsDF["sensitivityAt"].std(ddof=1)),
#                                 # threshold optional:
#                                 "thresholdMean": float(foldMetricsDF["threshold"].mean()),
#                                 "thresholdStd": float(foldMetricsDF["threshold"].std(ddof=1)),
#                                 "nFolds": int(foldMetricsDF.shape[0]),
#                                 "nFiniteThreshold": int(np.isfinite(foldMetricsDF["threshold"].to_numpy()).sum()),
#                             }])
#     return repeatRowMetricsDF

def averageMetricsAcrossFolds(
        foldMetricsDF : pd.DataFrame, 
        diagnosis : str,
        numFeatures : int,
        repeatIdx : int,
        repeatSeed : int,
    ) -> pd.DataFrame:
    ''' 
    refactored to improve readablity
    calculate metric averages across all the folds for a given repeat
    '''
    
    # TODO no magic prefix, see   makePredictions()averageMetricsAcrossFolds()

    retDF = pd.DataFrame()

    for prefix in ['realFit', 'SimTrain', 'realTest']:
        repeatRowMetricsDF: pd.DataFrame = pd.DataFrame([{
                                    prefix + "_diagnosis": diagnosis,
                                    prefix + '_numFeatures' : numFeatures,
                                    prefix + "_repeatId" : repeatIdx,
                                    prefix + "_repeatSeed" : repeatSeed,
                                    prefix + "_accMean": float(foldMetricsDF[prefix + "_accuracy"].mean()),
                                    prefix + "_accStd": float(foldMetricsDF[prefix + "_accuracy"].std(ddof=1)),
                                    prefix + "_aucMean": float(foldMetricsDF[prefix + "_auc"].mean()),
                                    prefix + "_aucStd": float(foldMetricsDF[prefix + "_auc"].std(ddof=1)),
                                    prefix + "_sensitivityAtMean": float(foldMetricsDF[prefix + "_sensitivityAt"].mean()),
                                    prefix + "_sensitivityAtStd": float(foldMetricsDF[prefix + "_sensitivityAt"].std(ddof=1)),
                                    # threshold optional:
                                   prefix +  "_thresholdMean": float(foldMetricsDF[prefix + "_threshold"].mean()),
                                    prefix + "_thresholdStd": float(foldMetricsDF[prefix + "_threshold"].std(ddof=1)),
                                    prefix + "_nFolds": int(foldMetricsDF.shape[0]),
                                    prefix + "_nFiniteThreshold": int(np.isfinite(foldMetricsDF[prefix + "_threshold"].to_numpy()).sum()),
                                }])
        
        byRows = 1 # combine rows horizontally
        retDF = pd.concat([retDF, repeatRowMetricsDF], axis=byRows)
        
    return retDF


# %%
def generateTrainingDataSets(
                            localCachePath : str,
                             realMetaDF : pd.DataFrame,
                             geneList : list[str],
                             repeatIdx : int, 
                             foldIdx : int,
                             fold : StratifiedFold, 
                             numFeatures : str, 
                             diagnosis : str, 
                             lfcThreshold : float, 
                             padjThreshold : float,
                             ) -> tuple[pd.DataFrame, pd.Series, str, str]:
    '''
    Use real fold data to generate simiulated training data 
    
    returns
        (simulatedXTrainDF, simulatedBinaryYTrainSeries, negativeClass, positiveClass)
    '''
    
    t = saveFold(localCachePath, realMetaDF, repeatIdx, foldIdx,  fold, numFeatures, diagnosis, lfcThreshold, padjThreshold, )
    tmpOutPath, localCachePathFoldXTrainPath, localCachePathFoldTrainMaskPath = t
    
    # generate samples using real data split
    t = generateSamplesHelper(geneList, tmpOutPath, localCachePathFoldXTrainPath, localCachePathFoldTrainMaskPath)
    simulatedXTrainDF, simulatedYTrainSeries = t
    logger.info(f'simulatedXTrainDF.shape : {simulatedXTrainDF.shape} simulatedYTrainSeries.shape : {simulatedYTrainSeries.shape}')
    
    # train the model using the simulated data
    simulatedXTrainDF = simulatedXTrainDF.set_index('Gene')
    
    # our simulatedYTrainSeries.cat.codes are in int [0 ... 5]
    # This will cause problems calculating sensitivity
    #  create a new ordered categorical serires for the labels    
    negativeClass="Healthy_donor"           
    positiveClass=f"{diagnosis}_Cancer"
    simulatedBinaryYTrainSeries = makeBinaryOrderedCategorical(
                                        simulatedYTrainSeries,
                                        negativeClass,
                                        positiveClass,
                                    )
    
    return (simulatedXTrainDF, simulatedBinaryYTrainSeries, negativeClass, positiveClass)

def makePredictions(
        diagnosis : str,
        numFeatures : int, 
        repeatIdx : int, 
        repeatSeed : int, 
        foldIdx : int, 
        rfModel : RandomForestClassifier, 
        negativeClass : str,
        positiveClass : str, 
        fold : StratifiedFold,
        simulatedXTrainDF : pd.DataFrame,
        simulatedYTrainSeries : pd.Series,
        specificityTarget : float = 0.9
    ) -> dict[str, float]:
    ''' 
    arguments 

        negativeClass : 
        
        positiveClass : 
        
        realFitFold : 
            the data set we used to create the distributions used to generated
            simulated data
            
        simulatedTrainDF :
            the simulated data set the model was trained on
            
        realTestFold:
            the real data hold out set.
            
    returns :
        dict
    '''
    
    tmpSimulatedXTrainDF =  simulatedXTrainDF.transpose()
    columnOrder = tmpSimulatedXTrainDF.columns
    realXTrainDF = fold.XTrainDF.loc[:,columnOrder] 
    realXTestDF  = fold.XTestDF.loc[ :,columnOrder] 
    
    # foldBinaryYTestSeries = makeBinaryOrderedCategorical(fold.yTestSeries, negativeClass, positiveClass)
    yTrainSeries = makeBinaryOrderedCategorical(fold.yTrainSeries, negativeClass, positiveClass)
    yTestSeries  = makeBinaryOrderedCategorical(fold.yTestSeries, negativeClass, positiveClass)
      
    # TODO no magic prefix, see   makePredictions() averageMetricsAcrossFolds()
    parmameters = [
            ('realFit',  realXTrainDF,         yTrainSeries),
            ('SimTrain', tmpSimulatedXTrainDF, simulatedYTrainSeries),
            ('realTest', realXTestDF,          yTestSeries)
        ]
    
    
    resultsDict = dict()

    for p in parmameters:
        prefix, XDF, ySeries = p
        
        accuracy, sensitivityDF = calculateMetrics(
                                        rfModel=rfModel,
                                        foldXDF=XDF,
                                        foldBinaryYSeries=ySeries,
                                        positiveClass=positiveClass,
                                        specifityTargets=[specificityTarget]
                                    )
    
        foldRowMetricDict = createFoldMetricRowDict(
                                columnPrefix=prefix, 
                                diagnosis=diagnosis, numFeatures=numFeatures, 
                                repeatIdx=repeatIdx, repeatSeed=repeatSeed, 
                                foldIdx=foldIdx, accuracy=accuracy, 
                                sensitivityDF=sensitivityDF)
        
        resultsDict.update( foldRowMetricDict)
    
    # create a dataframe with one row. 
    #rowDF = pd.DataFrame(resultsDict).reindex(sorted(rowDF.columns), axis=1)
    
                    
    return resultsDict
# %%
def crossFitSimulatedToRealRF(
        localCachePath : str,
        kFolds : int,
        numRepeats : int, 
        featureCounts : list[int],
        cancerTypes : list[str],
        lfcThreshold : float,
        padjThreshold : float,
        realXDF : pd.DataFrame,
        realYSeries : pd.Series,
        realMetaDF : pd.DataFrame,
        seed : float,
        specifityTarget : float,
    ) :
    '''
    Assesses binary-classifier performance as simulated feature count increases
    
    psudo code:
    for d in diagnosis:
            for nf in featureCounts:
                for r in numRepeats:
                    select binary classifiler real data set
                    splitAndStratify int k fold
                    
                    for each fold
                        generate samples using fold.train data set
                        train random forest using simulated samples
                        
                        calculate performance metrics on fold.test data set
                        
                    calculate averege of fold level performance metrics
                    
                save averages of repeat level performance metrics
                            
    
    warning there are some hard code values
        elifeDESeqResults is a global
    
    The elifed data set is small
        53 "Colorectal Cancer",
        31 "Esophagus Cancer",
        26 "Liver Cancer",
        35 "Lung Cancer",
        36 "Stomach Cancer"
        43 Healthy donor
    
    arguments:
        localCachePath
            location to write temp files
            
        kFolds 
            is the number of Stratified real data folds
            we will make prediction on all k folds and 
            average our performance metrics
            
            give the small number of sample in each class. use a small k so that
            the we have enough real samples make predictions with
            
        numRepeats:
           see psudo code
                        
            
        featureCounts
            The number of features to use in each simulation, train, predict iteration
            
        cancerTypes
            should not include "Healty_donor"
            valid types are "Colorectal", "Esophagus", "Liver", "Lung", "Stomach"
            
        lfcThreshold, padj
            used to select differentially expressed genes
            
        realXDF, realYDF, realMetaDF the elife data for all diagnosis
        
        seed 
            we need to vary the seed across (cancerType, nFeatures, repeatIndex)
            The seed is init before training the random forest
                rfSeed = seed + diagnosis + repeatNumber
                
        specifityTarget: 
            are the specificity the sensitivity is calculated at
            example : 0.9
    '''
    logger.info("BEGIN")
    startTime: datetime = datetime.now()

    
    validateClasses(cancerTypes)
        
    for diagnosis in cancerTypes:
        resultsDF, totalNumFeatures = differentialExpressionHelper(diagnosis, lfcThreshold, padjThreshold)
        
        # diagnosis accumlator
        diagnosisRepeatsAccumulatorDF = pd.DataFrame()

        # repeatOutDirPath directory to write all results
        repeatOutDirPath = f'{localCachePath_G}/{diagnosis}_Cancer'
        os.makedirs(repeatOutDirPath, exist_ok=True)

        for numFeatures in featureCounts:
            # print("\n#########")
            if numFeatures > totalNumFeatures :
                emsg = f'numFeatures = {numFeatures} is > totalNumFeatures = {totalNumFeatures}'
                logger.error(emsg)
                break
            
            # reset repeat Metrics accumlator
            repeatMetricsAccumulatorDF = pd.DataFrame()
            
            for repeatIdx in range(numRepeats) :
                t = selectRealDataSet(resultsDF, numFeatures, realXDF, diagnosis, realYSeries, realMetaDF)
                geneList, tmpRealXDF, tmpRealYSeries, tmpRealMaskDF = t
                            
                repeatSeed = generateUniqRepeatableSeed(diagnosis, tmpRealYSeries, theMeaningOfLife, 
                                                        numFeatures, repeatIdx)
                folds = splitAndStratify(tmpRealXDF, tmpRealYSeries, kFolds, random_state=repeatSeed)
                
                # reset fold metric accumulators
                foldMetricsAccumulator: list[dict[str, Any]] = []

                for foldIdx, fold in enumerate(folds) :
                    logger.info(f'diagnosis : {diagnosis} numFeatures : {numFeatures} lfc : {lfcThreshold} padj : {padjThreshold} : {foldIdx}')
                    
                    t = generateTrainingDataSets(
                            localCachePath=localCachePath, realMetaDF=realMetaDF,
                            geneList=geneList, repeatIdx=repeatIdx, foldIdx=foldIdx,
                            fold=fold, numFeatures=numFeatures, diagnosis=diagnosis,
                            lfcThreshold=lfcThreshold, padjThreshold=padjThreshold,)
                    
                    simulatedXTrainDF, simulatedBinaryYTrainSeries, negativeClass, positiveClass = t
                    
                    # train model
                    rfModel = trainRFClassifier(simulatedXTrainDF, simulatedBinaryYTrainSeries, theMeaningOfLife)
                                    
                    # # make prediction using the folds' test data set
                    # # the simulation and training process never saw test data set
                    # # there can not be any sort of data leakage
                    # columnOrder = simulatedXTrainDF.transpose().columns
                    # foldXTestDF = fold.XTestDF.loc[:,columnOrder] 
                    
                    # foldBinaryYTestSeries = makeBinaryOrderedCategorical(fold.yTestSeries, negativeClass, positiveClass)

                    # accuracy, sensitivityDF = calculateMetrics(
                    #                                 rfModel=rfModel,
                    #                                 foldXTestDF=foldXTestDF,
                    #                                 foldBinaryYTestSeries=foldBinaryYTestSeries,
                    #                                 positiveClass=positiveClass,
                    #                                 specifityTargets=[specifityTarget]
                    #                             )
                    
                    # foldRowMetricDict = createFoldMetricRowDict(
                    #                         diagnosis=diagnosis, numFeatures=numFeatures, 
                    #                         repeatIdx=repeatIdx, repeatSeed=repeatSeed, 
                    #                         foldIdx=foldIdx, accuracy=accuracy, 
                    #                         sensitivityDF=sensitivityDF)
                    
                    t = makePredictions(diagnosis=diagnosis, numFeatures=numFeatures, repeatIdx=repeatIdx, repeatSeed=repeatSeed,
                    foldIdx=foldIdx, rfModel=rfModel, positiveClass=positiveClass, 
                    negativeClass=negativeClass, fold=fold, simulatedXTrainDF=simulatedXTrainDF, 
                    simulatedYTrainSeries=simulatedBinaryYTrainSeries, specificityTarget=0.9)
                    foldRowMetricDict = t

                    foldMetricsAccumulator.append(foldRowMetricDict)
                # end for foldIdx, fold in enumerate(folds) :
                
                # add fold average metrics to repeat metrics accumulator
                foldMetricsDF = pd.DataFrame(foldMetricsAccumulator)
                # print(f'\n**************** {diagnosis} : numFeatures {numFeatures}  rIdx{repeatIdx} foldMetricsDF')
                # display(foldMetricsDF)
                
                repeatRowMetricsDF = averageMetricsAcrossFolds(foldMetricsDF, diagnosis, numFeatures, repeatIdx, repeatSeed)
                
                # print("repeatRowMetricsDF")
                # display(repeatRowMetricsDF)
                repeatMetricsAccumulatorDF = pd.concat([repeatMetricsAccumulatorDF, repeatRowMetricsDF])
                
            # end for repeatIdx in range(numRepeats) :
            
            # save partial results to cache
            repeatOutCacheDirPath = f'{repeatOutDirPath}/cache'
            os.makedirs(repeatOutCacheDirPath, exist_ok=True)
            repeatMetricsPath = f'{repeatOutCacheDirPath}/nFeatures{numFeatures}_repeats_lfc{lfcThreshold}_padj{padjThreshold}_specficity{specifityTarget}.csv'
            repeatMetricsAccumulatorDF.to_csv(repeatMetricsPath , index=None)
            logger.warning(f'saving parital results{repeatMetricsPath}')
    
            # print('AEDWIP END OF REPEAT LOOP')
            # print(f'\n**************** {diagnosis} : numFeatures {numFeatures}  numRepeats : {numRepeats} repeatIdx : {repeatIdx}  repeatMetricsAccumulatorDF')
            # display(repeatMetricsAccumulatorDF)
           
            diagnosisRepeatsAccumulatorDF = pd.concat([diagnosisRepeatsAccumulatorDF, repeatMetricsAccumulatorDF])
            # print("AEDWIP AEDWIP AEDWIP diagnosisRepeatsAccumulatorDF concat")
            # display(diagnosisRepeatsAccumulatorDF)
            
        # end of for numFeatures in featureCounts:
        logger.info('!!!!!!!!!!!! AEDWIP end of numFeatrures loop diagnosisRepeatsAccumulatorDF')
        display(diagnosisRepeatsAccumulatorDF.head())
        # repeatDirPath = f'{notebookOutDir}/{diagnosis}_Cancer'
        # repeatDirPath = f'{localCachePath_G}/{diagnosis}_Cancer'
        # os.makedirs(repeatDirPath, exist_ok=True)
        # diagnosisRepeatsAccumulatorPath = f'{repeatDirPath}/repeats_lfc{lfcThreshold}_padj{padjThreshold}_specficity{specifityTarget}.csv'
        diagnosisRepeatsAccumulatorPath = f'{repeatOutDirPath}/repeats_lfc{lfcThreshold}_padj{padjThreshold}_specficity{specifityTarget}.csv'
        logger.warning(f'saving {diagnosis} diagnosisRepeatsAccumulatorDF.shape : {diagnosisRepeatsAccumulatorDF.shape}')
        logger.warning(f'saving : {diagnosisRepeatsAccumulatorPath}')
        diagnosisRepeatsAccumulatorDF.to_csv(diagnosisRepeatsAccumulatorPath, index=False)
        
    endTime : datetime = datetime.now()
    logger.info(f"END total run time : {elapsedTimeStr(startTime, endTime)}")
    
def testCrossFitSimulatedToRealRF() :
    '''
    Docstring for testCrossFitSimulatedToRealRF
    
    number of differential expresed genes
    {'Colorectal': {'padj': [0.001, 0.01, 0.5], 'count': [593, 841, 1286]},
    'Esophagus': {'padj': [0.001, 0.01, 0.5], 'count': [22, 52, 514]},
    'Liver': {'padj': [0.001, 0.01, 0.5], 'count': [7515, 8489, 9400]},
    'Lung': {'padj': [0.001, 0.01, 0.5], 'count': [196, 322, 478]},
    'Stomach': {'padj': [0.001, 0.01, 0.5], 'count': [2760, 3269, 4068]}}
    '''
    ret = crossFitSimulatedToRealRF(
        localCachePath=localCachePath_G,
        kFolds=3,
        numRepeats=2,
        featureCounts =  [5, 20000], # [5, 10,], #  25, ], #50, 100, 250, 500, 1000, ...]
        cancerTypes=["Colorectal", "Esophagus"], # "Liver", "Lung", "Stomach"],
        lfcThreshold=2.0, 
        padjThreshold=0.001,
        realXDF=realXDF_G, 
        realYSeries=realYSeries_G,
        realMetaDF=realMetaDF_G,
        seed=theMeaningOfLife,
        specifityTarget=0.9
    )
    

# %%

################################################################################
def main(inCommandLineArgsList : list[str] = None ):
    # logger = configLogging(logLevel='INFO')
    logger = configLogging(logLevel='WARN')
    logger.warning("BEGIN")
    
    #testCrossFitSimulatedToRealRF()
    
    cli = overfitRunnerCommandLine(
        version=__version__ , 
        author=__author__ ,
        date=__date__, 
        update=__updated__
    )
    
    if inCommandLineArgsList is None:
        cli.parse()
    else:
        cli.parse( inCommandLineArgsList )
        
    cancerType = cli.args.cancerType
    lfcThreshold = cli.args.lfcThreshold
    padjThreshold =  cli.args.padjThreshold
    specifityTarget = cli.args.specifityTarget
    featureCounts = cli.args.featureCounts
    
    dMsg = f'cli argument : cancerType :{cancerType}'
    dMsg = f' {dMsg} lfcThreshold :{lfcThreshold}'
    dMsg = f' {dMsg} padjThreshold :{padjThreshold}'
    dMsg = f' {dMsg} specifityTarget :{specifityTarget}'
    logger.warning( dMsg)

    numRepeats = cli.args.numRepeats
    dMsg = f'cli argument : featureCounts :{featureCounts} numRepeats : {numRepeats}'
    logger.warning( dMsg)

    crossFitSimulatedToRealRF(
        localCachePath=localCachePath_G,
        kFolds=3,
        numRepeats=numRepeats,
        featureCounts = featureCounts,
        cancerTypes=[cancerType],
        lfcThreshold=lfcThreshold,  # 2.0
        padjThreshold=padjThreshold, # 0.001
        realXDF=realXDF_G, 
        realYSeries=realYSeries_G,
        realMetaDF=realMetaDF_G,
        seed=theMeaningOfLife,
        specifityTarget=specifityTarget
    )
    
    logger.warning("END")
    exitCode = 0 # no error
    sys.exit(exitCode)  

################################################################################
if __name__ == "__main__":
    main()
    
    # inCommandLineArgsList=['-h']
    
    # inCommandLineArgsList=[
    #     '--intCountsPath', 'src/python/stats/test/r_best_models.int.csv',
    #     '--realCountsPath', 'src/python/stats/test/r_best_models.real.csv',
    #     '--out', 'tmp/bestPropDist.csv'
    # ]   
    
    # test the code to find best distribution Q-Q plots
    # inCommandLineArgsList = [
    #     '--intCountsPath', '/private/groups/kimlab/aedavids/fagazi/simulatedData/romanEsoph/best_fit_simulator.py.out_ZINB_NB_int/r_best_models.csv',
    #     '--realCountsPath', '/private/groups/kimlab/aedavids/fagazi/simulatedData/romanEsoph/best_fit_simulator.py.out/r_best_models.csv',
    #     '--out', 'tmp/bestPropDist.csv'
    # ] 
    
    # main( inCommandLineArgsList )


