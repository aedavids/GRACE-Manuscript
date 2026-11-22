"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2026-05-21
Copyright (c) 2020-2023, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 

see 
- https://gseapy.readthedocs.io/en/latest/gseapy_example.html
    for example of use
    
- src/jupyterNotebooks/elifeVsTCGA/gsea.ipynb

public data classes

    @dataclass
    class CustomLibraryParameters :

    class Diagnosis(str, Enum):
    
    @dataclass
    class GSEAConfig :
    
    @data
    class GSEA_PlotData

public classes and functions

    def addHUGOIds()
    
    def cleanUpDups()
        
    def createBestGeneSetPlotData()
    
    def createEnsembleToHUGOMap
    
    def createGSEAConfig
    
    def createPreRankGeneSetsArgument
    
    def downloadMsigdbGmt
    
    def exploreGeneSetDict
    
    def runPrerank
    
    def saveGSEAConfig
            
"""
from __future__ import annotations  

from dataclasses import asdict, dataclass
from enum import Enum
import json

import logging
logger = logging.getLogger(__name__)

import gseapy as gp
import os
import pandas as pd
from pathlib import Path
import requests
import sys
import time

# local imports
from data.mapIds import createMapDF
from data.loadCounts import loadDataFrame

#
# data classes
# 

################################################################################
@dataclass
class CustomLibraryParameters :
    '''
    To ensure the "pedigree" and "repeatability" 
    caputure parameters to createGSEA_RepeatDataSet()
    '''
    key : str
    repeatMaskPath : str
    
################################################################################
class Diagnosis(str, Enum):
    ''' 
    Elife multiclassifier categories
    '''
    COLORECTAL = "Colorectal"
    ESOPHAGUS  = 'Esophagus'
    HEALTHY    ='Healthy'
    LIVER      = 'Liver'
    LUNG       = 'Lung'
    STOMACH    = 'Stomach'

################################################################################
@dataclass
class GSEAConfig :
    '''
    To ensure the "pedigree" and "repeatability" capture parameters 
    '''
    elifeDiagnosis : str
    diagnosis : Diagnosis
    
    # DESeq results file
    resultsFilePath : Path
    fdrThreshold : float
    nesThreshold: float 
    
    # define the deseq results col to rank by
    rankCol : str 
    
    # list of gene set collection tuples from MSigDB or local gmt file
    # example
    # built in MSigDB librarys [('KEGG_26', 'KEGG_26')]
    # local gmt file [('myCollection', 'path to myCollection.gmt')]
    geneSetLibraries : list[tuple[str, str]]
    
    # use to capture repeat mask collections
    customGeneSetLibraries: list[CustomLibraryParameters]
    
################################################################################
@dataclass
class GSEA_PlotData :
    dbName : str
    bestN : int
    bestUpDF : pd.DataFrame
    bestDownDF : pd.DataFrame
    
################################################################################
def addHUGOIds(
        gseaInputDF : pd.DataFrame,
        ensg2HugoMappingDF : pd.DataFrame,
        leftOnColName : str = "geneId"
    ) -> pd.DataFrame : 
    ''' 
    use join operation to add hugo id column to gseaInputDF
    
    arguments:

        gseaInputDF
            a data frame that has a column named 'geneId'. The values
            should be ensembl ids
            
            example a df produced by DESeq
            
        ensg2HugoMappingDF 
            a data frame with columns ['originalENSG', 'HUGO']
            example : something returned by createEnsembleToHUGOMap()    
    '''  
    # use outer join to make sure there where no mapping bugs
    mergedResultsDF = gseaInputDF.merge(
            ensg2HugoMappingDF.loc[:, ['originalENSG', 'HUGO']],  #'base',
            # left_on="geneId",
            left_on=leftOnColName,
            right_on='originalENSG', #"base",
            how="outer",
            indicator=True,
            suffixes=('_results', '_mapping'),
    )

    print(f'mergedResultsDF.shape :{ mergedResultsDF.shape}')
    print(f'gseaInputDF.shape       : {gseaInputDF.shape}')
    selectBothOnlyRows = mergedResultsDF['_merge'] == "both"
    retHugoResultsDF = mergedResultsDF.loc[selectBothOnlyRows, :]
    
    retHugoResultsDF = retHugoResultsDF.drop(columns="_merge")

    
    return retHugoResultsDF

################################################################################
def createGSEA_RepeatDataSet(
        key : str,
        repeatMaskPath : str="/private/groups/kimlab/genomes.annotations/genomes.annotations/gencode.39/named_ucsc.rmsk.insert.info.txt",
        
    ) -> tuple[dict[str, list[str]], pd.DataFrame] :
    ''' 
    given a repeat insert information, returns a dictionary that can be used in gseapy
    functions for argument gene_sets
    
    arguments :
        key : either 'clade' or 'family'
    
        example:
        /private/groups/kimlab/genomes.annotations/genomes.annotations/gencode.39/named_ucsc.rmsk.insert.info.txt
        
        clade	        family	        insert
        Simple_repeat	Simple_repeat	(TAACCC)n
        Satellite	    telo	        TAR1
        LINE	        L1	            L1MC5a
        DNA	            hAT-Charlie	    MER5B
        SINE	        MIR	            MIR3
        
    returns
        (geneSetDict, repeatMaskDF)
    '''
    validKeys = [ 'clade', 'family']
    if not key in validKeys :
        emsg = f'{key} in not in {validKeys}'
        raise ValueError(emsg)
    
    ucscRmskDF = pd.read_csv( repeatMaskPath, sep="\t")
    ucscRmskDF = ucscRmskDF.loc[:, ['clade', 'family', 'insert']]
    
    # ucscRmskDF.groupby('clade')['insert']
    # organizes all individual insert loci under their respective clade labels.
    #
    # .apply(list)
    # ensures that all insert IDs associated with a specific clade are bundled into a list
    geneSetDict = ucscRmskDF.groupby(key)['insert'].apply(list).to_dict()

    return (geneSetDict, ucscRmskDF)
    
################################################################################
def createPreRankGeneSetsArgument(gseaConfig : GSEAConfig
                                  ) :
    '''
    see documentation in runPrerank for geneSetDataBase argument
    '''
    retList = list()
    
    # add the MSigDB collections
    for tup in gseaConfig.geneSetLibraries:
        libraryName,libraryPath = tup
        retList.append( (libraryName, libraryPath) )
        
    # add any custome gen sets
    if len(gseaConfig.customGeneSetLibraries) > 0:
        for cgsl in gseaConfig.customGeneSetLibraries :
            key = cgsl.key
            path = cgsl.repeatMaskPath
            geneSetCladeDict, ucscRmskDF = createGSEA_RepeatDataSet(key, path)
            retList.append( (key, geneSetCladeDict) )
    
    return retList

################################################################################
def cleanUpDups(
        hugoResultsDF : pd.DataFrame,
        rankCol : str="stat",
        verbose : bool=True # display dups
        ) -> pd.DataFrame:
    ''' 
    it is possible that ensembl id do not map 1:1 with HUGO
    save the duplicate with the highest rankCol value,
    and drop the remaining dups
    
    '''
    # markAllDupsTrue = False # weird this will cause them all to be marked True
    hugoDupRows = hugoResultsDF.loc[:, 'HUGO'].duplicated(keep=False)

    if sum(hugoDupRows) > 0:
        dupDF = hugoResultsDF.loc[hugoDupRows, :]
        logger.warning('HUGO NAMES are not uniq')
        if verbose :
            print(dupDF)
                
        # Keep row with largest absolute rankCol for each HUGO
        bestDupIdx = (
            dupDF[rankCol]
            .abs()
            .groupby(dupDF["HUGO"])
            .idxmax() # returns the index of the largest value
        )
        
        logger.warning(f'keeping index: \n{hugoResultsDF.loc[bestDupIdx, :].index}')
        
        # Non-duplicated rows + best duplicated rows
        nonDupDF = hugoResultsDF.loc[~hugoDupRows, :]

        hugoResultsDedupDF = pd.concat(
            [nonDupDF, hugoResultsDF.loc[bestDupIdx, :]],
            axis=0
        )
        
    return hugoResultsDedupDF

################################################################################
def createBestGeneSetPlotData(
    signifignatTermsDir : str,
    diagnosis : list[str],
    geneSetLibraries : list[str],
    bestN : int,
    rankColName : str,
    ) -> dict[str, GSEA_PlotData]:
    ''' 
    for each gene set library
        create a dataframe from of all the best up sets in each diagnosis
        create a dataframe from of all the best down sets in each diagnosis
        
    arguments 
    
        signifignatTermsDir:
            path directory containins output from runPrerank. see gsea.ipynb
    '''
    
    retPlotDataDict = dict()
    
    # example dbName c8.all.v2026.1.Hs.symbols.gmt or clade
    for dbName in geneSetLibraries:    
        allBestUpDF = pd.DataFrame()
        allBestDownDF = pd.DataFrame()
        for diag in diagnosis:
            # example diag = "Healthy Donor"
            dirName = diag.split(" ")[0]
            signifignatTermsPath = f'{signifignatTermsDir}/{dirName}/significantTerms.csv'
            df = pd.read_csv(signifignatTermsPath)
            
            selectDBRows = df.loc[:, 'geneSetDB'] == dbName
            df = df.loc[selectDBRows, :]
            
            # gsea.ipyn filtered based on fdr and nes before writing
            # > 0 find the sets that are up regulated in extra
            # < 0 find the sets that up in intra
            bestUpDF = (
                    df.query(f"{rankColName} > 0")
                    .sort_values(["diagnosis", rankColName], ascending=[True, False])
                    .groupby("diagnosis")
                    .head(bestN)
            )
            if allBestUpDF.empty :
                allBestUpDF = bestUpDF
            else :
                tmpDF = pd.concat([allBestUpDF, bestUpDF], ignore_index=True)
                allBestUpDF = tmpDF
            
            bestDownDF = (
                df.query(f"{rankColName} < 0")
                .sort_values(["diagnosis", rankColName], ascending=[True, False])
                .groupby("diagnosis")
                .tail(bestN)
            )
            if allBestDownDF.empty :
                allBestDownDF = bestDownDF
            else :
                tmpDF = pd.concat([allBestDownDF, bestDownDF], ignore_index=True)
                allBestDownDF = tmpDF

        plotData = GSEA_PlotData(dbName, bestN, allBestUpDF, allBestDownDF)
        retPlotDataDict[dbName] = plotData
            
        # logger.error(f'AEDWIP DO NOT break')
        # break
        
    return retPlotDataDict

################################################################################
def createEnsembleToHUGOMap(
    mapFilePath : str, 
    genomesAnnotationsPath : str="/private/groups/kimlab/genomes.annotations/genomes.annotations/gencode.39/gencode.v39.ucsc.rmsk.tx.to.gene.csv"
    ) -> pd.DataFrame :
    ''' 
    check local cache to see if mapping file exits.
    If does not exits
        create a maping data frame and save in csv and pandas.h5 format
    else :
        loads local cached version 
        
    returns a dataframe with columns 
        [HUGO	ENSG	base	version	originalENSG]
        
    most of the MSigDB gene sets use HUGO name
    
    Most of Kim lab files use Ensemble ids
    
    do not use v35_to_v39Mapping mapping, ensembl ids map 1 to many HUGO id'
    matchingFeaturesRefDir = "/scratch/aedavids/fagazi/v35_2_v39Mapping.out"
    mappingFeaturesPath = f'{matchingFeaturesRefDir}/v35_to_v39Mapping.csv'
    
    arguments :
        mapFilePath :
            location in local cache
    '''
    if not os.path.exists(mapFilePath) :
        # KLDir = "/private/groups/kimlab/genomes.annotations"
        # v39Tx2GeneRefPath= f"{KLDir}/genomes.annotations/gencode.39/gencode.v39.ucsc.rmsk.tx.to.gene.csv"

        v39GeneMapDF = createMapDF(genomesAnnotationsPath)
        v39GeneMapDF.to_csv(mapFilePath, index=False)
        print(f'saved : {mapFilePath}')
        h5Path = Path(mapFilePath).with_suffix(".pandas.h5")
        v39GeneMapDF.to_hdf(
            h5Path,
            key="df",
            mode="w",
            #format="table"
        )
        print(f'saved : {h5Path}')
    else :
        v39GeneMapDF = loadDataFrame(mapFilePath)
        
    # add isENSG to make it easier to figure out type of mapping error
    v39GeneMapDF['isENSG'] = v39GeneMapDF['ENSG'].str.startswith("ENSG")
    
    return v39GeneMapDF

################################################################################
def createGSEAConfig(
            diagnosis : Diagnosis,
            geneSetLibraries :  list[tuple[str, str]], 
            # geneSetLibraries :  list[str], 
            customGeneSetLibraries : list[CustomLibraryParameters],
            resultsDir : str="/scratch/aedavids/fagazi/analyzeIntraExtra.out",
            resultsFileName : str='expressedInElife.results',
            fdrThreshold : float=0.05,
            nesThreshold : float=1.5, 
            rankCol : str = "stat"
    ) -> GSEAConfig:
    ''' 
    prevent configuration bug. 
    
        geneSetLibraries :
            list of gene set collection from MSigDB or local gmt file
            see GSEAConfig for examples
           
        customGeneSetLibraries :
            see dataclass documentation
    
        resultsDir :
            assumes the resultes file path is 
            f'{resultsDir}/{diagnosis}/(resultsFileName)
            
    
        fdrThreshold
            Use a value like 0.25 for generating new hypotheses
            
            Use a value like 0.05 for validating models to ensure the biological signals
            are highly robust
        
        nesThreshold
            use a value like like 1.0 for for generating new hypotheses
            
            use a value >= 1.5, (Strong Signal) for validating models to ensure the biological signals
            are highly robust
            
        rankCol : 
            the column to sort the DESeq in decending order by
            
    '''
    diagnosisDir = diagnosis.split(" ")[0]
    resultsFilePath = Path(f'{resultsDir}/{diagnosisDir}/{resultsFileName}')

    retConfig= GSEAConfig(
            #geneSetDatabase='MSigDB_Hallmark_2020' 
            elifeDiagnosis=diagnosis,
            diagnosis=Diagnosis[ diagnosisDir.upper() ],
            resultsFilePath=resultsFilePath,
            fdrThreshold=fdrThreshold,
            nesThreshold=nesThreshold,
            rankCol=rankCol,
            geneSetLibraries=geneSetLibraries,
            customGeneSetLibraries=customGeneSetLibraries,
        )
    
    return retConfig   

################################################################################
def downloadMsigdbGmtBuggy(
        geneSetCollectionName: str,
        cacheDir: Path,
        # version: str = "2026.1.Hs",
        overwrite: bool = False,
    ) -> Path:
    ''' 
    Checks to see if MSigDB file is in local cache. If not. it will download it
    
    arguments:
        geneSetCollectionName :
            example 'KEGG_2066'
    
    returns the path the gmt file in the local cache
    
 
    '''
    targetFileName = f'{geneSetCollectionName}.gmt'
    cacheDir.mkdir(parents=True, exist_ok=True)
    # fullPath = os.path.join(cacheDir, targetFileName)
    fullPath = cacheDir / targetFileName

    if not overwrite and fullPath.exists() :
        return fullPath
    
    gp.get_library(
        name=geneSetCollectionName,
        organism="Human",
        save=fullPath,
    )

    

################################################################################
def downloadMsigdbGmt(
        fileName: str,
        cacheDir: Path,
        version: str = "2026.1.Hs",
        overwrite: bool = False,
    ) -> Path:
    ''' 
    Checks to see if MSigDB file is in local cache. If not. it will download it
    
    returns the path the gmt file in the local cache
    
    example of use
    
        # you must first register an email address with https://www.gsea-msigdb.org/gsea/msigdb
        msigDBRegisteredEmail = "myAddress@ucsc.edu"
        os.environ["MSIGDB_EMAIL"] = msigDBRegisteredEmail
        
        gmtPath = downloadMsigdbGmt(
            fileName='c6.all.v2026.1.Hs.symbols.gmt',
            cacheDir=Path(cacheDir),
        )
        
    created by chatGPT
    '''
    
    cacheDir.mkdir(parents=True, exist_ok=True)
    outFilePath = cacheDir / fileName

    if outFilePath.exists() and not overwrite:
        return outFilePath

    url = (
        "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/"
        f"{version}/{fileName}"
    )

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    contentType = response.headers.get("content-type", "")
    if "text/html" in contentType or response.text.lstrip().startswith("<"):
        raise RuntimeError(f"Expected GMT, got HTML from {url}")

    outFilePath.write_bytes(response.content)
    return outFilePath


################################################################################
def exploreGeneSetDict(
    geneSetDict : dict[str, list[str]]
    ) -> list[ tuple[str, str]]:
    ''' 
    use to get an idea of how many data sets and how many genes are in dictionary
    
    example of a returned list
    
    [('DNA', 322), ('DNA?', 30), ('LINE', 183)]
    
    only len(geneSetDict['key]) > 1
    '''
    keys = [k for k in geneSetDict.keys()][-5:]
    retList = list()
    for k in geneSetDict.keys() :
        # print(f'\n***********k : {k}')
        lociList = geneSetDict[k]
        n = len(lociList)
        if n > 1 :
            retList.append( (k, n) )

    return retList
       
################################################################################
def runPrerank(
        geneSetDatabase : list[tuple[str, str | dict[str, list[str]]]],
        gseaConfig : GSEAConfig,
        rankDF : pd.DataFrame,
        fdrThreshold : float=0.05,
        nesThreshold : float=1.5,
        seed : int=42
    ) -> tuple[pd.DataFrame, dict['str',gp.Prerank]]:
    
    ''' 
    sorts the gene set results by nes and fdr
    sigifigant sets = (fdr <= fdrThreshold) and ( abs(nes) >= nesThreshold)   
    
    arguments 
        geneSetDataBase :
            a list of tuples   

            preRank() gene_sets argument can be the name of a database that is shipped with gseapy
            like 'MSigDB_Hallmark_2020' or a custom dictionary.
            
            example :
                [ ('KEGG_2021', 'KEGG_2021_Human'),('My_Custom_Repeats', myRepeatDict)]
                
        fdrThreshold
            Use a value like 0.25 for generating new hypotheses
            
            Use a value like 0.05 for validating models to ensure the biological signals
            are highly robust
        
        nesThreshold
            use a value like like 1.0 for for generating new hypotheses
            
            use a value >= 1.5, (Strong Signal) for validating models to ensure the biological signals
            are highly robust
         
        seed 
            random seed   
        
    returns 
        (retSignificantResultsDF, retPreRankDict)
        
        significantResultsDF 
            columns = [Term, ES, NES, FDR q-val, Lead_genes, diagnosis, geneSetDB]
            
        preRankDict 
            key = an item from the list of geneSetDatabase
            value = gseapy.prerank(). Use to create plots

    '''
    retSignificantResultsDF = pd.DataFrame()
    retPreRankDict = dict()
    
    for geneSetDBTuple in geneSetDatabase:
        # print(f'use time.sleep() to make debug easier should be able to interupt')
        time.sleep(1.0)
        
        geneSetDBKey, geneSetDBArgument = geneSetDBTuple
        print(f'\n{gseaConfig.diagnosis} {geneSetDBKey} fdrThreshold : {fdrThreshold} nesThreshold : {nesThreshold}')
        try :
            preRankResults = gp.prerank(rnk=rankDF, gene_sets=geneSetDBArgument, seed=seed) #outdir=
        except Exception as e:
            logger.exception("Operation failed")
            continue
        
        retPreRankDict[geneSetDBKey] = preRankResults
        
        df = preRankResults.res2d
        statsDF = df.loc[:,['Term', 'ES', 'NES', 'FDR q-val', 'Lead_genes']].sort_values(by='FDR q-val').reset_index(drop=True)
        
        # select signifigant data sets
        selectFRDThresholdRows = statsDF.loc[:, 'FDR q-val'] <= fdrThreshold
        fdrPlotDF = statsDF.loc[selectFRDThresholdRows, :]
        print(f'select <= fdrThreshold :fdrPlotDF.shape : {fdrPlotDF.shape}')
        
        selectNESThresholdRows = fdrPlotDF.loc[:, 'NES'].abs() >= nesThreshold
        nesPlotDF = fdrPlotDF.loc[selectNESThresholdRows, :]
        print(f'select abs(NES) >= nesThreshold nesPlotDF.shape : {nesPlotDF.shape}')
        # display(nesPlotDF)

        sortedDF = nesPlotDF.sort_values(by="NES", ascending=False)
        # print(f'sortedDF.NES :]\n{sortedDF.NES}')
        
        sortedDF['diagnosis'] = [gseaConfig.diagnosis] * sortedDF.shape[0]
        sortedDF['geneSetDB'] = [geneSetDBKey] * sortedDF.shape[0]
        
        # print(f'sortedDF.shape : {sortedDF.shape} numberToPlot : {numberToPlot}')
        # print(f'plotDF.shape : {plotDF.shape} numberToPlot : {numberToPlot}')
        
        if retSignificantResultsDF.empty :
            retSignificantResultsDF = sortedDF
        else :
            tmpDF = pd.concat([retSignificantResultsDF, sortedDF],
                              ignore_index=True,)
            retSignificantResultsDF = tmpDF
        
    return (retSignificantResultsDF, retPreRankDict)

################################################################################
def saveGSEAConfig(
        gseaConfig : GSEAConfig,
        outFilePath : str
        ) -> str:
    ''' 
    save configuration in json format to ensure"pedigree" and "repeatability"
    
    returns
        path to json file
    '''
    configDict = asdict( gseaConfig )
    
    # convert Path to str
    configDict['resultsFilePath'] = str( gseaConfig.resultsFilePath )
    # convert enum to str
    configDict['diagnosis'] = gseaConfig.diagnosis.value
    
    if os.path.exists(outFilePath) :
        raise FileExistsError(f"{outFilePath} already exists")
    
    with open(outFilePath, "w") as f:
        json.dump(configDict, f, indent=4)
        
    print(f'saved : {outFilePath}')
    
################################################################################
def main():

    # we only configure logging in main module
    # loglevel = "WARN"
    loglevel = "INFO"
    # logFMT = "%(asctime)s %(levelname)s [thr:%(threadName)s %(name)s %(funcName)s() line:%(lineno)s] [%(message)s]"
    logFMT = "%(asctime)s %(levelname)s %(name)s %(funcName)s() line:%(lineno)s] [%(message)s]"
    logging.basicConfig(format=logFMT, level=loglevel)
    logger = logging.getLogger(os.path.basename(__file__))
    logger.warning("BEGIN")

    #
    # always log run time env to make debug easier
    #
    ORIG_PYTHONPATH = os.environ['PYTHONPATH']
    logger.warning(f'PYTHONPATH : {ORIG_PYTHONPATH}')
    logger.warning(f'FILE: {__file__}')
    logger.warning(f'PWD: {os.getcwd()}')

    # if inCommandLineArgsList is None:
    #     cli.parse()
    # else:
    #     cli.parse( inCommandLineArgsList )

    # logger.warning(f'command line arguments : {cli.args}')
    # features = cli.args.features
    # 	outDir   = cli.args.outDir

    #	exitCodeOkay = 0
    #	# exitCodeError = 1
    #	logger.warning(f"END exitCode : {exitCodeError}")
    #	sys.exit( okay )

################################################################################
if __name__ == "__main__":
    main()
#hack used to test parsing of vargs
# mainCLITest(
#     #inCommandLineArgsList=['-h']
#     inCommandLineArgsList=[
#         '--features', 'LUAD',
#         '--outDir', 'tmp'
#     ]
# )
