"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-09-07
Copyright (c) 2020-2023, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 
    splits count data and meta data file into small data sets 
    defined by a list of categorical variables suitable for 
    training binary or multi-classifere models
    or differential expression analysis
    
    ref:
        fagazi/src/bin/deseq/binary/binaryDeseq.sh
"""

from abc import ABC, abstractmethod
from argparse import ArgumentParser
#https://stackoverflow.com/a/3853776/4586180
from argparse import RawDescriptionHelpFormatter
from baseCommandLine import BBaseCommandLine
from cliUtilities import configLogging

import logging
logger = logging.getLogger(__name__)
import pandas as pd
from pathlib import Path
import os
import sys

# global variables use by cli
__all__ = []
__version__ = 0.1
__author__ = "Andrew Davidson aedavids@ucsc.edu"
__date__ = '2025-09-21'
__updated__ = '2025-09-21'

################################################################################
class LoadCommandLine( BBaseCommandLine ):
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
        self.parser.add_argument( '-p', '--prefix', default="", metavar="", type=str, action="store", 
                                  help="save output files with prefix")

        
        #
        # group required arguments. This will create a better help message
        # make sure to set required=True
        #
        self.requiredArg = self.parser.add_argument_group( 'required arguments' )

        self.requiredArg.add_argument( '-c', '--countPath', required=True, 
                                                # default=".", 
                                                # metavar="",   
                                                action='store', 
                                                type=str,
                                                help="path to count data csv file"
        ) 

        self.requiredArg.add_argument( '-k', '--categoryColName', 
                                default=None, metavar="", 
                                action="store",
                                type=str,  
                                help="name of the column in the meta data file that contains categorical data"
        )
        
        self.requiredArg.add_argument( '-m', '--metaPath', required=True, 
                                                # default=".", 
                                                # metavar="",   
                                                action='store', 
                                                type=str,
                                                help="path to meta data csv file"
        ) 

        self.requiredArg.add_argument( '-o', '--outDir', required=True, 
                                                # default=".", 
                                                # metavar="",   
                                                action='store', 
                                                type=str,
                                                help="output directory"
        ) 

        self.requiredArg.add_argument( '-s', '--sampleIdColName', 
                                default=None, metavar="", 
                                action="store",
                                type=str,  
                                help="name of the column in the meta data file that contains sample ids data"
        )
        
        self.requiredArg.add_argument(
                                "-", "--types",
                                type=lambda s: s.split(","),
                                help="Comma-separated list of categories of interest"
        )

################################################################################
def fixCategories(df : pd.DataFrame,
                  categoryColName : str,
                  categories : list[str]
                  ) -> tuple[pd.DataFrame, list[str] ] :
    '''
    trims white space from beining and ends of strings, 
    
    replaces " " with "_"
    '''
    retCategories = []
    for c in categories:
        cc = c.strip().replace(" ", "_")
        retCategories.append(cc)
        
    retDF = df.copy()
    retDF[categoryColName] = df[categoryColName].str.strip()
    retDF[categoryColName] = retDF[categoryColName].str.replace(" ", "_")
    
    return(retDF, retCategories)
  
################################################################################  
def loadDataFrame(
    csvPath: str | Path,
    indexCol: int | str | None = None,
    h5Key: str = "df",
) -> pd.DataFrame:
    """
    Load a DataFrame from a cached .pandas.h5 file if it exists,
    otherwise load the CSV file.

    Parameters
    ----------
    csvPath : str | Path
        Path to the CSV file.

    indexCol : int | str | None, optional
        Column to use as index (same meaning as pandas.read_csv).

    Returns
    -------
    pd.DataFrame
    """

    csvPath = Path(csvPath)
    h5Path = csvPath.with_suffix(".pandas.h5")

    if h5Path.exists() :
        logger.warning(f'loading : {h5Path}')
        retDF = pd.read_hdf(h5Path, key=h5Key)
        
        # If caller expects an index and H5 currently has a default RangeIndex,
        # promote the requested column to the index.
        if indexCol is not None : # and isinstance(retDF.index, pd.RangeIndex):
            if isinstance(indexCol, int):
                if not (0 <= indexCol < retDF.shape[1]):
                    raise IndexError(f"indexCol={indexCol} out of range for {retDF.shape[1]} columns")
                retDF = retDF.set_index(retDF.columns[indexCol])
            else:
                # indexCol is a column name like "Gene"
                if indexCol not in retDF.columns:
                    raise KeyError(f"indexCol column '{indexCol}' not found in H5 columns")
                retDF = retDF.set_index(indexCol)
        
    else :
        logger.warning(f'loading : {csvPath}')
        retDF = pd.read_csv(csvPath, index_col=indexCol)
        
    logger.debug(f'AEDWIP {retDF.shape=}')
    logger.debug(f'AEDWIP \n{retDF.iloc[0:5, 0:4]}')
    return retDF

################################################################################
def loadGenesBySamplesData( categories : list[str], 
           xDFPath : str,
           yDFPath : str,
           yCategoryColName : str = 'Classes',
           yRowIdName : str = 'sampleIds',
           withFullMetadata : bool=False
           ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None] :
    '''    
    assums yDF is meta data with a column name 

    wrap this in a function so that unused data can be released from memory

    input csv file format must be genes x samples

    example:
    ```
    head -n 5 X.csv | cut -d , -f 1,2,3,4
    Gene,              s0,                s1,               s2
    (AGCCAAGCC)n,      3198.168304984847, 2533.642915110422, 10150.633555859104
    (GAGA)n,           1073.269682327276, 535.261260252683,  171.93040560936683
    5S,                44237.82874355771, 3348.077019511625, 134.27848331002588
    ENSG00000004939.16, 482.0,            359.0,             294.0
    ````


    returns a tuple with
        a data frame with form sample_ids x gene names
        a series of labels
        
        if withFullMetadata
            meta data dataframe
        else:
            None
    '''
        
    # XDF = pd.read_csv( xDFPath, index_col=0 )
    # logger.info(f'loading : {xDFPath}')
    XDF = loadDataFrame(xDFPath, indexCol=0) 
    logger.info(f'XDF.shape : {XDF.shape}')
    

    # yDF = pd.read_csv( yDFPath )
    # logger.info(f'loading : {yDFPath}')
    yDF =loadDataFrame( yDFPath )

    # #
    # 
    # cleanCategories = []
    # for c in categories:
    #     cc = c.strip().replace(" ", "_")
    #     cleanCategories.append(cc)
    # yDF[yCategoryColName] = yDF[yCategoryColName].str.strip()
    # yDF[yCategoryColName] = yDF[yCategoryColName].str.replace(" ", "_")
    
    # trim white space from ends and replace space with ategory names with '_'    
    yDF, cleanCategories = fixCategories(yDF, yCategoryColName, categories)
    

    # select the sample ids of interest
    selectRows = yDF.loc[:, yCategoryColName].isin( cleanCategories )
    logger.info(f'selectRows.shape : {selectRows.shape}')
    sampleIdSeries = yDF.loc[selectRows, yRowIdName]#.values

    retXDF = XDF.loc[:, sampleIdSeries].transpose()
    retXDF.index.name = 'sampleIds'

    retYSeries = yDF.loc[selectRows, yCategoryColName]
    
    if withFullMetadata :
        #yDF = yDF.set_index(yDF.columns[0])
        # yDF.index.name = retXDF.index.name
        ret = (retXDF, retYSeries, yDF)
    else :
        ret = (retXDF, retYSeries)
        
    return ret

# we need a function that you pass classes of interested and mdatadatapath and returns df of matching 

################################################################################
def loadMetaData(
            yDFPath : str,
            yCategoryColName : str,
            categories : list[str]
    ) -> pd.DataFrame :
    '''
    we can not change loadGenesBySamplesData, need to preserve backwards compatiblity
    '''
    
    yDF = pd.read_csv( yDFPath )
    logger.info(f'loading : {yDFPath}')
    
    # trim white space from ends and replace space with ategory names with '_'    
    yDF, cleanCategories = fixCategories(yDF, yCategoryColName, categories)
    
    selectRows = yDF.loc[:, yCategoryColName].isin( cleanCategories )
    logger.info(f'selectRows.shape : {selectRows.shape}')
    retDF = yDF.loc[selectRows, :]
    
    return retDF

################################################################################ 
def _parseCLI(logger : logging.Logger, 
              inCommandLineArgsList : list[str]) -> tuple[str,str,str,str,str,str]:
    '''
    TODO
    '''

    cli = LoadCommandLine(
            version=__version__ , 
            author=__author__ ,
            date=__date__, 
            update=__updated__
    )
    
    if inCommandLineArgsList is None:
        cli.parse()
    else:
        cli.parse( inCommandLineArgsList )

    logger.warning(f'command line arguments : {cli.args}')
    logger.warning("Parsed arguments:")
    for arg_name, arg_value in vars(cli.args).items():
        logger.warning(f"{arg_name}: {arg_value}")
    
    categoryColName = cli.args.categoryColName
    countPath       = cli.args.countPath
    metaPath        = cli.args.metaPath
    outDir          = cli.args.outDir
    sampleIdColName = cli.args.sampleIdColName
    types           = cli.args.types
    prefix          = cli.args.prefix

    return (categoryColName, countPath, metaPath, outDir, sampleIdColName, types, prefix)

#################################################################################
def main( inCommandLineArgsList : list[str] = None ):
    '''
    TODO
    '''

    rootLogger = configLogging( logLevel='INFO')
    #rootLogger = configLogging(logLevel='WARN')
    logger.warning("BEGIN")

    t = _parseCLI(logger, inCommandLineArgsList )
    categoryColName, countPath, metaPath, outDir, sampleIdColName, types, prefix = t
    
    xDF, ySeries = loadGenesBySamplesData(
                    categories = types,
                    xDFPath=countPath,
                    yDFPath=metaPath,
                    yCategoryColName=categoryColName,
                    yRowIdName=sampleIdColName
                    )
    
    yDF = loadMetaData(metaPath, categoryColName, types)
    
    # loadGenesBySamplesData tranposes xDF, undo to preserver original format
    xDF = xDF.transpose()
    
    logger.warning(f'xDF.shape : {xDF.shape}')
    logger.warning(f'yDF.shape : {yDF.shape}')
   
    os.makedirs(outDir, exist_ok=True)
    
    xDFPath = f'{outDir}/{prefix}X.csv'
    xDF.to_csv(xDFPath) # index = none we want the index
    logger.warning(f'saved : {xDFPath}')    
    
    yDFPath = f'{outDir}/{prefix}Y.csv'
    yDF.to_csv(yDFPath, index=None)
    logger.warning(f'saved : {yDFPath}')    

    logger.warning("END")
    exitCode = 0 # no error
    sys.exit(exitCode)

################################################################################
if __name__ == "__main__" :
    main()      

    #inCommandLineArgsList=['-h']

    # rootDir = "/private/groups/kimlab/data/elife"
    # countPath = f"{rootDir}/elife_all_norm_counts_2023-05-18.csv"
    # metaPath = f"{rootDir}/elife_scaled_metaData_2023-05-18.csv"
    # inCommandLineArgsList=[
    #     '--countPath', countPath,
    #     '--categoryColName', "diagnosis",
    #     '--metaPath', metaPath,
    #     '--types', "Esophagus Cancer, Healthy donor",
    #     '--outDir', 'tmp/loadCounts.py.out',
    #     '--sampleIdColName', "sample_id",
    #     '--prefix', 'aedwip'
    # ]
    
    # inCommandLineArgsList = [
    #     '--countPath', '/scratch/aedavids/fagazi/selectIntracellularSamples.out/elifeGTExTCGA_count.csv',
    #     '--categoryColName', 'elifeDiagnosis',
    #     '--metaPath', '/scratch/aedavids/fagazi/selectIntracellularSamples.out/elifeGTExTCGA_metaData.csv',
    #     '--outDir', './tmp/./elifeGTExTCGA_Deseq.sh.out/Colorectal',
    #     '--sampleIdColName', 'sample_id',
    #     '--types', 'Colorectal_Cancer'
    # ]

    # main( inCommandLineArgsList )

