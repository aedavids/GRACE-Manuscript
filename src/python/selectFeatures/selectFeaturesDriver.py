#
# 
# main(): 
# ref: 
#   extraCellularRNA/deconvolutionAnalysis/python/pipeline/upstreamPipeline.py
#   extraCellularRNA/deconvolutionAnalysis/python/pipeline/dataFactory/test/testIntegration.py
#
# Andrew E. Davidson
# aedavids@ucsc.edu
# 2025-09-01
#
# Copyright (c) 2020-2023, Regents of the University of California All rights reserved.
# https://polyformproject.org/licenses/noncommercial/1.0.0
#

# upstreamPiplineCLI CommandLine display the doc string 
'''
    prepares DESeq and transcript count files so that we run CIBERSORTx
        1. select genes of interest
        2. create upset plots and find interesection elements
        3. create CIBERSORTx signatureGenes matrix
        4. create CIBERSORTx mixture matrix    
    
    See extraCellularRNA/deconvolutionAnalysis/python/pipeline/dataFactory/test/testIntegration.py

'''
import glob
import importlib
import logging
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os

from selectFeatures.driver import runSelectGenesOfInterest
from selectFeatures.signatureGeneConfig import SignatureGeneConfiguration
from plots.upsetPlotDataFactory import UpsetPlotDataFactory
from selectFeatures.selectFeaturesDriverCLI import SelectFeaturesDriverCommandLine

import pandas as pd
import pprint as pp

import sys
from plots.upsetPlots import UpsetPlot

# global variables use by PipelineCommandLine
__all__ = []
__version__ = 0.1
__author__ = "Andrew Davidson aedavids@ucsc.edu"
__date__ = '2023-11-13'
__updated__ = '2025-09-01'


################################################################################
def _step1(logger           : logging.Logger,
           deseqResultsDir  : str,
           findModule       : str,
           outDir           : str,
           resultsSuffix   : str, 
           vargs            : list
           ) -> tuple[SignatureGeneConfiguration, dict, list ]:
    '''
    step 1.
        - create user defined SignatureGeneConfiguration obj
        - run SignatureGeneConfiguration.findGenes() on all candidate results file
        
    arguments 
        resultsSuffix should be "results" or "lfcShrink"
    
    returns 
        (sgc, selectedGeneSetsDict, outFileList)
    '''
    logger.info(f'BEGIN')
    
    if not resultsSuffix in ("results", "lfcShrink"):
        logger.error(f' resultsSuffix : {resultsSuffix} is in valid, must be either "results" or "lfcShrink"')
        sys.exit(1)
        
    try:
        logger.info('step 1: create user defined SignatureGeneConfiguration obj')
        myModule = importlib.import_module(findModule, package=None)
        sgc = myModule.createSignatureGeneConfig(outDir, vargs)
        logger.info(f'sgc : {sgc}')
        # AEDWIP logger statement only works if '--findModule  pipeline.dataFactory.test.exampleCreateSignatureGeneConfig'
        #retStr = sgc.testDynamicalyloadedModules()
        #logger.info(f"sgct.testDynamicalyloadedModules() : {retStr}")

    except ImportError as e:
        emsg = f'unable to dynamically import {findModule} exception: {e}'
        logger.error(emsg)
        sys.exit(1)  

    #
    # get a list of deseq results file we want to select gene signatures from
    #
    # deseqResultFiles = os.path.join(deseqResultsDir, "*.results")
    # 2/28/26 add '**' to pattern. This matches and directory
    # at any depth
    pattern = f'**/*.{resultsSuffix}'
    deseqResultFiles = os.path.join(deseqResultsDir, pattern)
    
    # 2/28/26 add glop recusive so that we can create multi classification
    # data sets from binary deseq results
    # the results files are in directories like Lung/*.results
    deseqResultFileList = glob.glob(deseqResultFiles, recursive=True)
    if len(deseqResultFileList) == 0 :
        logger.error(f'no deseqResultFiles found in {deseqResultsDir}')
        sys.exit(1)
        
    logger.warning(f'deseqResultFileList : \n{pp.pformat(deseqResultFileList)}')


    #
    # runSelectGenesOfInterest
    # filters/selects the genes of interest from each of the 1vsAll DESeq  results file
    # the selectedGeneSetsDict contains a key/value for each file
    # the outFileList is a file with a list of path to filtered DESeq results file.
    # ie only DESeq results we are interested
    #

    (selectedGeneSetsDict, outFileList) = runSelectGenesOfInterest(
                                    signatureGeneConfig=sgc, 
                                    candidateSignatureFileList=deseqResultFileList 
                                    ) 

    logger.info("END")
    return (sgc, selectedGeneSetsDict, outFileList) 

################################################################################
def _savePlot(logger : logging.Logger,
                upsetPlot : UpsetPlot,
                vargs : str,
                upsetPlotOutDir : str,
                ):
    '''
    TODO
    '''
    logger.warning(f'create upset plot for {vargs}')
    # fig, pltDict = upsetPlot.configurablePlot(show_counts=True, min_degree=min_degree, max_degree=max_degree)
    try:
        plotPath = upsetPlot.savePlot( outdir=upsetPlotOutDir,
                                extraFileNameParts=vargs)
        logger.warning(f'plotPath : {plotPath}')  
        
    except ValueError as ve :
        logger.warning(f'ValueError upset plot of all intersections was to big to save')
        logger.warning(ve)

    except Exception as e:
        logger.warning(f'Exception upset  not saved')
        logger.warning(e)

    except BaseException as be: 
        # stepped into upset plot save. looks like it might throw a BaseException.
        # I would think Exception would catch it
        logger.warning(f'BaseException: upset plot of all intersections was to big to save')
        logger.warning(e)

################################################################################
def _step2(logger           : logging.Logger,
           outDir : str,
           selectedGeneSetsDict : dict,
           sgc : SignatureGeneConfiguration,
           ) -> dict[str, tuple[Figure, dict[str, Axes]]]:
    '''
    # create upset plots and find interesection elements
    '''
    logger.info("BEGIN")
    
    plots: dict[str, tuple[Figure, dict[str, Axes]]] = {}

    if len(selectedGeneSetsDict.keys()) <= 1:
        logger.warning("upset plot was not created because there is only a single set")
        return plots
    
    logger.info('step 2 create plots and find intersection elements')
    upsetFactory = UpsetPlotDataFactory()
    logger.debug(f'AEDWIP debug selectedGeneSetsDict\n{pp.pformat(selectedGeneSetsDict)}')
    upsetPlotDataDF, geneSetsDict = upsetFactory.createUpsetPlotDataFromSetDict(selectedGeneSetsDict)
    logger.info(f'upsetPlotDataDF:\n{pp.pformat(upsetPlotDataDF)}')

    upsetPlot = UpsetPlot(signatureGeneConfig=sgc, 
                    geneSetsDict=geneSetsDict, 
                    geneSetsUpsetPlotData=upsetPlotDataDF)
    
    #
    # find all the intersections
    #
    intersectionDict = upsetPlot.findIntersectionElements()  
    upsetPlotOutDir = os.path.join(outDir, "upsetPlot.out")
    intersectionDictPath = upsetPlot.saveInteresection(upsetPlotOutDir, intersectionDict) 
    logger.warning(f'intersectionDictPath : {intersectionDictPath}')

    #
    # create a plot of all intersections that are not empty
    # 
    logger.warning(f'create upset plot for all')
    fig, pltDict = upsetPlot.configurablePlot(show_counts=True)
    vargs = 'allDegrees'
    plots[vargs] = (fig, pltDict)
    _savePlot(logger, upsetPlot, vargs, upsetPlotOutDir)

     # Create degree-specific plots.
    degrees = upsetPlot.findDegrees(intersectionDict)
    logger.info(f'degrees: {degrees}')

    lumpMinDegree = 0
    if len(degrees) > 10:
        # create plots for the first 10 degrees
        # we will create another plot for the remaining
        lumpMinDegree = degrees[10]
        degrees = degrees[0:10]

    for degree in degrees:
        min_degree = degree
        max_degree = degree
        vargs = f"min_degree={min_degree},_max_degree={max_degree}"

        fig, pltDict = upsetPlot.configurablePlot(show_counts=True, min_degree=min_degree, max_degree=max_degree)
        plots[vargs] = (fig, pltDict)
        _savePlot(logger, upsetPlot, vargs, upsetPlotOutDir)

    if lumpMinDegree > 0:
        vargs = f"min_degree={min_degree}"
        logger.warning(f'create upset plot for {vargs}')
        fig, pltDict = upsetPlot.configurablePlot(show_counts=True, min_degree=lumpMinDegree,)
        # plotPath = upsetPlot.savePlot( outdir=upsetPlotOutDir,
        #                         extraFileNameParts=vargs)      
        plots[vargs] = (fig, pltDict)    
        _savePlot(logger, upsetPlot, vargs, upsetPlotOutDir)

    logger.info("END")
    return plots

################################################################################
def main(
        inCommandLineArgsList=None,
        *,
        returnPlots: bool = False,
    ) ->  dict[str, tuple[Figure, dict[str, Axes]]] :
    '''
    ref: pipeline/dataFactory/test/testInjection.py 

    ```
    # run using unit test data
    cd /private/home/aedavids/extraCellularRNA/deconvolutionAnalysis/python
    $ see pipelineCLITest.sh
    ```
    '''
    # we only configure logging in main module
    #loglevel = "WARN"
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

    #
    # parse command line arguments
    #
    cli = SelectFeaturesDriverCommandLine( 
                            version=__version__ , 
                            author=__author__ ,
                            date=__date__, 
                            update=__updated__)

    if inCommandLineArgsList is None:
        cli.parse()
    else:
        cli.parse( inCommandLineArgsList )

    logger.warning(f'command line arguments : {cli.args}')

    deseqResultsDir = cli.args.deseqResultsDir
    findModule      = cli.args.findModule
    outDir          = cli.args.outDir
    resultsSuffix   = cli.args.resultsSuffix
    vargs           = cli.args.vargs

    logger.info(f"vargs: {vargs}")

    #
    # get upset plot input files
    #
    sgc, selectedGeneSetsDict, outFileList = _step1(logger, deseqResultsDir, findModule, outDir, resultsSuffix, vargs)
    
    #
    # create a list of features
    #
    tmpSeries = pd.Series()
    for key, resultDF in selectedGeneSetsDict.items():
        logger.info("fselectedGeneSetsDict key : {key}")
        nameSeries = resultDF.loc[:, "name"]
        
        if tmpSeries.empty:
            tmpSeries = nameSeries
        else:
            tmpSeries = pd.concat([tmpSeries, nameSeries])
            
    # use set to remove duplicates
    retSet = set( tmpSeries.to_list() )
    
    # save list of genes
    geneListPath = f'{outDir}/genes.txt'
    if len(retSet) == 0:
        logger.error(f'no genes found')
        sys.exit( 1 ) 
        
    pd.Series( sorted(retSet) ).to_csv( geneListPath, index=False, header=False)
    logger.warning(f'saving {geneListPath}')
    #
    # create upset plots and find interesection elements
    #
    retUpsetPlotDict = _step2(logger, outDir, selectedGeneSetsDict, sgc)

    logger.warning("END")
    if returnPlots:
        return retUpsetPlotDict

    # Preserve the original behavior for existing callers.
    sys.exit(0)

################################################################################
if __name__ == '__main__':
    main()

    # # inCommandLineArgsList=['-h']
    
    # baseName = os.path.basename( __file__)
    # outDir = f'tmp/{baseName}.out'

    # # arguments to pass to BestSignatureGeneConfig.__init__()
    # topN=25
    # title=f"best{topN}" # do not uses spaces, title will be part of file paths
    # vargs = [
    #     '--design', 'tilda_gender_diagnosis',
    #     '--padjThreshold', '0.01',
    #     '--lfcThreshold', '2.0',
    #     '--dataSetName', 'elife',
    #     '--number', str(topN),
    #     '--title', title,
    #     '--localCacheRoot', outDir
    # ]

    # inCommandLineArgsList=[
    #     '--deseqResultsDir', 'src/python/selectFeatures/test/data',
    #     '--findModule', 'selectFeatures.createBestCreateSignatureGeneConfig',
    #     '--out', outDir,
    #     '--vargs', *vargs # <-- splat the list so they become multiple args
    # ]   
    
    # main( inCommandLineArgsList  )
