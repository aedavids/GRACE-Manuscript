"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-10-07
Copyright (c) 2020-2023, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 
    public functions:
        volcanoPlotWithBaseMean(resultsFilePath : str, title : str) -> plt.figure
        
        plotVolcanoPanel(
                            dataDict : dict[str,DiagnosticBundle],
                             dataSet : str,
                             title: str = "") -> plt.Figure:
"""

import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np


# local imports
extraErrorMsg = "git pull extracellularRNA, add 'extraCellularRNA' amd 'extraCellularRNA/src'  to PYTHONPATH"

try :
    from bme263DataVis.utilities import MatPlotLibUtilities
    from bme263DataVis.volcanoPlot import loadData, VolcanoPlot
except ModuleNotFoundError as e:
    emsg = f'{e}  {extraErrorMsg}, '
    raise ImportError(emsg)

from plots.plotPanel import make_panel
from archive.dataTypes import DiagnosticBundle

###############################################################################
def volcanoPlotWithBaseMean(resultsFilePath : str, title : str) -> plt.figure :
    '''
    ref: extraCellularRNA/src/bme263DataVis/volcanoPlot.py main()
    
    creates a plot of all values colored by base mean
    
    The reference version only color points wiht lfc > threshold 
    how ever the threshold value is not passed as a parameter
    
    plus we have a realtively small number of features we want to
    color all the points
    '''

    mplu = MatPlotLibUtilities()
    mplu.loadStyle()
    
    # loadData returns a _volcanoPlotData
    # it filters the data into 2 sets
    # "abundent "x"
    volcanoPlotData = loadData( resultsFilePath )


    # set up figure
    # standard paper size is 8.5 inches x 11 inches
    pageWidthInInches = 4
    pageHeightInInches = 3
    fig = plt.figure( figsize=( pageWidthInInches, pageHeightInInches ) )

    # create color map 
    panelHeightInInches = 2
    bottomRelativeSize = 0.2
    leftRelativeSize = 0.9
    colorBarPanel = mplu.createPanel( fig,
                        3 / 16, panelHeightInInches,
                        leftRelativeSize, bottomRelativeSize )
    

    # set up the base mean color gradient legend
    label = "base mean"
    numSteps = 10  # 20  # 4  # 1000 # 4 #50 #20
    volcanoPlot = VolcanoPlot( mplu )
    
    # python issue, we can not pass a generator because
    #  it is not indexable
    # yellow = tuple( ( c / 255 for c in [255, 255, 84] ) )
    # red = tuple( ( c / 255 for c in [255, 0, 0] ) )    
    
    lowColor = tuple(c / 255 for c in [255, 0, 0] ) # red
    highColor = tuple(( c / 255 for c in [0, 0, 255] )) # blue
    allX = volcanoPlotData.x + volcanoPlotData.abundantX
    allY = volcanoPlotData.y + volcanoPlotData.abundantY
    allBaseMean = volcanoPlotData.baseMean + volcanoPlotData.abundantBaseMean

    allBaseMeanMin = np.min(allBaseMean)
    allBaseMeanMax = np.max(allBaseMean)
    #print(f'aedwip xMin : {xMin}, xMax : {xMax}')
    colorMapTuple = volcanoPlot.createColorMapLedgend( colorBarPanel,
                                                        lowColor, highColor,
                                                        allBaseMeanMin,
                                                        allBaseMeanMax,
                                                        numSteps, label )

    # create the color list data
    RList, GList, BList = colorMapTuple
    # colorByValues = mplu.getColors( volcanoPlotData.abundantBaseMean,
    #                                RList, GList, BList )
    colorByValues = mplu.getColors( allBaseMean,
                                   RList, GList, BList )

    # black = ( 0, 0, 0 )
    # allColors = [black] * len( volcanoPlotData.x ) + colorByValues
    allColors = colorByValues
  

#     colorByValues = mplu.getColors( dataList, RList, GList, BList )

    # plot the main volcano plot
    panelWidthInInches = 2
    leftRelativeSize = 0.2
    volcanoPanel = mplu.createPanel( fig,
                        panelWidthInInches, panelHeightInInches,
                        leftRelativeSize, bottomRelativeSize )

    volcanoPlot.plot( volcanoPanel, allX, allY, allColors )

    if title:
        volcanoPanel.set_title( title, fontsize=8 )  # arial is not installed on courtyard, default font is huge
        
    return fig

################################################################################
def plotVolcanoPanel(
                            dataDict : dict[str,DiagnosticBundle],
                             dataSet : str,
                             title: str = "") -> plt.Figure:
    """ 
    TODO
    
    arguments
        dataSets should a one of the items in ["simTrain", "simVal", "simTest", "real"]
        
    typically we only generate volcanoPlots on simTrain because all the modelBudles are 
    evaluated using the same features sets.
    """
    validDataSets = ["simTrain", "simVal", "simTest", "real"]
    if not dataSet in validDataSets :
        logger.error(f"invalide dataSet : {dataSet} should be key with value {validDataSets}")
        return
    
    volcanos = []
    # plots = []
    for diganosis, diagnosticBundle in dataDict.items():
        mb = getattr(diagnosticBundle, dataSet)
        volcanos.append( mb.volcanoFig )
        
    titles = None #[ 'aaaa', 'bbbb', 'Ccccccccccc', 'd', 'TTTTTTTTTTTTT']
    panel = make_panel(volcanos, ncols=3, mode="uniform", cm_shared_scale=True, hide_colorbars=True, 
                       titles=titles,
                       displayAsPrecentage=True
                       )
    # figTitle = "dataSet"
    panel.suptitle( title )
    plt.show()    
    
    return panel