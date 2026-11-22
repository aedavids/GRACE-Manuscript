
#!/usr/local/bin/python3
# encoding: utf-8
'''
volcanoPlot.py -- shortdesc create a volcano plot

It defines classes_and_methods

@author:     Andrew Davidson

@copyright:  2022 organization_name. All rights reserved.

@license:    license

@contact:    aedavids@ucsc.edu
@deffield    updated: Updated
'''

''' 
Table of Content:

    class VolcanoPlot( object )
    
    def loadData( inputFile, numHeaderLines, teGeneNamesSet=None ):
    
    def createVolcanoPlot (
        title : str,
        inputFilePath : str,
        numHeaderLines : int,
        geneNameFilePath : str = None,
        label : bool = False,
        pageWidthInInches : int = 3,
        pageHeightInInches : int = 3
    ) -> plt.figure :
    
    def main( inComandLineArgsList=None )
    
    #
    # panel function to create a fig with multple volcano plots
    #
    @dataclass
    class VolcanoPlotData
    
    def createVolcanoData(
        resultsFiles : list[str],
        geneNamePath : str,
    ) -> dict[str, VolcanoPlotData]:
    
    def debugVolcanoDataDict( vdd :  dict[str, VolcanoPlotData] ) -> None 
    
    def volcanoPlotsWithHighlights(
        deseqResultsFilePath : str,
        # imgOutDir : str,
        geneNamesPath : str = None,
        title : str = None,
        numHeaderLines : int = 8,
        pageWidthInInches : int = 4,
        pageHeightInInches : int = 4,
        label : bool = False,
    ) -> plt.figure:
    
    def createVolcanoPanel(
        vdd : dict[str, VolcanoPlotData],
        numHeaderLines : int=8,
    ) -> plt.figure:
'''

from dataclasses import asdict, dataclass

import logging

import colorama
logger = logging.getLogger(__name__)

from plots.plotPanel import make_panel
from plots.matPlotLibUtilities import MatPlotLibUtilities
from plots.volcanoPlotCommandLine import VolcanoPlotCommandLine
from plots.DESeqSelect import DESeqSelect

import matplotlib.pyplot as plt
import numpy as np

__all__ = []
__version__ = 0.1
__date__ = '2020-05-26'
__updated__ = '2020-05-26'
__user_name__ = 'Andrew E. Davidson'


###############################################################################
class VolcanoPlot( object ):
    '''
    classdocs
    '''

    ###############################################################################
    def __init__( self, mplu ):
        '''
        arguments:
            mplu: object of type MatPlotLibUtilities
            panel
        '''
        self.mplu = mplu
#         self.panel = panel

#     ###############################################################################
#     def createColorMapLedgend( self, colorBarPanel, minValueInDataCordinates,
#                          maxValueInDataCoordinates, numSteps, label ):
#         '''
#         AEDWIP
#         '''
#         # python issue, we can not pass a generator because
#         #  it is not indexable
#         yellow = tuple( ( c / 255 for c in [255, 255, 84] ) )
#         red = tuple( ( c / 255 for c in [255, 0, 0] ) )
#         green = tuple( ( c / 255 for c in [0, 255, 0] ) )
#         blueV = tuple( ( c / 255 for c in [  0, 9, 240] ) )
#         blue = tuple( ( c / 255 for c in [0, 0, 255] ) )
#
#         colorMapTuple = self.mplu.createColorBar( colorBarPanel,
#                                                  minValueInDataCordinates,
#                                                  maxValueInDataCoordinates,
#                                                   yellow,
#                                                   red, numSteps, yLabel=label )
#         # RList, GList, BList = colorMapTuple
#         return colorMapTuple

    ###############################################################################
    def createColorMapLedgend( self, colorBarPanel, lowerColorTup, upperColorTup,
                                 minValueInDataCordinates, maxValueInDataCoordinates, numSteps, label ):
        '''
        AEDWIP
        '''

        colorMapTuple = self.mplu.createColorBar( colorBarPanel,
                                                 minValueInDataCordinates,
                                                 maxValueInDataCoordinates,
                                                  lowerColorTup,
                                                  upperColorTup, numSteps, yLabel=label )
        # RList, GList, BList = colorMapTuple
        return colorMapTuple

    ###############################################################################
    def plotDepreciated( self, panel, xList, yList, colorByValues, volcanoPlotData, plotNames ):
        '''
        04/30/2026. seems like we are getting over plotting
        arguments:
            panel:
                type matplotlib.pyplot.axes

            x, y:
                array like object

            colorByValue:
                array type object of color tuples.  example color = (0.5, 0.3, 07)
                default = None
                
            volcanoPlotData
                12/2/22 rework: we want to change marker size to make eassier to see TE

            plotNames:
                boolean: if true print the name of the colored values
        '''
        # make sure all points are plotted
        xMin = np.floor( np.min( xList ) )
        xMax = np.ceil( np.max( xList ) )

        # want the x axis to be balanced. It is posible min = -10 and max = 5
        limit = np.max( np.abs([xMin, xMax]) )
        panel.set_xlim( limit * -1  , limit )

        # make sure all points are plotted
        yMin = np.floor( np.min( yList ) )
        yMax = np.ceil( np.max( yList ) )
        panel.set_ylim( yMin, yMax )

        # https://matplotlib.org/tutorials/text/mathtext.html
        panel.set_xlabel( r'$log_2(fold\ change)$' )
        panel.set_ylabel( r'$-log_{10}(adj\ p\ value)$' )

        # if colorByValues is not None:
        #     print( "AEDWIP scatter colorByValues\n{}".format( colorByValues ) )
        #     # scatter markersize is area
        #     plotMarkersize = 0.75  # 0.6  # 0.75 #0.5 #1 #0.75 #0.5
        #
        #     plotMarkerArea = np.pi * ( ( plotMarkersize / 2 ) ** 2 )
        #     scatterMarkersize = plotMarkerArea * 2  # strange did not look good with * 2
        #     panel.scatter( xList,
        #                   yList,
        #                   s=scatterMarkersize,
        #                   facecolor=colorByValues,
        #                   linewidth=0,
        #                   alpha=0.8 )  # alpha=0.3
        #
        # else:
        #     print( "AEDWIP plot()" )
        #     # plot is faster than scatter how ever does not allow points to be
        #     # indvidually colored
        #     panel.plot( xList,
        #                yList,
        #                marker='o',
        #                markerfacecolor='black',  # (56/255,66/255,156/255),
        #                markeredgecolor='black',
        #                markersize=1.5,  # diameter of mark
        #                markeredgewidth=0,
        #                linewidth=0,
        #                alpha=0.3 )
        
        for i in range(len(xList)):
            color = 'black'
            markersize=1.5 #1.5,  # diameter of mark
            alpha=0.3
            # zorder is used to for our TE points and txt to be on top
            # of all other planes 
            zorder = 1           
            if volcanoPlotData.isHighlighted[i]:
                print(f'AEDWIP 159 isHighlighted i : {i}')
                # color =' #3b528b' #blues https://waldyrious.net/viridis-palette-generator/
                color = '#a2da37' # greenish
                # color = 'red'
                markersize= markersize + markersize * 0.25 #2,  # diameter of mark
                alpha = 0.8 
                zorder = 1

                # print(f'AEDWIP line : 168 boolean plotNames : {plotNames}')
                if plotNames:
                    # add gene name to plot
                    gn = volcanoPlotData.geneName[i].strip()
                    labelY = yList[i]
                    # print(f'AEDWIP l : 171 labelY : {labelY} gn : {gn}')
                    if xList[i] > 0:
                        # up regulated
                        ha = 'left'
                        labelX = xList[i]
                        gn = "  " + gn
                    else:
                        ha = 'right'
                        labelX = xList[i]
                        gn = gn + "  "
                        
                    panel.text(labelX, labelY, gn, fontsize=2, 
                            horizontalalignment=ha, 
                            verticalalignment='center',
                            color=color,
                            zorder=zorder)
                
            panel.plot( xList[i],
                       yList[i],
                       marker='o',
                       # markerfacecolor='black',  # (56/255,66/255,156/255),
                       # markeredgecolor='black',
                       markerfacecolor=color,
                       markeredgecolor=color,
                       #markersize=1.5,  # diameter of mark
                       markersize=markersize, 
                       markeredgewidth=0,
                       linewidth=0,
                       alpha=alpha,
                       zorder=zorder )
            
        # # add gene names to TE  points
        # for i, in range(len(xList)):
        #     x = xList[i]
        #     y = yList[i]
        #     if 
        #     gn = geneName.strip() + " "
        #     panel.text(x, y, gn, fontsize=6,
        #                horizontalalignment='right',
        #                verticalalignment='center')        

#
#         self.panel.tick_params(bottom=True, labelbottom=True,
#                           left=True, labelleft=True,
#                           right=False, labelright=False,
#                           top=False, labeltop=False)


   ###############################################################################
    def plot( self, panel, xList, yList, colorByValues, volcanoPlotData, plotNames,
                     color = '#a2da37' # greenish aedwip
                    ):
        '''
        arguments:
            panel:
                type matplotlib.pyplot.axes

            x, y:
                array like object

            colorByValue:
                array type object of color tuples.  example color = (0.5, 0.3, 07)
                default = None
                
            volcanoPlotData
                12/2/22 rework: we want to change marker size to make eassier to see TE

            plotNames:
                boolean: if true print the name of the colored values
                
            color 
                9/21/26 hack for manuscript
                is a string version of a hexadecimal color code
                example '#a2da37' # greenish 
                
                it used insead of colorByValues
        '''
        # make sure all points are plotted
        # xMin = np.floor( np.min( xList ) )
        # xMax = np.ceil( np.max( xList ) )

        # # want the x axis to be balanced. It is posible min = -10 and max = 5
        # limit = np.max( np.abs([xMin, xMax]) )
        # panel.set_xlim( limit * -1  , limit )

        # # make sure all points are plotted
        # yMin = np.floor( np.min( yList ) )
        # yMax = np.ceil( np.max( yList ) )
        # panel.set_ylim( yMin, yMax )
        
        # this fixed clipping bug 9/19/26
        # Keep the x-axis symmetric, with extra room around the points.
        limit = max(float(np.max(np.abs(xList))), 1.0)
        panel.set_xlim(-1.05 * limit, 1.05 * limit)
        
        # Add padding above and below the data.
        yMin = float(np.min(yList))
        yMax = float(np.max(yList))
        yPadding = 0.05 * max(yMax - yMin, 1.0)
        panel.set_ylim(yMin - yPadding, yMax + yPadding)

        # https://matplotlib.org/tutorials/text/mathtext.html
        panel.set_xlabel( r'$log_2(fold\ change)$' )
        panel.set_ylabel( r'$-log_{10}(adj\ p\ value)$' )

        
        # collect the points by the zorder they should be rendered
        # so that we plot all the TE's on top
        # should make zorder more consistent
        xPlotList = list()
        yPlotList = list()
        
        xPlotTEList = list()
        yPlotTEList = list()
       
        # label to the right of point
        TELabelRightList = list()
        xPlotTERightList = list()
        yPlotTERightList = list()
        
        # label to the left of point
        TELabelLeftList = list()
        xPlotTELeftList = list()
        yPlotTELeftList = list()
        
        logger.warning(f'AEDWIP TODO move coloring to createVolcanoPlot so we can add a legend in future see createVolcanoPlot()')
        for i in range(len(xList)):
            # color = 'black'
            # markersize=1.5 #1.5,  # diameter of mark
            markerSize = 6 #4 #2
            # alpha=0.3
            # # zorder is used to for our TE points and txt to be on top
            # # of all other planes 
            # zorder = 1     
            fontSize = 6 #4 # 2
                 
            if volcanoPlotData.isHighlighted[i]:
                xPlotTEList.append(xList[i])
                yPlotTEList.append(yList[i])
                
                if plotNames:
                    # add gene name to plot
                    gn = volcanoPlotData.geneName[i].strip()
                    
                    if xList[i] > 0:
                        # ha = 'left'
                        # labelX = xList[i]
                        xPlotTELeftList.append(xList[i])
                        yPlotTELeftList.append(yList[i])
                        gn = "  " + gn
                        TELabelLeftList.append(gn)
                    else:
                        # ha = 'right'
                        xPlotTERightList.append(xList[i])
                        yPlotTERightList.append(yList[i])
                        gn = gn + "  "
                        TELabelRightList.append(gn)
                    
            else :
                xPlotList.append(xList[i])
                yPlotList.append(yList[i])
                
        # plot the background points first 
        panel.plot( xPlotList,
                    yPlotList,
                    marker='o',
                    # markerfacecolor='black',  # (56/255,66/255,156/255),
                    # markeredgecolor='black',
                    markerfacecolor='black',
                    markeredgecolor='black',
                    #markersize=1.5,  # diameter of mark
                    markersize=markerSize, 
                    markeredgewidth=0,
                    linewidth=0,
                    alpha=0.3,
                    zorder=1 ) # low values are on bottom/back
 
        # plot TE's on top
        # color =' #3b528b' #blueshttps://waldyrious.net/viridis-palette-generator/
       # color = '#a2da37' # greenish aedwip
        # color = 'red'
        panel.plot( xPlotTEList,
                    yPlotTEList,
                    marker='o',
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=markerSize,  # diameter of mark
                    markeredgewidth=0,
                    linewidth=0,
                    alpha=0.8,
                    zorder=2 ) # low values are on bottom/back
           
        # plot names on top
        if plotNames:
            
            for i in range(len(xPlotTELeftList)):
                panel.text(xPlotTELeftList[i], yPlotTELeftList[i], TELabelLeftList[i], 
                        fontsize=fontSize, 
                        horizontalalignment='left', 
                        verticalalignment='center',
                        color=color,
                        zorder=2)
            
            for i in range(len(xPlotTERightList)):
                panel.text(xPlotTERightList[i], yPlotTERightList[i], TELabelRightList[i], 
                        fontsize=fontSize, 
                        horizontalalignment='right', 
                        verticalalignment='center',
                        color=color,
                        zorder=2)

            
     

########################################################################
class _VolcanoPlotData:
    '''
    future proof private class use to manage data.

    avoid return list of parallel lists. Its hard to keep track of of kind of data is at
    each given position. using a class lets us name the various lists. Reduces refactor bugs
    '''

    def __init__( self ):
        # self.abundantX = []
        self.x = []
        
        # if true color the point and possibly add text label
        self.isHighlighted = []

        # self.abundantY = []
        self.y = []
        # self.abundant = []

        # self.abundantBaseMean = []
        # self.baseMean = []
        #
        # # the xtick values for color gradient
        # self.minAbundanceValue = 0
        # self.maxAbundanceValue = 0

        self.geneName = []  # if not a TE name is ""

########################################################################
def loadDataDeprecated( inputFile, numHeaderLines, teGeneNamesSet=None ):
    '''
    returns a _VolcanoPlotData object
    '''
    ret = _VolcanoPlotData()

    dataLoader = DESeqSelect( inputFile )
    geneNamesNP, baseMeanNP, xlog2FoldChangeNP, yNeglog10pValueNP = dataLoader.readVolcanoPlotData( numHeaderLines )

    # find abundant values
    # create two data sets, values that are abundant and should be colored
    # and data that should be ploted as black points
    std = np.std( baseMeanNP )
    mean = np.mean( baseMeanNP )
    threshold = mean + 2 * std
    #print( "AEDWIP mean:{} std:{} threshold:{}".format( mean, std, threshold ) )

    for i in range( len( baseMeanNP ) ):
        # bm = baseMeanNP[i]
        geneName = geneNamesNP[i]
        x = xlog2FoldChangeNP[i]
        y = yNeglog10pValueNP[i]
        # if bm >= threshold:
        #     ret.abundantX.append( x )
        #     ret.abundantY.append( y )
        #     ret.abundantBaseMean.append( bm )
        # else:
        #     ret.x.append( x )
        #     ret.y.append( y )
        #     ret.baseMean.append( bm )

        # append is really slow, we are not use abundance
        ret.x.append( x )
        ret.y.append( y )
        ret.geneName.append( geneName )
        if teGeneNamesSet:
            if geneName in teGeneNamesSet:
                #print( "AEDWIP geneName:{}".format( geneName ) )
                ret.isHighlighted.append( True )
            else:
                ret.isHighlighted.append( False )

#     print( "AEDWIP mean:{} std:{} minCut:{}".format( mean, std, mean + 2 * std ) )

    # find the range of tick mark values for the color gradient panel
    # ret.minAbundanceValue = int( np.floor( threshold ) )
    # ret.maxAbundanceValue = int( np.ceil( np.max( baseMeanNP ) ) )

    return ret


########################################################################
def loadData( inputFile, numHeaderLines, teGeneNamesSet=None ):
    '''
    for each geneName
        if geneName is in teGeneNamesSet
            isHighlighted is set to True
            
    returns a _VolcanoPlotData object
    '''
    ret = _VolcanoPlotData()

    dataLoader = DESeqSelect( inputFile )
    geneNamesNP, baseMeanNP, xlog2FoldChangeNP, yNeglog10pValueNP = dataLoader.readVolcanoPlotData( numHeaderLines )

    ret.x = xlog2FoldChangeNP
    ret.y = yNeglog10pValueNP
    ret.geneName = geneNamesNP
    ret.isHighlighted = [False] * len(xlog2FoldChangeNP)

    # find abundant values
    # create two data sets, values that are abundant and should be colored
    # and data that should be ploted as black points
    std = np.std( baseMeanNP )
    mean = np.mean( baseMeanNP )
    threshold = mean + 2 * std
    # print( "AEDWIP mean:{} std:{} threshold:{}".format( mean, std, threshold ) )

    #for i in range( len( baseMeanNP ) ):
    for i in range( len( xlog2FoldChangeNP ) ):
        geneName = geneNamesNP[i]
        if teGeneNamesSet:
            if geneName in teGeneNamesSet:
                ret.isHighlighted[i] = True
    return ret

########################################################################
def createVolcanoPlot (
        title : str,
        inputFilePath : str,
        numHeaderLines : int,
        geneNameFilePath : str = None,
        label : bool = False,
        pageWidthInInches : int = 3,
        pageHeightInInches : int = 3,
        color = '#a2da37' # greenish aedwip aedwip

    ) -> plt.figure :
    ''' 
    TODO
    
    arguments
        color
            9/21/26 hack for manuscript
            is a string version of a hexadecimal color code
            example '#a2da37' # greenish     
    '''
    
    mplu = MatPlotLibUtilities()
    # missing fonts on mustard mplu.loadStyle()

    #print( "DEBUG cli.args.geneNamesFile:{}".format(geneNameFilePath) )

    # read list of TE gene names. These are point we want to color
    teGeneNamesSet = None
    if geneNameFilePath:
        teGeneNamesSet = set()
        with open( geneNameFilePath ) as fd:
            for line in fd:
                geneName = fd.readline().strip()
                teGeneNamesSet.add( geneName )

    plotNames = False
    if label :
        plotNames = True

    volcanoPlotData = loadData( inputFilePath, numHeaderLines, teGeneNamesSet )

    # set up figure
    # standard paper size is 8.5 inches x 11 inches
    fig = plt.figure( figsize=( pageWidthInInches, pageHeightInInches ) )

    # # create color map
    panelHeightInInches = 2
    bottomRelativeSize = 0.2
    # leftRelativeSize = 0.9
    # colorBarPanel = mplu.createPanel( fig,
    #                     3 / 16, panelHeightInInches,
    #                     leftRelativeSize, bottomRelativeSize )
    #
    # # set up the base mean color gradient legend
    # label = "base mean"
    # numSteps = 10  # 20  # 4  # 1000 # 4 #50 #20
    volcanoPlot = VolcanoPlot( mplu )

#         # python issue, we can not pass a generator because
#         #  it is not indexable
    # yellow = tuple( ( c / 255 for c in [255, 255, 84] ) )
    red = tuple( ( c / 255 for c in [255, 0, 0] ) )
    # colorMapTuple = volcanoPlot.createColorMapLedgend( colorBarPanel,
    #                                                     yellow, red,
    #                                                     volcanoPlotData.minAbundanceValue,
    #                                                     volcanoPlotData.maxAbundanceValue,
    #                                                     numSteps, label )

    # create the color list data
    # RList, GList, BList = colorMapTuple
    # colorByValues = mplu.getColors( volcanoPlotData.abundantBaseMean,
    #                                RList, GList, BList )

    black = ( 0, 0, 0 )
    # allColors = [black] * len( volcanoPlotData.x ) + colorByValues
    allColors = [black] * len( volcanoPlotData.x )

    for i in range( len( volcanoPlotData.x ) ):
        if volcanoPlotData.isHighlighted[i]:
            allColors[i] = red

    # colorByValues = mplu.getColors( dataList, RList, GList, BList )

    # plot the main volcano plot
    panelWidthInInches = 2# 0.0 #0.2 # 2 9/19/26 clipping bug
    leftRelativeSize = 0.2 #0.0 # 0.2 9/19/26 clipping bug
    # volcanoPanel = mplu.createPanel( fig,
    #                     panelWidthInInches, panelHeightInInches,
    #                     leftRelativeSize, bottomRelativeSize )
    
    # volcanoPanel = mplu.createPanel( fig,
    #                     pageWidthInInches, pageHeightInInches,
    #                     leftRelativeSize, bottomRelativeSize )    
    
    volcanoPanel = fig.add_axes([0.20, 0.20, 0.72, 0.70])

    # put colored points at end of the list to avoid over plotting by black points
    # allX = volcanoPlotData.x + volcanoPlotData.abundantX
    # allY = volcanoPlotData.y + volcanoPlotData.abundantY
    # volcanoPlot.plot( volcanoPanel, allX, allY, allColors )
    
    # 5/3/26
    # aedwip. looks like we hacked the code. allColors is not currently used
    # plot does coloring. plot should not do coloring? we will want to have 
    # colorBarPanel 
    
    volcanoPlot.plot( volcanoPanel, volcanoPlotData.x, volcanoPlotData.y, 
                        allColors, volcanoPlotData, plotNames, color=color )

    # title = cli.args.title
    if title:
        volcanoPanel.set_title( title, fontsize=8 )  # arial is not installed on courtyard, default font is huge

    return fig

    

########################################################################
def main( inComandLineArgsList=None ):
    '''
    process command line arguments load data and  call createPlot()
    '''
    cli = VolcanoPlotCommandLine( __user_name__, __version__, __date__, __updated__ )
    if inComandLineArgsList is None:
        cli.parse()
    else:
        cli.parse()( inComandLineArgsList )

#     mplu = MatPlotLibUtilities()
#     # missing fonts on mustard mplu.loadStyle()

#     print( "DEBUG cli.args.geneNamesFile:{}".format( cli.args.geneNamesFile ) )

#     # read list of TE gene names. These are point we want to color
#     teGeneNamesSet = None
#     if cli.args.geneNamesFile:
#         teGeneNamesSet = set()
#         with open( cli.args.geneNamesFile ) as f:
#             for line in f:
#                 geneName = f.readline().strip()
#                 teGeneNamesSet.add( geneName )

#     plotNames = False
#     if cli.args.label :
#         plotNames = True

#     # print(f'############## AEDWIP plotNames:{plotNames}')

#     # print(teGeneNamesSet)

#     # print("\n\n******* '(GAAGGCA)n' in teGeneNamesSet: {}".format("(GAAGGCA)n" in teGeneNamesSet))

#     volcanoPlotData = loadData( cli.args.inputFile, cli.args.numHeaderLines, teGeneNamesSet )

#     # set up figure
#     # standard paper size is 8.5 inches x 11 inches
#     pageWidthInInches = 3  # 4
#     pageHeightInInches = 3
#     fig = plt.figure( figsize=( pageWidthInInches, pageHeightInInches ) )

#     # # create color map
#     panelHeightInInches = 2
#     bottomRelativeSize = 0.2
#     # leftRelativeSize = 0.9
#     # colorBarPanel = mplu.createPanel( fig,
#     #                     3 / 16, panelHeightInInches,
#     #                     leftRelativeSize, bottomRelativeSize )
#     #
#     # # set up the base mean color gradient legend
#     # label = "base mean"
#     # numSteps = 10  # 20  # 4  # 1000 # 4 #50 #20
#     volcanoPlot = VolcanoPlot( mplu )

# #         # python issue, we can not pass a generator because
# #         #  it is not indexable
#     # yellow = tuple( ( c / 255 for c in [255, 255, 84] ) )
#     red = tuple( ( c / 255 for c in [255, 0, 0] ) )
#     # colorMapTuple = volcanoPlot.createColorMapLedgend( colorBarPanel,
#     #                                                     yellow, red,
#     #                                                     volcanoPlotData.minAbundanceValue,
#     #                                                     volcanoPlotData.maxAbundanceValue,
#     #                                                     numSteps, label )

#     # create the color list data
#     # RList, GList, BList = colorMapTuple
#     # colorByValues = mplu.getColors( volcanoPlotData.abundantBaseMean,
#     #                                RList, GList, BList )

#     black = ( 0, 0, 0 )
#     # allColors = [black] * len( volcanoPlotData.x ) + colorByValues
#     allColors = [black] * len( volcanoPlotData.x )

#     for i in range( len( volcanoPlotData.x ) ):
#         if volcanoPlotData.isHighlighted[i]:
#             allColors[i] = red

#     # colorByValues = mplu.getColors( dataList, RList, GList, BList )

#     # plot the main volcano plot
#     panelWidthInInches = 2
#     leftRelativeSize = 0.2
#     volcanoPanel = mplu.createPanel( fig,
#                         panelWidthInInches, panelHeightInInches,
#                         leftRelativeSize, bottomRelativeSize )

#     # put colored points at end of the list to avoid over plotting by black points
#     # allX = volcanoPlotData.x + volcanoPlotData.abundantX
#     # allY = volcanoPlotData.y + volcanoPlotData.abundantY
#     # volcanoPlot.plot( volcanoPanel, allX, allY, allColors )
#     volcanoPlot.plot( volcanoPanel, volcanoPlotData.x, volcanoPlotData.y, 
#                         allColors, volcanoPlotData, plotNames )

#     title = cli.args.title
#     if title:
#         volcanoPanel.set_title( title, fontsize=8 )  # arial is not installed on courtyard, default font is huge

    title = cli.args.title
    createVolcanoPlot(
                title,
                inputFilePath=cli.args.inputFile,
                numHeaderLines=cli.args.numHeaderLines,
                geneNameFilePath=cli.args.geneNamesFile,
                label=cli.args.label,
        )
    
    # save image
    outputFile = cli.args.outputFile
    # png is an uncompressed bitmap format
    # output format is determined by output file name's suffix '.png'
    # plt.savefig( outputFile, dpi=600 )  # BME163 style sheet should set dpi=600
    logger.warning(f'AEDWIP 8/5/26 changed code to save png and pdf check file names and types')
    cleanOutputFile = stripMatplotlibSuffix(outputFile)
    plt.savefig( cleanOutputFile, dpi=600, format='png')  # BME163 style sheet should set dpi=600
    plt.savefig( cleanOutputFile, dpi=600, format='pdf')  # BME163 style sheet should set dpi=600

########################################################################
@dataclass
class VolcanoPlotData :
    ''' 
    TODO 
    '''
    resultsPath : str
    diagnosis : str
    geneNamePath : str = None
    
########################################################################
def createVolcanoData(
        resultsFiles : list[str],
        geneNamePath : str,
    ) -> dict[str, VolcanoPlotData]:
    ''' 
    returns a dictionary with diagnosis as the key
    the values are VolcanoPlotData objects
    
    the geneName identify gene loci, name, id that will be colored
    
    arguments : 
        resultsFile :
            example :
            [
            '/private/groups/kimlab/data/elifeGTExTCGA/cromwell.output/Colorectal/intra_vs_all.results',
            '/private/groups/kimlab/data/elifeGTExTCGA/cromwell.output/Esophagus/intra_vs_all.results'
            ]
            
        geneNamePath :
            example :
                geneNamePath = createTENames(tmpDir, repeatDF_G)
                
    '''
    retDict = dict()
    for resultFile in resultsFiles:
        tokens = resultFile.split("/")
        token = tokens[-2]
        diagnosis = token + " Cancer"
        
        if tokens[-1] == "Healthy":
            diagnosis = token + " Donor"
        
        vpd = VolcanoPlotData(
                    resultFile,
                    diagnosis,
                    geneNamePath=geneNamePath
                    )
        
        retDict[diagnosis]  = vpd
        
    return retDict

########################################################################
def debugVolcanoDataDict( vdd :  dict[str, VolcanoPlotData] ) -> None :
    print("volcanoDataDict")
    for key, obj in vdd.items():
        details = asdict(obj)
        print(f"\nkey: {key}")
        # print(obj)
        for attr, value in details.items():
            print(f"  {attr} -> {value}")
            
########################################################################
def volcanoPlotsWithHighlights(
        deseqResultsFilePath : str,
        # imgOutDir : str,
        geneNamesPath : str = None,
        title : str = None,
        numHeaderLines : int = 8,
        pageWidthInInches : int = 4,
        pageHeightInInches : int = 4,
        label : bool = False,
        color = '#a2da37' # greenish aedwip
    ) -> plt.figure:
    ''' 
    Creates a volcano plot wiht optional highlighting
    
    arguments:
        deseqResultsFilePath : 
        
        numHeaderLines : the number of header lines in deseqResultsFilePath to skip
            example 
                mean of normalized counts for all samples 
                log2 fold change (MLE): extraIntra extra vs intra 
                standard error: extraIntra extra vs intra 
                Wald statistic: extraIntra extra vs intra 
                Wald test p-value: extraIntra extra vs intra 
                BH adjusted p-values 
                design: ~gender + extraIntra 
                "name","baseMean","log2FoldChange","lfcSE","stat","pvalue","padj"
                "ENSG00000081760.17",647.484654238966,-5.58264001613488,0.144706635899224,-38.5790187260156,0,0
                
                numHeaderLines = 8
                
        highlight arguments
            geneNamesPath :
                path to a file with a single column. ie list of names
                if the 'name' column in the deseq results file matches one of these name the point will be colored
                
            label :
                if geneNamesPath and label==True
                    the geneName will be displayed next to the point
                    
        color 
            9/21/26 hack for manuscript
                is a string version of a hexadecimal color code
                example '#a2da37' # greenish 
                    
        
    '''
       
    fig = createVolcanoPlot(
        title=title,
        inputFilePath=deseqResultsFilePath,
        numHeaderLines=numHeaderLines,
        geneNameFilePath=geneNamesPath,
        label=label,
        pageWidthInInches=pageWidthInInches,
        pageHeightInInches=pageHeightInInches ,
        color=color,
        )
    
    return fig

########################################################################
def createVolcanoPanel(
        vdd : dict[str, VolcanoPlotData],
        numHeaderLines : int=8,
        label : bool=True, 
        color = '#a2da37' # greenish
    ) -> tuple[plt.figure, list[plt.figure]]:
    ''' 
    TODO 
    
    arguments 
        if VolcanoPlotData as 
        
        color 
            9/21/26 hack for manuscript
            is a string version of a hexadecimal color code
            example '#a2da37' # greenish 
    
    returns :
        (panelFig, figList)
        figList is a list of the subplot figures
    '''
    figList = list()
    panelTitles = list()
    for diagnosis, volcanoPlotData in vdd.items():

        title = f'{volcanoPlotData.diagnosis}'
        panelTitles.append(title)
        fig = volcanoPlotsWithHighlights(
                        deseqResultsFilePath=volcanoPlotData.resultsPath,
                        #imgOutDir=tmpDir,
                        geneNamesPath=volcanoPlotData.geneNamePath,
                        title=title,
                        numHeaderLines=numHeaderLines,
                        # pageWidthInInches : int = 4,
                        # pageHeightInInches : int = 4,
                        # label=True,
                        label=label,
                        color=color
                )

        # prevent figure from being automatically displayed
        # notebook behaves differently than if run from cli
        plt.close(fig)

        figList.append(fig)
        
    panel = make_panel(figList, 
                       ncols=3, 
                       mode="uniform", 
                       #cm_shared_scale=True, 
                       #hide_colorbars=True, 
                       titles=panelTitles,
                       displayAsPrecentage=True
                       )
    
    return (panel, figList)
########################################################################
        
########################################################################
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    main()
