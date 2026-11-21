#
# scatterPlot.py
# Andrew E. Davidson
# aedavids@ucsc.edu
# 1/24/24
#
# Copyright (c) 2020-2027, Regents of the University of California All rights reserved.
# https://polyformproject.org/licenses/noncommercial/1.0.0 
#
'''
public functions:
    addJitter()
    createScatterPlot()
    pcaScatterPlot()
    plotScatterOnAxis()
    tSNEScatterPlot()
    umapScatterPlot()
'''

import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

logger = logging.getLogger(__name__)

################################################################################
def addJitter(x, y, jitterFrac : float=0.02) :
    '''
    scale jitter relative to the axis range
    
    arguments :
    
        jitterFrac:
            default = 0.02 ie 2%
        
    
    '''

    # Scale jitter by data range
    xJitterStrength = (x.max() - x.min()) * jitterFrac
    yJitterStrength = (y.max() - y.min()) * jitterFrac

    # Apply jitter
    xJittered = x + np.random.uniform(-xJitterStrength, xJitterStrength, size=x.shape)
    yYittered = y + np.random.uniform(-yJitterStrength, yJitterStrength, size=y.shape)
    
    return (xJittered, yYittered)

################################################################################
def createScatterPlot(
        x : list[float],
        y : list[float],
        title : str,
        xLabel : str,
        yLabel : str, 
        categories : list[str],
        #listOfCategoryTy : list[str],
        colors : list[str],
        figSize : tuple = (4,4),
        axisPaddingPrecentage : float = 0.02,
        markerSize : float = 10, # The marker size in points**2 (typographic points are 1/72 in.)
        jitterFrac : float = None

    ) -> None:
    '''
    make it easy to create a larger panel. Instead of passing a figSize tuple
    you pass a axis object
    
    arguments:
        panel:
            axis
        
        markerSize : float = 10
            The marker size in points**2 (typographic points are 1/72 in.)
    '''

    fig = plt.figure(figsize=figSize)
    panel = fig.add_subplot(1,1,1) 
    
    plotScatterOnAxis(
                    panel,
                    x, y,
                    title, xLabel, yLabel, 
                    categories,
                    colors,
                    figSize,
                    axisPaddingPrecentage,
                    markerSize,
                    jitterFrac
    )

    return (fig, panel)

################################################################################
def pcaScatterPlot (
    data : np.array,
    ySeries : pd.Series, 
    panel : plt.axes,
    title : str,
    markerSize : float = 10,
    jitterFrac : float = None
    ) -> None:
    '''
    create a scatter plot.
    
    if principalComponents has more than 2 columns will run 
    
    will use PCA to reduce principalComponents to 2 dimension if neede
    
    arguments: 
        principalComponents :
            the principalComponents to plot
            
        ySeries
            length = number of samples
            value = string class name 
    '''
    
    if data.shape[1] > 2:
        logger.info(f'principalComponents.shape : {data.shape} running PCA')
        pca2d = PCA( n_components=2, random_state=42)
        pcNP = pca2d.fit_transform( data)
    
        x = pcNP[:,0]
        y = pcNP[:,1]
        
    else :
        x = data[:, 0]
        y = data[:, 1]
    
    # Get N evenly spaced colors from the colormap
    nCategories = len(ySeries.unique())
    colors = plt.cm.viridis(np.linspace(0, 1, nCategories))
    
    plotScatterOnAxis(
                panel=panel,
                x = x,
                y = y,
                title = title,
                xLabel = 'Principal Component 1',
                yLabel = 'Principal Component 2', 
                categories = ySeries,
                colors=colors,
                axisPaddingPrecentage = 0.02,
                markerSize=markerSize,
                jitterFrac=jitterFrac #0.1 #0.05
        )

################################################################################
def plotScatterOnAxis(
        panel : plt.axes,
        x : list[float],
        y : list[float],
        title : str,
        xLabel : str,
        yLabel : str, 
        categories : list[str],
        #listOfCategoryTy : list[str],
        colors : list[str],
        figSize : tuple = (4,4),
        axisPaddingPrecentage : float = 0.02,
        markerSize : float = 10, # The marker size in points**2 (typographic points are 1/72 in.)
        jitterFrac : float = None
    ) -> None:
    '''
    
    arguments:
        TODO
        
        markerSize : float = 10
            The marker size in points**2 (typographic points are 1/72 in.)
    '''

    # fig = plt.figure(figsize=figSize)
    # pannel = fig.add_subplot(1,1,1) 
    panel.set_xlabel(xLabel, fontsize = 10)
    panel.set_ylabel(yLabel, fontsize = 10)
    panel.set_title(title, fontsize = 10)

    xMin = np.min(x)
    xMax = np.max(x)

    yMin = np.min(y)
    yMax = np.max(y)

    # print(f'xmin: {xMin} xMax : {xMax}')
    # print(f'yMin: {yMin} yMax : {yMax}')
    
    if jitterFrac :
        x,y = addJitter(x, y, jitterFrac=jitterFrac)

    # move points off of edge of plot
    xBuffer = np.abs((xMax - xMin)) * axisPaddingPrecentage
    yBuffer = np.abs((yMax - yMin)) * axisPaddingPrecentage

    panel.set_xlim(xMin - xBuffer, xMax + xBuffer)
    panel.set_ylim(yMin - yBuffer, yMax + yBuffer)

    df = pd.DataFrame( {'x' : x, 
                        'y' : y,
                        'kind' : categories
                        })
    # print(f'df.shape : {df.shape}')
    
    kinds = sorted( set(categories) )
    for kind, color in zip(kinds,colors):
        indicesToKeep = df['kind'] == kind
        panel.scatter(
                    df.loc[indicesToKeep, 'x'],
                    df.loc[indicesToKeep, 'y'],
                    color = color,
                    #, s = 50
                    #s=10, #5, #1,
                    s=markerSize,
                    linewidth=0 ,
                    alpha=0.5
                )
    panel.legend(kinds)
    panel.grid()

################################################################################
def tSNEScatterPlot(
    data : np.array,
    ySeries : pd.Series, 
    panel : plt.axes,
    title : str,
    markerSize : float = 10,
    jitterFrac : float = None
    ) -> None:
    '''
    create a scatter plot.
    
    if data has more than 2 columns will run 
    
    will use t-sne to reduce data to 2 dimension if neede
    
    arguments: 
        data :
            the data to plot
            
        ySeries
            length = number of samples
            value = string class name 
    '''
    
    if data.shape[1] > 2:
        logger.info(f'data.shape : {data.shape} running UMAP')
        tEmbedding = TSNE( n_components=2, random_state=42)
        tsneNP = tEmbedding.fit_transform( data)
    
        x = tsneNP[:,0]
        y = tsneNP[:,1]
        
    else :
        x = data[:, 0]
        y = data[:, 1]
    
    # Get N evenly spaced colors from the colormap
    nCategories = len(ySeries.unique())
    colors = plt.cm.viridis(np.linspace(0, 1, nCategories))
    
    plotScatterOnAxis(
                panel=panel,
                x = x,
                y = y,
                title = title,
                xLabel = 't-SNE 1',
                yLabel = 't-SNE 2', 
                categories = ySeries,
                colors=colors,
                axisPaddingPrecentage = 0.02,
                markerSize=markerSize,
                jitterFrac=jitterFrac #0.1 #0.05
        )
    
################################################################################
def umapScatterPlot(
    data : np.array,
    ySeries : pd.Series, 
    panel : plt.axes,
    title : str,
    markerSize : float = 10,
    jitterFrac : float = None
    ) -> None:
    '''
    create a scatter plot.
    
    if data has more than 2 columns will run 
    
    will use umap to reduce data to 2 dimension if neede
    
    arguments: 
        data :
            the data to plot
            
        ySeries
            length = number of samples
            value = string class name 
    '''
    
    if data.shape[1] > 2:
        logger.info(f'data.shape : {data.shape} running UMAP')
        umapL = umap.UMAP( n_components=2, random_state=42)
        umapNP = umapL.fit_transform( data)
    
        x = umapNP[:,0]
        y = umapNP[:,1]
        
    else :
        x = data[:, 0]
        y = data[:, 1]
    
    # Get N evenly spaced colors from the colormap
    nCategories = len(ySeries.unique())
    colors = plt.cm.viridis(np.linspace(0, 1, nCategories))
    
    plotScatterOnAxis(
                panel=panel,
                x = x,
                y = y,
                title = title,
                xLabel = 'UMAP 1',
                yLabel = 'UMAP 2', 
                categories = ySeries,
                colors=colors,
                axisPaddingPrecentage = 0.02,
                markerSize=markerSize,
                jitterFrac=jitterFrac #0.1 #0.05
        )