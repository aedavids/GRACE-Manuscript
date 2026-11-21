# Andrew E. Davidson  
# aedavids@ucsc.edu  
# 9/12/25  
#
# Copyright (c) 2020-2027, Regents of the University of California All rights reserved. https://polyformproject.org/licenses/noncommercial/1.0.0 

'''
calculatePurity()
hyperparameterTunning()
'''

from re import S

from anndata import AnnData
import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc

logger = logging.getLogger(__name__)


# import scanpy as sc
# from anndata import AnnData

# # (optional) nicer logging
# sc.settings.verbosity = 2

# scampyImgOutDir = f'{imgOutDir}/scanpy'
# os.makedirs(scampyImgOutDir, exist_ok=True)
# print(f'scampyImgOutDir: {scampyImgOutDir}')

# sc.settings.figdir = scampyImgOutDir  # where to save figures with save=...
# sc.set_figure_params(figsize=(5, 5), dpi=120)

################################################################################
def calculatePurity(adata : AnnData) -> float :
    '''
    For each cluster, take the majority class proportion, then average.
    
    purity = fraction of samples that are in the “dominant class” of their cluster.
    
    returns 
        range 0 - 1, higher is better
        
    assumes you have run leiden community detection mode
    '''
    xtab = pd.crosstab(adata.obs["label"], adata.obs["leiden"])
    byColumns = 0
    # sum adds all the values in the datafram
    purity = np.sum(xtab.max(axis=byColumns)) / np.sum(xtab.values)
    
    return purity

################################################################################
def hyperparameterTunning(
        data : np.array,
        labels : pd.Series,
    ) -> tuple[pd.DataFrame, dict[str,float]]:
    '''
    find best combination of resolution and number of neigbors across
    resolutions values in np.arange(0,1,0.1) and neigbors in range(3,11,1
    
    returns
        best confusion/crosstab   
        dictionary best hyperparmeter keys = purity, nNeighbors, resolution
    '''
    bestDict = {}
    bestDict['purity'] = 0
    bestDict['nNeighbors'] = 0
    bestDict['resolution'] = 0
    
    bestXTableDF = None
    # for tResolution in np.arange(0,5,0.2):
    for tResolution in np.arange(0,1,0.1):
        for tNeighbors in range(3,11,1):
            t = runScanpy(data, 
                labels,
                nNeighbors=tNeighbors, 
                resolution=tResolution,
                plotData=False
            )
            adata, xTabDF, xTabNormalizedDF, axesList = t
            
            purity = calculatePurity(adata)
            if purity > bestDict['purity']:
                logger.info(f'new best purity : {purity} nNeighbors : {tNeighbors} resolution : {tResolution:.02f} nCols :{ xTabDF.values.shape[1]}')
                # best = (purity, tNeighbors, tResolution)
                bestDict['purity'] = purity
                bestDict['nNeighbors'] = tNeighbors
                bestDict['resolution'] = tResolution
                bestXTableDF = xTabDF

    logger.info(f'\n best purity : {bestDict['purity']:.02f} nNeighbors : { bestDict['nNeighbors']} resolution : {bestDict['resolution']}')
    
    return( bestXTableDF,  bestDict)
    
################################################################################
def runScanpy(
        data : np.array,
        labels : pd.Series,
        nNeighbors : int, 
        resolution : float,
        plotData : bool = False,
        palette : str = None,
        displayOrSave : bool = True
    ) -> tuple[AnnData, pd.DataFrame, pd.DataFrame, list[plt.axes]] :
    ''' 
    TODO return umap figures, axes, ... If call from juypter notebook plots will be displayed
    
    hard coded leiden community detection. I had trouble install louvain
    
    you can control plotting as follows
        import scanpy as sc
        sc.settings.figdir
        sc.set_figure_params
    
    arguments:
        data:
            assume you have already used PCA to reduce the dimensions
            
        palette is optional, default None works. 
            example palette="viridis"
            
        displayOrSave
            if true will display plots
            else will save plots.
            
    return uple[AnnData, pd.DataFrame, pd.DataFrame, list[plt.axes]
        the list will have 4 axes
            1. Color by clusters
            2. Color by known class labels
            3. Side-by-side panels to compare (same embedding, different coloring)
            
    '''
    
    # python single cell packages use AnnData (annotated data) format
    # AnnData is like R/Bioconductor SummarizedExperiment
    # create an AnnData object from numpy arrar 
    adata = AnnData(X=data.copy())
    adata.obsm["X_pca"] = data  # lets Scanpy know you already have PCA coords
    adata.obs["label"] = pd.Categorical(labels)


    # create a graph
    # we have 6 classes. 224 samples. Some classes only ?30 examples.
    # what should n_neighbors be?
    # nNeighbors = 3
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=nNeighbors, n_pcs=None)

    # resolution controls the scale and granulatrity
    # higher resolution leads to more smaller clusters
    # default = 1.0
    # typically a value between 0.1 and 1

    # resolution = 0.5
    # results in adata.obs['louvain']
    # trouble installing louvain sc.tl.louvain(adata, resolution=resolution)  

    # does leiden community detection algorithym work better?
    # results in adata.obs['leiden']
    sc.tl.leiden(adata, resolution=resolution, key_added="leiden")

    # results in adata.obsm['X_umap']
    sc.tl.umap(adata)  

    axesA = None
    axesB = None
    axesCList = []
    retAxesList = []
    if plotData :
        # A) Color by Louvain clusters
        # sc.pl.umap(adata, color="louvain", frameon=False)
        # 8/11/26 added palette
        axesA  = sc.pl.umap(adata, color="leiden", frameon=False,  palette=palette, show=displayOrSave)
        outPathA = sc.settings.figdir / "leidenUMAP.pdf"
        # figA.savefig(outPathA, bbox_inches="tight",  dpi=300)

        # B) Color by known class labels
        axesB  = sc.pl.umap(adata, color="label", frameon=False,  palette=palette, show=displayOrSave) # legend_loc="on data", 
        outPathB = sc.settings.figdir / "leidenUMAPLabel.pdf"
        # figB.savefig(outPathB, bbox_inches="tight",  dpi=300)


        # C) Side-by-side panels to compare (same embedding, different coloring)
        # sc.pl.umap(adata, color=["louvain", "label"], ncols=2, frameon=False)
        axesCList  = sc.pl.umap(adata, color=["leiden", "label"], ncols=2, frameon=False,  palette=palette, show=displayOrSave)
        # outPathC = sc.settings.figdir / "2PlotleidenUMAPLabel.pdf"
        outPathCRoot = sc.settings.figdir / "2PlotleidenUMAPLabel"
        
        if not displayOrSave :
            axesA.figure.savefig(outPathA, bbox_inches="tight",  dpi=300)
            print(f'saved {outPathA}')
            
            axesB.figure.savefig(outPathB, bbox_inches="tight",  dpi=300)
            print(f'saved {outPathB}')

            axesC1 = axesCList[0]
            outPathC1 = str(outPathCRoot) + "First.pdf"
            axesC1.figure.savefig(outPathC1, bbox_inches="tight",  dpi=300)
            print(f'saved {outPathC1}')
            
            axesC2 = axesCList[1]
            outPathC2 = str(outPathCRoot) + "Second.pdf"
            axesC2.figure.savefig(outPathC2, bbox_inches="tight",  dpi=300)
            print(f'saved {outPathC2}')

    # ---------- how well do clusters match labels? (confusion/crosstab) ----------
    # xtab = pd.crosstab(adata.obs["label"], adata.obs["louvain"], normalize="index")  # row-normalized
    xtabNormalizedDF = pd.crosstab(adata.obs["label"], adata.obs["leiden"], normalize="index")  # row-normalized

    # print(f'\nnormalized crosstab/confusion table nNeighbors : {nNeighbors} resolution : {resolution}')
    # print(xtabNormalized.round(3))

    xtabDF = pd.crosstab(adata.obs["label"], adata.obs["leiden"])  

    # print(f'\n crosstab/confusion table')
    # print(xtabDF.round(3))
    
    retAxesList = None
    if axesA and axesB and axesCList[0] and axesCList[1] :
        retAxesList = [axesA, axesB,] + axesCList
        
    return (adata, xtabDF, xtabNormalizedDF, retAxesList )


# t = runScanpy(principalComponentsNP, 
#                 realYSeries,
#                 nNeighbors=3, 
#                 resolution=0.5,
#                 plotData=True
#             )

# adata, xTabDF, xTabNormalizedDF = t