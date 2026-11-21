'''
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-08-23
Copyright (c) 2020-2027, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

description:
    public functions:
        plotConfusionMatrix()
        plotConfusionMatrixPanel()
'''
import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import sklearn.metrics as skm
from sklearn.preprocessing import LabelEncoder

# local import
from archive.dataTypes import DiagnosticBundle
from plots.plotPanel import make_panel

################################################################################
def plotConfusionMatrix(
        # lb : LabelEncoder,
        classes : list[str],
        yTrue : np.array ,
        yPredicted : np.array,
        title : str,
        displayAsPrecentage : bool = False,
        ) -> tuple[np.array, skm._plot.confusion_matrix.ConfusionMatrixDisplay]:
    '''
    yTrue and yPredict are arrays of encoded labels
    
    classes: order list of classes.
        example: s is a pandas series with ordered categorical values
            classes = s.cat.categories
    
    returns tuple:
        cfm : 
            the confusion matrix
            
        ConfusionMatrixDisplay : 
            this is an object with attributes for the figure, ax, 
            see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay
            
            use the plot() function to display
    '''
    
    # print(f'{yTrue[0:5]}')
    # print(f'{yPredicted[0:5]}')
    cfm = confusion_matrix(yTrue, yPredicted, labels=classes)
    # print(f'\n &&&&&&&&&&&&&&&&&\n{classes}')
    # print(cfm)
    
    if displayAsPrecentage :
        # Normalize to percentages (row-wise, so each row sums to 100)
        cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis] * 100
  
    disp = ConfusionMatrixDisplay(confusion_matrix=cfm, display_labels=classes)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cfm)
    disp.plot()
    
    if displayAsPrecentage:
        # Add % signs to each text annotation
        for t in disp.ax_.texts:
            #t.set_text(t.get_text() + "%") default matplot lib number format displays 1e+2 instead of 100
            val = float(t.get_text())
            # t.set_text(f"{val:.1f}%")
            # print(f"aEDWIP  displayAsPrecentage {val:.0f} {val:.1f}")
            t.set_text(f"{val:.0f}%")
                    
    ax = disp.ax_
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.title(title)
    
    ret = (cfm, disp)
    return ret

################################################################################
def plotConfusionMatrixPanel(
                            dataDict : dict[str,DiagnosticBundle],
                             dataSet : str,
                             title: str = "") -> plt.Figure:
    """ 
    TODO
    
    arguments
        dataSets should a one of the items in ["simTrain", "simVal", "simTest", "real"]
    """
    validDataSets = ["simTrain", "simVal", "simTest", "real"]
    if not dataSet in validDataSets :
        logger.error(f"invalide dataSet : {dataSet} should be key with value {validDataSets}")
        return
    
    cms = []
    # plots = []
    for diganosis, diagnosticBundle in dataDict.items():
        mb = getattr(diagnosticBundle, dataSet)
        # rocAxes = mb.rocAx
        # plots.append(rocAxes)
        
        disp = mb.cmDisplay
        cms.append(disp)
        
    titles = None #[ 'aaaa', 'bbbb', 'Ccccccccccc', 'd', 'TTTTTTTTTTTTT']
    panel = make_panel(cms, ncols=3, mode="uniform", cm_shared_scale=True, hide_colorbars=True, 
                       titles=titles,
                       displayAsPrecentage=True
                       )
    # figTitle = "dataSet"
    panel.suptitle( title )
    plt.show()    
    
    return panel