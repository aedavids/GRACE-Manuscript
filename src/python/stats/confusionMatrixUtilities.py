"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-08-27
Copyright (c) 2020-2027, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 
"""

# enable forward references
from __future__ import annotations

# from sklearn.preprocessing import LabelEncoder
import numpy as np
import numpy.typing as npt

################################################################################
def statistics(
    confusionMatrix : npt.NDArray[np.int_],
    # labelEncoder : LabelEncoder
    classes : list[str],

    ) -> dict[str, dict[str, float | int]] :
    '''
    
    arguments:
        confusionMatrix :
            a numpy array returned by sklearn.metrics.confusion_matrix
            
        # deprecated
        # LabelEncoder :
        #     contains the class names
            
            
        classes: order list of classes.
            example: s is a pandas series with ordered categorical values
                    classes = s.cat.categories
    returns
        a dictionary
            key = class 
            items = a dictionary with keys
                'tp', 'fp', 'fn', 'tn', 
                'sensitivity', 'specificity'

            
    '''
    # json can not serialzed np int and float
    # retMetricsDict = {}
    # for i, cls in enumerate(classes):
    #     tp = confusionMatrix[i, i]
    #     fn = confusionMatrix[i, :].sum() - tp
    #     fp = confusionMatrix[:, i].sum() - tp
    #     tn = confusionMatrix.sum() - (tp + fp + fn)

    #     sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0
    #     specificity = tn / (tn + fp) if (tn+fp) > 0 else 0

    #     retMetricsDict[cls] = dict(tp=tp, fp=fp, fn=fn, tn=tn,
    #                         sensitivity=sensitivity,
    #                         specificity=specificity) 
    
    retMetricsDict: dict[str, dict[str, float | int]] = {}

    total: int = int(confusionMatrix.sum())

    for i, cls in enumerate(classes):
        tp: int = int(confusionMatrix[i, i])
        fn: int = int(confusionMatrix[i, :].sum() - tp)
        fp: int = int(confusionMatrix[:, i].sum() - tp)
        tn: int = int(total - (tp + fp + fn))

        sens: float = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec: float = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        retMetricsDict[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "sensitivity": sens,
            "specificity": spec,
        }
        
    return retMetricsDict
