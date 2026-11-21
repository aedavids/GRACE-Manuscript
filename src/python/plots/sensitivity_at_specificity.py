#!/usr/bin/env python3
"""

Copyright (c) 2020-2027, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

public functions:
    sensitivity_at_specificity(y_true, y_score, target_spec=0.90, interpolate=True)
    computeSensitivitiesAtTargets()
    plotROC()

Sensitivity at Fixed Specificity (TPR @ target TNR)
--------------------------------------------------
Usage:
    1) Demo with synthetic data (default):
        python sensitivity_at_specificity.py

    2) Use your own scores from a CSV with headers `y_true,y_score`:
        python sensitivity_at_specificity.py --csv my_scores.csv --targets 0.90 0.95 0.99

        - `y_true` must be 0/1 (negative/positive ground truth).
        - `y_score` should be the predicted probability/score for the positive class.

    Optional outputs:
        --out_csv results.csv    Save the sensitivity table.
        --save_plot roc.png      Save ROC plot image to a file.

Requirements:
    pip install numpy pandas scikit-learn matplotlib
"""
import logging
logger = logging.getLogger(__name__)

import argparse
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from plots.plotUtilities import stripMatplotlibSuffix

###############################################################################
def sensitivity_at_specificity(y_true, y_score, target_spec=0.90, interpolate=True):
    """
    Compute sensitivity (TPR) at a desired specificity (TNR).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels (0/1).
    y_score : array-like of shape (n_samples,)
        Predicted scores or probabilities for the positive class.
    target_spec : float, default=0.90
        Desired specificity (TNR). Must be in (0, 1).
    interpolate : bool, default=True
        If True, linearly interpolate TPR at the target FPR (=1 - target_spec).
        If False, choose the threshold that achieves specificity >= target_spec
        with the highest sensitivity; if none achieve it, choose the closest specificity.

    Returns
    -------
    sens : float
        Sensitivity (TPR) at/near the target specificity.
    achieved_spec : float
        The specificity (TNR) actually achieved (exact if interpolate=True, within ROC grid).
    threshold : float
        The (approximate) decision threshold associated with the point.
        When interpolate=True, this is the threshold nearest to the target FPR.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    spec = 1.0 - fpr
    target_fpr = 1.0 - float(target_spec)

    if interpolate:
        # Clip to the observed FPR range then interpolate TPR
        target_fpr = np.clip(target_fpr, fpr.min(), fpr.max())
        sens = float(np.interp(target_fpr, fpr, tpr))
        idx = int(np.argmin(np.abs(fpr - target_fpr)))
        threshold = float(thresholds[idx])
        achieved_spec = 1.0 - float(target_fpr)
        return sens, achieved_spec, threshold
    else:
        # spec is a NumPy array of specificity values for each ROC threshold
        # mask is a Boolean array where mask[i] is True if spec[i] >= target_spec, else False
        mask = spec >= target_spec
        if np.any(mask):
            idxs = np.where(mask)[0]
            best = idxs[np.argmax(tpr[idxs])]
            return float(tpr[best]), float(spec[best]), float(thresholds[best])
        else:
            closest = int(np.argmin(np.abs(spec - target_spec)))
            return float(tpr[closest]), float(spec[closest]), float(thresholds[closest])
    
###############################################################################
def computeSensitivitiesAtTargets(
        yTrue : list[float],
        yScore : list[float], 
        targets : list[float] = [0.90],
        interpolate : bool = False,
        ) -> tuple[ pd.DataFrame, dict[str, tuple[float,float]]] :
    '''
    arguments 
        targets: are the specificity the sensitivity is calculated at
            Desired specificity (TNR). Must be in (0, 1).

         yTrue : array-like of shape (n_samples,)
            Ground truth labels (0/1).
            
        yScore : array-like of shape (n_samples,)
            Predicted scores or probabilities for the positive class.
            
    returns:
        (dataframe, points)
        
        dataFrame with columns 
            "Target Specificity", "Sensitivity (TPR)",
            "Achieved Specificity (TNR)", "Approx. Threshold",
            "auc"
            
        points
            dictionary of points to plot 
                key = target specificity
                value = (FPR, TPR)
        
    '''
    rows = [] # data frame rows
    points = {} # points to plots
    for target in targets:
        fpr, tpr, thresholds = roc_curve(yTrue, yScore)
        roc_auc = auc(fpr, tpr)
        sens_i, spec_i, thr_i = sensitivity_at_specificity(yTrue, yScore, target_spec=target, interpolate=interpolate)
        rows.append({
            "Target Specificity": target,
            "Sensitivity (TPR)": sens_i,
            "Achieved Specificity (TNR)": spec_i,
            "Approx. Threshold": thr_i,
            "auc" : roc_auc
        })
        points[target] = (1.0 - spec_i, sens_i)  # (FPR, TPR)
        
    df = pd.DataFrame(rows)
    
    return (df, points)


###############################################################################
def plotROC(        
            points : dict[ str, tuple[float,float] ],   
            fpr : np.array,
            tpr : np.array,         
            figSize: tuple[float, float] = (6.0, 5.0),
            title : str = "ROC Curve with Sensitivity @ Fixed Specificities",
            # targets : list[float] = [0.90],
    ) -> Figure:
    ''' 
    creates an ROC plot with specificity at target specificity 
    marked on the curve
    
    arguments 
        points:
            returned by computeSensitivitiesAtTargets()
            
        fpr : false positive reate
        tpr : true positivate
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
            
            
        # targets are the specificity the sensitivity is calculated at
        # Desired specificity (TNR). Must be in (0, 1).

    '''
    fig = plt.figure(figsize=figSize)
    
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    
    for ts, (xf, yf) in points.items():
        plt.scatter([xf], [yf], label=f"TPR @ {int(ts*100)}% spec")
        
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()    
    
    return fig
    
###############################################################################
def load_csv_scores(path):
    df = pd.read_csv(path)
    if not {'y_true', 'y_score'}.issubset(df.columns):
        raise ValueError("CSV must contain headers: y_true,y_score")
    y_true = df['y_true'].to_numpy().astype(int)
    y_score = df['y_score'].to_numpy().astype(float)
    return y_true, y_score

###############################################################################
def run_demo(random_state=42):
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=8, n_redundant=2,
        n_clusters_per_class=2, flip_y=0.02, class_sep=1.2, random_state=random_state
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.35, stratify=y, random_state=random_state)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)
    
    # we want the probablity of the positive diagnostic test class
    y_score = clf.predict_proba(X_te)[:, 1]
    return y_te, y_score

###############################################################################
def main(inCommandLineArgsList=None):
    parser = argparse.ArgumentParser(description="Compute Sensitivity at fixed Specificities and plot ROC.")
    parser.add_argument("--csv", type=str, default=None, help="CSV with columns y_true,y_score")
    parser.add_argument("--targets", type=float, nargs="+", default=[0.90], help="Target specificities (e.g., 0.90 0.95)")
    parser.add_argument("--no_interpolate", action="store_true", help="Disable interpolation (pick closest threshold that meets spec)")
    parser.add_argument("--out_csv", type=str, default=None, help="Optional path to save results table as CSV")
    parser.add_argument("--save_plot", type=str, default=None, help="Optional path to save ROC plot image")

    if inCommandLineArgsList is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(inCommandLineArgsList)

    if args.csv:
        y_true, y_score = load_csv_scores(args.csv)
    else:
        print("[INFO] No CSV provided; running demo with synthetic data.")
        y_true, y_score = run_demo()
        print(f'y_true : {y_true}')
        print(f'y_score : {y_score}')

    # ROC & AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    df, points = computeSensitivitiesAtTargets(y_true, y_score, args.targets, interpolate=not args.no_interpolate)
    print("\n=== Sensitivity at Fixed Specificities ===")
    print(df.to_string(index=False))

    # Save results if requested
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"[INFO] Saved table to: {args.out_csv}")
        
    fig = plotROC(points, fpr, tpr  )


    if args.save_plot:
        logger.warning(f'AEDWIP 8/5/26 changed code to save png and pdf check file names and types')
        cleanFname = stripMatplotlibSuffix(args.save_plot)
        # plt.savefig(args.save_plot, dpi=300)
        plt.savefig(cleanFname, dpi=300, format='png')
        plt.savefig(cleanFname, dpi=300, format='pdf')
        print(f"[INFO] Saved plot to: {args.save_plot} .png .pdf")
    else:
        plt.show()

###############################################################################
if __name__ == "__main__":
    # main()

    tmpDir = "./tmp"
    os.makedirs(tmpDir, exist_ok=True)
    inCommandLineArgsList=[
        '--targets', '0.8', '0.85', '0.9', '0.95', '0.99',
        '--out_csv', f'{tmpDir}/sensitivitySpecificity.csv',
        '--save_plot', f'{tmpDir}/sensitivitySpecificity.png'
    ]   
    main( inCommandLineArgsList )