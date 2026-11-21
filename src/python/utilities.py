"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-05-14
Copyright (c) 2020-202r, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 

some common easy of use utility functions:
    configureTestLogging()
    isOrderedCategoricalSeries()    
    isOrderedCategoricalLike()
    inIPython() 
    logMemoryUsage(prefix="")
    printTestEnvInfo()
    runShellCommand()
    smartDisplay()
    urlify()
"""

import importlib.util
from IPython.display import display
import logging
# from git import Optional
import pandas as pd
from pandas.api.types import is_categorical_dtype
import psutil
import os
import re
import subprocess
import sys
from typing import Optional


logger = logging.getLogger( __file__ )

################################################################################
def configureTestLogging():
    # Don't double-configure if something else already set up logging
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(funcName)s() line:%(lineno)s] [%(message)s]",
            stream=sys.stdout,  # <-- visible in terminal, VS Code Debug Console, and Testing Output
        )

################################################################################
def isOrderedCategoricalSeries(obj) -> bool:
    """True iff obj is a Series with an ordered categorical dtype."""
    if not isinstance(obj, pd.Series):
        return False
    # Works across pandas versions
    dtype = obj.dtype
    if isinstance(dtype, pd.CategoricalDtype) or is_categorical_dtype(dtype):
        # For Series, .dtype.ordered is reliable
        return bool(getattr(dtype, "ordered", False))
    return False

################################################################################
def isOrderedCategoricalLike(obj) -> bool:
    """
    True for Series *or* Categorical objects with ordered categories.
    """
    if isinstance(obj, pd.Series):
        return isOrderedCategoricalSeries(obj)
    if isinstance(obj, pd.Categorical):
        return bool(obj.ordered)
    return False

################################################################################
def inIPython() -> bool:
    ''' 
    returns true if running in IPython, ie a juypter notebook
    '''
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False

################################################################################
def logMemoryUsage(prefix=""):
    rss = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    logger.warning(f"{prefix}Memory usage: {rss:.2f} MB")
    
################################################################################
def printTestEnvInfo():
    ORIG_PYTHONPATH = os.environ['PYTHONPATH']
    for d in ORIG_PYTHONPATH.split(':'):
        print(f'PYTHONPATH : {d}')
    
    print(f'FILE: {__file__}')
    print(f'PWD: {os.getcwd()}')
  
################################################################################
def runShellCommand(cmd: str) -> str:
    '''
    Run a shell command and return its output as a string.
    '''
    logger.info("BEGIN")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    logger.info("END")
    return result.stdout.strip()

################################################################################
def smartDisplay(
    obj,
    *,
    maxRows: Optional[int] = None,
    maxColWidth: Optional[int] = None,
    header: Optional[str] = None
):
    """
    Display rich objects in IPython; fall back to print/log elsewhere.
    Safe for notebooks, scripts, and SLURM jobs.
    """
    if header:
        print(f"\n{'='*10} {header} {'='*10}")

    if inIPython():
        from IPython.display import display
        import pandas as pd

        if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            with pd.option_context(
                "display.max_rows", maxRows,
                "display.max_colwidth", maxColWidth
            ):
                display(obj)
        else:
            display(obj)
    else:
        # Fallback: text-safe
        print(obj)

###############################################################################
def urlify(s):
    '''
    useful function for converting plot titles to strings
    https://stackoverflow.com/a/1007615/4586180

    print(urlify("I can't get no satisfaction!"))
    Prints: I-cant-get-no-satisfaction"
    '''

    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)

    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", '-', s)
    
    return s
