"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-08-13
Copyright (c) 2020-2027, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

common functions used to create cli programs

configLogging()

"""

import logging
import os
import sys
import warnings

################################################################################
def configLogging( logLevel : str = "INFO") -> logging.Logger:
    '''
    This should only be called from main()
    
    1) It configure the logger output format 
    2) converts scipy RuntimeWarning message to be exception.
    The default behavior is to print a message to stderr
    
    logFMT = "%(asctime)s %(levelname)s %(name)s %(funcName)s() line:%(lineno)s] [%(message)s]"
    
    It also logs the PYTHONPATH, file name, and current working directory

    arguments:
    
                
        loglevel : "DEBUG", "INFO", "WARN", "ERROR"
            default value = "INFO"
        
    returns:
        root logger
    '''
    
    # Tell Python to treat RuntimeWarnings as exceptions
    warnings.filterwarnings("error", category=RuntimeWarning)

    # we only configure logging in main module
    
     # Map str → numeric (INFO → 20). Fallback to INFO on bad input.
    lvl = logging._nameToLevel.get(logLevel.upper(), logging.INFO)
 
    # logFMT = "%(asctime)s %(levelname)s [thr:%(threadName)s %(filename)s %(funcName)s() line:%(lineno)s] [%(message)s]"
    logFMT = "%(asctime)s %(levelname)s %(filename)s %(funcName)s() line:%(lineno)s] [%(message)s]"
    
    # Force reconfiguration in case something configured logging earlier
    logging.basicConfig(format=logFMT, level=lvl, force=True)
    # (Optional) Add a file handler as well:
    # fh = logging.FileHandler("app.log", encoding="utf-8")
    # fh.setFormatter(logging.Formatter(log_fmt))
    # root.addHandler(fh)
    
    rootLogger = logging.getLogger()
    rootLogger.warning("BEGIN")

    #
    # always log run time env to make debug easier
    #
    ORIG_PYTHONPATH = os.environ['PYTHONPATH']
    rootLogger.warning(f'PYTHONPATH : {ORIG_PYTHONPATH}')
    rootLogger.warning(f'FILE: {__file__}')
    rootLogger.warning(f'PWD: {os.getcwd()}')
    
    # test conversion of scipy runtime warnings to exceptions
    # try :
    #     warnings.warn("AEDWIP was this convert to an exception? This is a test warning!", category=RuntimeWarning)
    # except  Exception as e :
    #     rootLogger.error("runtime warning was caught")
    
    return rootLogger
