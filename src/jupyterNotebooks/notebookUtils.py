"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-05-14
Copyright (c) 2020-2023, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 

Notebook easy of use functions:
    getNotebookName()

"""

import logging
import os
import sys

# local imports
from utilities import runShellCommand
 
logger = logging.getLogger(__name__)

################################################################################
def appendToPYTHONPATH(paths : list[str]) -> None:
    ''' 
    add paths to PYTHONPATH and sys.path
    '''
    # TODO refactor into utility function
    for p in paths:
        # to be able to import our local python files we need to set the sys.path
        # https://stackoverflow.com/a/50155834
        if p not in sys.path:
            sys.path.append(p)
            
        pPath = os.environ["PYTHONPATH"]
        os.environ["PYTHONPATH"] = f"{pPath}:{p}"
    

    logger.warning( f"Updated PYTHONPATH: {os.environ["PYTHONPATH"]}" )
        
    
################################################################################
def getNotebookName( defaultNotebookName : str) -> tuple[str,str]:
    '''
    Many notebooks create images or create other files. We want to save these files
    in a directory with the name of the notebook, close to the location of the notebook file.
    This makes it easy to know how the new images and files where created

    This function is an easy of use wrapper. When running on a jupyter notebook it uses ipynbname module
    and ignores the defaultNotebookName argument. When running in vscode it uses os.getcwd() and 
    the defaultNotebookName

    Typical usage:
        ```
        notebookName, notebookDir = getNotebookName(defaultName="myNotebook")

        outDir = f'{notebookDir}/{notebookName}.out'
        os.makedirs(outDir, exist_ok=True)
        print(f'outDir:\n{outDir}')

        modelOutDir = os.path.join(outDir, "model")
        os.makedirs(modelOutDir, exist_ok=True)
        print(f'\nmodelOutDir ;\n{modelOutDir}')

        imgOut = f'{outDir}/img'
        os.makedirs(imgOut, exist_ok=True)
        print(f'\nimgOut :\n{imgOut}')
        ```


    notebookName, notebookDir = getNotebookName(defaultName="myNotebook")

    arugments:
        defaultNotebookName : string

    return 
        tuple(notebookName, notebookDir)

        example:
            (vscodeTestNotebook, '/Users/andrewdavidson/googleUCSC/kimLab/fagazi/src/jupyterNotebooks')
    '''
    logger.info("BEGIN")
    try:
        import ipynbname

        notebookName = ipynbname.name()
        notebookPath = ipynbname.path()
        notebookDir = os.path.dirname(notebookPath)
    except Exception as e:
        logger.info(f"ipynbname only works if you run on a Jupyter server")
        logger.info(f'e : {e}')
        
        # when a notebook runs in vscode os.getcwd() returns '/'
        # notebookPath = os.environ.get('PWD', os.getcwd())
        #
        # you can not use notebook magic like ! from a python module
        # run shell command to find pwd
        # pwdList = ! pwd

        pwd = runShellCommand("pwd")
        logger.info(f'pwd : {pwd}')
        logger.info(f"os.getcwd()) {os.getcwd()}")        
        logger.info(f"os.environ.get('PWD', os.getcwd()) {os.environ.get('PWD', os.getcwd())}")
        notebookDir = pwd
        logger.info(f'notebookDir: {notebookDir}')
        logger.info(f"os.environ.get('VSCODE_IPYNB_FILENAME') : {os.environ.get('VSCODE_IPYNB_FILENAME')}")
        
        notebookName = defaultNotebookName

    logger.info("END")
    return (notebookName, notebookDir)
