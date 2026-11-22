"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-12-31
Copyright (c) 2020-2027, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

public functions:
    getDefaultTermLibrary() -> Dict[str, List[str]]
        returns a dictionary of nlp friendly cancer regular expression
        
        initial set of terms suggested by chatGPT
"""

import logging
import os
import sys
from typing import Dict, List #  Iterable,  Pattern, Tuple, Optional

################################################################################
def getDefaultTermLibrary() -> Dict[str, List[str]]:
    """
    Returns category -> list of regex patterns (strings).
    Patterns are case-insensitive by default (we compile with re.IGNORECASE).
    Keep patterns small and specific; use stems where helpful (e.g., metastas\w*).
    
    keys: 
        "cancerType", "cellFate", "cellCycle", "genomeIntegrity"
        "genomeIntegrity", "invasionMetastasis", "angiogenesisHypoxia"
        "immuneInflammation",  "clinical", "nonCodingRepeat"
        
    """
    return {
        # High-specificity disease words
        "cancerType": [
            r"\bcarcinoma\b",
            r"\badenocarcinoma\b",
            r"\bsarcoma\b",
            r"\blymphoma\b",
            r"\bleukemia\b",
            r"\bmelanoma\b",
            r"\bneoplasm(s)?\b",
            r"\bmalignan(cy|t)\b",
            r"\btumou?r(s)?\b",
            r"\boncogene(s)?\b",
            r"\boncogenic\b",
            r"\bmetastas(is|es|ize|ized|izing|atic)\b",
        ],

        # Mechanisms / processes
        "cellFate": [
            r"\bapoptos(is|e|es|ed|ing)\b",
            r"\bsenescen(ce|t)\b",
            r"\bautophag(y|ic)\b",
        ],
        "cellCycle": [
            r"\bcell cycle\b",
            r"\bmitosis\b",
            r"\bproliferat(e|ion|ive|ing)\b",
            r"\bgrowth\b",
        ],
        "genomeIntegrity": [
            r"\bdna damage\b",
            r"\bdna repair\b",
            r"\bgenomic instabilit(y|ies)\b",
            r"\bdouble[-\s]?strand break(s)?\b",
            r"\bhomologous recombination\b",
            r"\bnon[-\s]?homologous end joining\b",
            r"\bchromatin remodel(ing|er|ers)\b",
            r"\bepigen(et|omic)\w*\b",
        ],

        # Invasion / microenvironment / immune
        "invasionMetastasis": [
            r"\binvasion\b",
            r"\bmigration\b",
            r"\bepithelial[-\s]?mesenchymal transition\b",
            r"\bemt\b",
            r"\bextracellular matrix\b",
        ],
        "angiogenesisHypoxia": [
            r"\bangiogen(es|ic|esis)\b",
            r"\bhypox(ia|ic)\b",
            r"\bvegf\b",
        ],
        "immuneInflammation": [
            r"\bimmune\b",
            r"\binflamm(at|ation|atory)\w*\b",
            r"\bcytokine(s)?\b",
            r"\bchemokine(s)?\b",
            r"\bcheckpoint\b",
            r"\bpd-?1\b",
            r"\bpd-?l1\b",
            r"\bctla-?4\b",
            r"\bt[-\s]?cell(s)?\b",
            r"\bb[-\s]?cell(s)?\b",
            r"\bmacrophage(s)?\b",
        ],

        # Clinical language
        "clinical": [
            r"\bbiomarker(s)?\b",
            r"\bdiagnos(tic|is)\b",
            r"\bprognos(tic|is)\b",
            r"\bpredict(ive|or|s)\b",
            r"\bsurvival\b",
            r"\btherapy resistance\b",
            r"\bdrug resistance\b",
        ],

        # Non-coding / repeats (useful in exRNA / EV contexts)
        "nonCodingRepeat": [
            r"\blncRNA\b",
            r"\blong non[-\s]?coding rna\b",
            r"\bmiRNA(s)?\b",
            r"\bmicroRNA(s)?\b",
            r"\bantisense\b",
            r"\bpseudogene(s)?\b",
            r"\btransposable element(s)?\b",
            r"\bendogenous retrovirus(es)?\b",
            r"\bherv\b",
            r"\brepeat element(s)?\b",
        ],
    }

################################################################################
def main():

    # we only configure logging in main module
    # loglevel = "WARN"
    loglevel = "INFO"
    # logFMT = "%(asctime)s %(levelname)s [thr:%(threadName)s %(name)s %(funcName)s() line:%(lineno)s] [%(message)s]"
    logFMT = "%(asctime)s %(levelname)s %(name)s %(funcName)s() line:%(lineno)s] [%(message)s]"
    logging.basicConfig(format=logFMT, level=loglevel)
    logger = logging.getLogger(os.path.basename(__file__))
    logger.warning("BEGIN")

    #
    # always log run time env to make debug easier
    #
    ORIG_PYTHONPATH = os.environ['PYTHONPATH']
    logger.warning(f'PYTHONPATH : {ORIG_PYTHONPATH}')
    logger.warning(f'FILE: {__file__}')
    logger.warning(f'PWD: {os.getcwd()}')

    # if inCommandLineArgsList is None:
    #     cli.parse()
    # else:
    #     cli.parse( inCommandLineArgsList )

    # logger.warning(f'command line arguments : {cli.args}')
    # features = cli.args.features
    # 	outDir   = cli.args.outDir

    #	exitCodeOkay = 0
    #	# exitCodeError = 1
    #	logger.warning(f"END exitCode : {exitCodeError}")
    #	sys.exit( okay )

################################################################################
if __name__ == "__main__":
    main()
#hack used to test parsing of vargs
# mainCLITest(
#     #inCommandLineArgsList=['-h']
#     inCommandLineArgsList=[
#         '--features', 'LUAD',
#         '--outDir', 'tmp'
#     ]
# )
