"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-12-31
Copyright (c) 2020-2027, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 
"""

# enables foward refernces
from __future__ import annotations


from dataclasses import dataclass

import logging
import os
import pprint
import re
import sys
from typing import Dict, Iterable, List, Pattern, Tuple, Optional

# local imports
from archive.archiveUtilities import printDictTree
from data.termLibrary import getDefaultTermLibrary

################################################################################
@dataclass(frozen=True) # make instances immutable
class MatchHit:
    category: str
    pattern: str
    matchText: str
    span: Tuple[int, int]


################################################################################
def annotateGeneSummary(
    summaryText: str,
    compiledLibrary: Dict[str, List[Pattern[str]]],
) -> Dict[str, object]:
    """
    One-stop function: find hits + summarize.
    Returns a dict you can store in JSON / a dataframe cell.
    """
    hits = findMatchesInText(summaryText, compiledLibrary) 
    return summarizeHits(hits)

################################################################################
def annotateManySummaries(
    summariesByGeneId: Dict[str, str],
    compiledLibrary: Dict[str, List[Pattern[str]]],
) -> Dict[str, Dict[str, object]]:
    """
    Batch annotate: geneId -> annotation summary dict.
    Keeps function pure and easy to test.
    """
    out: Dict[str, Dict[str, object]] = {}
    for geneId, summary in summariesByGeneId.items():
        out[geneId] = annotateGeneSummary(summary, compiledLibrary)
    return out

################################################################################
def compileTermLibrary(
    termLibrary: Dict[str, List[str]],
    *, # weird syntax force caller to use key words, ie name arguments
    flags: int = re.IGNORECASE
) -> Dict[str, List[Pattern[str]]]:
    """Compile string patterns into regex Pattern objects."""
    compiled: Dict[str, List[Pattern[str]]] = {}
    for category, patterns in termLibrary.items():
        compiled[category] = [re.compile(p, flags) for p in patterns]
    return compiled

################################################################################
def findMatchesInText(
    text: str,
    compiledLibrary: Dict[str, List[Pattern[str]]]
    ) -> List[MatchHit]:
    """
    Return all hits across categories.
    Flat loop style to avoid deep conditionals.
    """
    hits: List[MatchHit] = []
    cleanText = normalizeText(text)

    for category, regexList in compiledLibrary.items():
        for rx in regexList:
            for m in rx.finditer(cleanText):
                hits.append(
                    MatchHit(
                        category=category,
                        pattern=rx.pattern,
                        matchText=m.group(0),
                        span=(m.start(), m.end()),
                    )
                )
    return hits

################################################################################
def normalizeText(text: Optional[str]) -> str:
    """
    Lower-cost normalization: coerce None, collapse whitespace, strip.
    """
    if not text:
        return ""
    # Don’t lowercase here; regex compilation uses IGNORECASE.
    return re.sub(r"\s+", " ", text).strip()

################################################################################
def summarizeHits(hits: List[MatchHit]) -> Dict[str, object]:
    """
    Produce an easy-to-store summary object.
    - matchedCategories: unique categories
    - matchedTerms: category -> unique matched strings
    - hitCount: total hits
    """
    matchedCategories = sorted({h.category for h in hits})

    matchedTerms: Dict[str, List[str]] = {}
    for h in hits:
        matchedTerms.setdefault(h.category, [])
        if h.matchText not in matchedTerms[h.category]:
            matchedTerms[h.category].append(h.matchText)

    # sort term lists for stable output
    for k in matchedTerms:
        matchedTerms[k] = sorted(matchedTerms[k], key=lambda s: s.lower())

    return {
        "hitCount": len(hits),
        "matchedCategories": matchedCategories,
        "matchedTerms": matchedTerms,
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
    
    termLibrary = getDefaultTermLibrary()
    compiledLibrary = compileTermLibrary(termLibrary)

    exampleSummary = """
    This gene is frequently upregulated in lung adenocarcinoma and promotes
    cell cycle progression, proliferation, and metastasis. It may affect DNA repair
    and is associated with poor survival and therapy resistance.
    """

    annotation = annotateGeneSummary(exampleSummary, compiledLibrary)
    print(f'exampleSummary:\n{exampleSummary}')
    print(f'\nannotated text\n')
    # print(annotation)

    print()
    # printDictTree(annotation)
    print()
    ppAnnotationStr = pprint.pformat(annotation,  width=80, indent=2, sort_dicts=True, compact=False, )
    print(ppAnnotationStr)
    
#     { 'hitCount': 7,
#   'matchedCategories': [ 'cancerType',
#                          'cellCycle',
#                          'clinical',
#                          'genomeIntegrity'],
#   'matchedTerms': { 'cancerType': ['adenocarcinoma', 'metastasis'],
#                     'cellCycle': ['cell cycle', 'proliferation'],
#                     'clinical': ['survival', 'therapy resistance'],
#                     'genomeIntegrity': ['DNA repair']}}
    
    exitCodeOkay = 0
    #	# exitCodeError = 1
    logger.warning(f"END exitCode : {exitCodeOkay}")
    sys.exit( exitCodeOkay )

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
