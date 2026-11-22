"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2026-01-01
Copyright (c) 2020-2027, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Tuple

################################################################################
def addCancerRelevanceScore(
    annotation: Dict[str, object],
    *,
    categoryWeights: Dict[str, float] | None = None,
    topKContributors: int = 5,
    ) -> Dict[str, object]:
    """
    Takes existing annotation dict (from summarizeHits/annotateGeneSummary)
    and returns a NEW dict with:
      - cancerRelevanceScore
      - categoryHitCounts
      - weightedCategoryScores
      - topContributors

    Assumes annotation has key: "matchedTerms" (category -> list[str]).
    """
    weights = categoryWeights or getDefaultCategoryWeights()

    matchedTerms = annotation.get("matchedTerms", {})
    categoryHitCounts = getCategoryHitCounts(matchedTerms)  # type: ignore[arg-type]

    weightedCategoryScores = computeWeightedCategoryScores(categoryHitCounts, weights)
    score = computeCancerRelevanceScore(weightedCategoryScores)
    topContributors = getTopScoreContributors(weightedCategoryScores, topK=topKContributors)

    # Return a copy (keeps functions pure + testable)
    out = dict(annotation)
    out["cancerRelevanceScore"] = score
    out["categoryHitCounts"] = categoryHitCounts
    out["weightedCategoryScores"] = weightedCategoryScores
    out["topContributors"] = topContributors
    return out

################################################################################
def computeCancerRelevanceScore(weightedCategoryScores: Dict[str, float]) -> float:
    """
    Total score = sum of category contributions.
    """
    return float(sum(weightedCategoryScores.values()))

################################################################################
def computeWeightedCategoryScores(
    categoryHitCounts: Dict[str, int],
    categoryWeights: Dict[str, float],
    ) -> Dict[str, float]:
    """
    Category -> weightedScore = hitCount * weight.
    Categories missing from weights default to 0.0 (no branching needed).
    """
    return {
        category: hitCount * categoryWeights.get(category, 0.0)
        for category, hitCount in categoryHitCounts.items()
    }

################################################################################
def getCategoryHitCounts(matchedTerms: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Convert category -> list of matched strings into category -> hitCount.
    Uses unique matched strings (as produced by your summarizeHits()).
    """
    return {category: len(terms) for category, terms in matchedTerms.items()}

################################################################################
def getDefaultCategoryWeights() -> Dict[str, float]:
    """
    Category -> weight. Higher means more cancer-specific / higher evidence.
    Tune these weights as you learn what works for your summaries.
    """
    return {
        "cancerType": 5.0,
        "genomeIntegrity": 4.0,
        "cellCycle": 3.0,
        "cellFate": 3.0,
        "invasionMetastasis": 3.0,
        "angiogenesisHypoxia": 2.0,
        "immuneInflammation": 2.0,
        "clinical": 2.0,
        "nonCodingRepeat": 1.0,
    }

################################################################################
def getTopScoreContributors(
    weightedCategoryScores: Dict[str, float],
    *,
    topK: int = 5
    ) -> List[Tuple[str, float]]:
    """
    Return top contributing categories, highest first.
    """
    return sorted(weightedCategoryScores.items(), key=lambda kv: kv[1], reverse=True)[:topK]


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
