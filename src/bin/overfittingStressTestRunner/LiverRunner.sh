#!/bin/bash
# Andrew Davidson
# aedavids@ucsc.edu
# 02/17/2026
# 

scriptName=$0
printf "\n################# BEGIN ${scriptName} \n"

# see bash man page "SHELL BUILTIN COMMANDS" for details
# ref: https://gist.github.com/vncsna/64825d5609c146e80de8b1fd623011ca 
set -euxo pipefail
# set -e Exit immediately if a command in pipeline returns non-zero exit status
# set -u Treat unset variables and parameters as  an  error
# set -x turn debug trace on. output goes to stderr, normal output goes to stdout
# set -o print value of current options

condaEnv=fagazi
if [[ "${CONDA_DEFAULT_ENV:-}" == $condaEnv ]]; then
  echo "conda env $condaEnv is active"
else
  echo "ERROR conda env '$condaEnv' is NOT active"
  exit 1
fi

# find the root path to the local git repo
gitRoot=$(git rev-parse --show-toplevel)
export PYTHONPATH="${PYTHONPATH}:${gitRoot}/src/python:${gitRoot}/src/jupyterNotebooks/flagship/src"

# number of differential expresed genes
# Liver lfc : 2.0  padj : 0.001 : num genes : 7515
# Liver lfc : 2.0  padj : 0.01  : num genes : 8489
# Liver lfc : 2.0  padj : 0.05  : num genes : 8958



# use a base grid that include cap
# 5 10 25 50 100 200 400 800 1200 2000 4000 8000 8958


# 2/23/26
# changed code to calculate stats for realFix, simTrain, and realTest data sets
# only run up to 400. simulated folds where saved to disk
# it will take several days to complete featureCounts > 400

# --featureCounts 5 10 25 50 100 200 400 800 1200 2000 4000 8000 8958
args=(
  -m overfittingStressTestRunner
  --cancerType Liver
  --featureCounts 5 10 25 50 100 200 400
  --padjThreshold 0.05
  --lfcThreshold 2.0
  --numRepeats 50 
  --specifityTarget 0.9
)

#
# when we run from bash sometime we get tons of warnings
#  warnings.warn(
# /private/home/aedavids/miniconda3/envs/fagazi/lib/python3.13/site-packages/sklearn/utils/parallel.py:144: 
#     UserWarning: `sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` 
#     to make it possible to propagate the scikit-learn configuration of the current thread to the joblib workers.
# 
# this come from configuring random Forest to use all avalable cores
# the sys admin has some sort of max number of threads limit so we
# do not have to worry about how to config this
#
# randomforestClassifiler with n_jobs=-1
# all of the warning make our log files unusable
#
export PYTHONWARNINGS="ignore::UserWarning:sklearn.utils.parallel"

 python "${args[@]}"