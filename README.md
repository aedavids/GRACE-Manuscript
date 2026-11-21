# Generative RNA Cohort Expansion (GRACE) Manuscript

Andrew E. Davidson   
aedavids@ucsc.edu.  
Date: 2026-09-30.  

Copyright (c) 2025-2026, Regents of the University of California All rights reserved.   
https://polyformproject.org/licenses/noncommercial/1.0.0


a framework for expanding high-dimensional molecular training cohorts while preserving real biological samples for evaluation. Classifiers trained on simulated molecular distributions achieved comparable or improved diagnostic performance relative to conventional small-cohort approaches, remained robust as feature dimensionality increased, and identified biologically interpretable features associated with disease-specific pathways and diverse genomic feature classes.

**To create your env from scratch with name graceM run:**. 
```
conda env create -f environment.yml
```
**If you use vscode**. 
- check .vscode/setting.json
- it expects a .env file in your home directory. Here is an example
```
$ cat ~/graceM/.env
WORKSPACE_ROOT_DIR=/private/home/aedavids/graceM
PYTHONPATH=${WORKSPACE_ROOT_DIR}/src/python
PYTHONPATH=${PYTHONPATH}:${WORKSPACE_ROOT_DIR}/src/jupyterNotebooks
```
