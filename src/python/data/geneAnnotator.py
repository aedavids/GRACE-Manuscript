"""
Andrew E. Davidson
aedavids@ucsc.edu
Date: 2025-10-21
Copyright (c) 2020-2023, Regents of the University of California All rights reserved.
https://polyformproject.org/licenses/noncommercial/1.0.0

Description: 
    cli data mining tool for info about a list of genes
    exmple of use
        $ python gene_annotator.py genes.txt -o annotations.csv
        
    ref: 
        https://docs.mygene.info/en/latest/index.html
        https://www.bioconductor.org/packages/release/bioc/manuals/mygene/man/mygene.pdf
        https://www.bioconductor.org/packages/release/bioc/manuals/mygene/man/mygene.pdf#page=7&zoom=100,133,149
        https://docs.mygene.info/en/latest/doc/annotation_service.html
        
utility function
def remapColumnNames(df : pd.DataFrame, 
                    annotationsDF : pd.DataFrame, 
                    keyColName : str='ensembl_id',
                    mapToColName : str='hgnc_symbol'
                    ) -> pd.DataFrame:
"""

"""
TODO:
can we mine info about genes as a complement to GSEA?

our challenge:
- we have features. deseq show they are biologically signifigant. 
does know biology verify our feature sets?
Can we use biotypes, pathyway, gene summary to confirm our models learned somethign useful
or possible unknow biologic relationship

for each gene 
    - get the matching HUGO name,
    - get biotype
    - summary info
    - get pathway info
    
problem 1: genes can participate in more than one pathway
 head data/test/data/test.pure.hugo.txt TP53 returns 49 pathways 
 
 ```
 [{'id': 'hsa04010', 'name': 'MAPK signaling pathway - Homo sapiens (human)'},
{'id': 'hsa04071', 'name': 'Sphingolipid signaling pathway - Homo sapiens (human)'},
{'id': 'hsa04110', 'name': 'Cell cycle - Homo sapiens (human)'},
```
- this does not work in a flat, single csv
    * flatten the list of dictionary and write to a separate csv?
    
    
problem 2: summary can be huge, 
    - ? write first line to "master csv"
    - create a separate csv file with fill summary?
    
problem 3: mygene info returns a single biotype for each gene.
TP53 has 12 at least 12 isoforms

INPP4A has lot of protien coding transcripts, some are retained intron
    
DDX11L1 :lncRNA, transcribed_unprocessed_pseudo
WASH7P : lncRNA, processed_pseudogene
    
    
problem 4: given a gene can we get a list of transcripts?
for each transcript can we get a biotype?, pathway, or protien

""" 


import logging
# __name__ is the current python module file name
logger = logging.getLogger(__name__)
import os
import sys

#!/usr/bin/env python3
# gene_annotator.py  (version-aware Ensembl handling)

import argparse, csv, sys, time, re
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import requests
import mygene

# Accept ENSG with optional .version
ENSEMBL_RE = re.compile(r"^ENSG\d{11}(?:\.\d+)?$")        # with optional version
ENSEMBL_STRIP_RE = re.compile(r"^(ENSG\d{11})(?:\.\d+)?$")  # capture unversioned core

###############################################################################
def remapColumnNames(df : pd.DataFrame, 
                    annotationsDF : pd.DataFrame, 
                    keyColName : str='ensembl_id',
                    mapToColName : str='hgnc_symbol'
                    ) -> pd.DataFrame:
    '''
    maps dataframe column names from one set of ids to another.
    for example: ensembl_id -> hgnc_symbols
        'ENSG00000004939.16' -> 'ENSG00000004939-SLC4A1'
    
    Arguments:
        df
        
        annotationsDF
            a dataframe with columns keyColName and mapToColName
            example a file create by python.data.geneAnnotator
        
        keyColname
            default : 'ensembl_id'
        
        mapToColName 
            default : 'hgnc_symbol'
            
    returns
        a deep copy of the original dataFrame argument with column names
        with format keyColname-mapToColName
    '''
    
    retDF = df.copy(deep=True).head(n=5)
    retColNames = list(retDF.columns)
    # print(f'retColNames:\n {retColNames} \n')
    
    for i in range(len(retColNames)):
        colName = retColNames[i]
        if colName.startswith( "ENSG" ) :
            # ensembl_id have a version number. drop the version number if it exists 'ENSG00000004939.16'
            baseName = colName.split(".")[0]
            #print(f'\ncolName : {colName} baseName : {baseName}')    
            try :        
                hngcNameSeries = annotationsDF[ annotationsDF[keyColName] == baseName ]
                hngcNameSeries = hngcNameSeries[mapToColName]
                
                hgncNameStr = hngcNameSeries.values[0]
                retColNames[i] = baseName + "-" + hgncNameStr
            except Exception as e:
                # either the baseName or hgnc_symbol was not defined
                # use the original col name
                pass
            
    # print(f'\n*****\nretColNames:\n{retColNames}')
    
    retDF.columns = retColNames
    # print(f'AEDWIP\n{retDF.columns}')
    return retDF
    
    
###############################################################################
def stripEnsemblVersion(x: str) -> Tuple[str, Optional[str]]:
    """
    Return (unversioned, version_str or None). If not Ensembl, returns (x, None).
    """
    m = ENSEMBL_STRIP_RE.match(x)
    if not m:
        return x, None
    core = m.group(1)
    version = x[len(core):].lstrip(".") if x != core else None
    return core, version

###############################################################################
def readGeneList(path: str) -> List[str]:
    # Try CSV/TSV with a 'gene' column; fallback to plain text
    for sep in (",", "\t", None):
        try:
            if sep is None:
                with open(path) as f:
                    return [ln.strip() for ln in f if ln.strip()]
            df = pd.read_csv(path, sep=sep)
            if "gene" in df.columns:
                return [str(x).strip() for x in df["gene"].dropna().tolist()]
        except Exception:
            continue
    return []

###############################################################################
def splitByNamespace(ids: List[str]) -> Tuple[List[str], List[str]]:
    ens, sym = [], []
    for g in ids:
        if ENSEMBL_RE.match(g):
            ens.append(g)
        else:
            sym.append(g)
    return ens, sym

###############################################################################
def chunked(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

###############################################################################
def queryMygene(ids_for_query: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    ids_for_query already have Ensembl versions stripped when needed.
    """
    mg = mygene.MyGeneInfo()
    out: Dict[str, Dict[str, Any]] = {}
    ens_ids, sym_ids = splitByNamespace(ids_for_query)

    # fields = "ensembl.gene,symbol,hgnc,type_of_gene,summary,name"
    fields = "ensembl.gene,symbol,hgnc,type_of_gene,name,pathway.kegg,summary" # 
    # Ensembl lookups
    if ens_ids:
        for chunk in chunked(ens_ids, 1000):
            res = mg.querymany(
                chunk, scopes="ensembl.gene", fields=fields,
                species="human", as_dataframe=False, returnall=False, verbose=False
            )
            for r in res:
                logger.debug(f"AEDWIP ENS query result keys: {list(r.keys())}")
                q = r.get("query")
                if not q:
                    continue
                
                pw = r.get('pathway')
                if pw:
                    logger.debug(f'AEDWIP : {pw}')
                
                out[q] = {
                    "mygene_symbol": r.get("symbol"),
                    "mygene_name": r.get("name"),
                    "mygene_type": r.get("type_of_gene"),
                    "mygene_summary": r.get("summary"),
                    "hgnc_id": r.get("hgnc"),
                    "ensembl_ids": (
                        [r["ensembl"]["gene"]] if isinstance(r.get("ensembl"), dict) and "gene" in r["ensembl"]
                        else list({e.get("gene") for e in r.get("ensembl", []) if isinstance(e, dict) and e.get("gene")})
                    ) or None,
                    'pathway' : r.get('pathway') or None
                }
    # Symbol lookups
    if sym_ids:
        for chunk in chunked(sym_ids, 1000):
            res = mg.querymany(
                chunk, scopes="symbol,hgnc", fields=fields,
                species="human", as_dataframe=False, returnall=False, verbose=False
            )
            for rDict in res:
                q = rDict["query"]
                if not q:
                    continue
                
                # normalize ensembl ids (dict or list)
                ens = rDict.get("ensembl")
                if isinstance(ens, dict):
                    ensembl_ids = [ens.get("gene")] if ens.get("gene") else []
                else:
                    ensembl_ids = sorted({e.get("gene") for e in (ens or []) if isinstance(e, dict) and e.get("gene")})

                kegg = []
                pw = rDict.get("pathway")
                if isinstance(pw, dict) and "kegg" in pw:
                    k = pw["kegg"]
                    if isinstance(k, dict):
                        kegg = [k]
                    elif isinstance(k, list):
                        kegg = [x for x in k if isinstance(x, dict)]
                        
                kegg_records = [{"id": x.get("id"), "name": x.get("name")} for x in kegg]
                logger.debug(f'AEDWIP kegg_records: {kegg_records}')
                kegg_names = sorted({x.get("name") for x in kegg if x.get("name")})
                logger.debug(f'AEDWIP kegg_names: {kegg_names}')


                        
                out[q] = {
                    # use get()instead of [], get() handles missing keys by return None
                    "mygene_symbol": rDict.get("symbol"),
                    "mygene_name": rDict.get("name"),
                    "mygene_type": rDict.get("type_of_gene"),
                    "mygene_summary": rDict.get("summary"),
                    "hgnc_id": rDict.get("hgnc"),
                    
                    "ensembl_ids": ensembl_ids or None,
                    
                    "kegg_pathways_full": kegg_records or None,
                    "kegg_names" : kegg_names or None,
                }
    return out

###############################################################################
def fetchEnsemblBiotype(ensembl_ids: List[str], pause: float = 0.0) -> Dict[str, Optional[str]]:
    biotypes: Dict[str, Optional[str]] = {}
    if not ensembl_ids:
        return biotypes
    s = requests.Session()
    headers = {"Accept": "application/json"}
    url_tmpl = "https://rest.ensembl.org/lookup/id/{}?content-type=application/json"
    for eid in ensembl_ids:
        try:
            r = s.get(url_tmpl.format(eid), headers=headers, timeout=15)
            biotypes[eid] = r.json().get("biotype") if r.status_code == 200 else None
        except Exception:
            biotypes[eid] = None
        if pause:
            time.sleep(pause)
    return biotypes

###############################################################################
def best_ensembl_id(ens_list: Optional[List[str]]) -> Optional[str]:
    return (ens_list or [None])[0]

###############################################################################
def main(inCommandLineArgsList=None):
    
    description = """
    Annotate human genes (Ensembl or HGNC) with Ensembl ID, HGNC symbol, biotype, and summary.
    
    example : 
        cat genes.txt
        ENSG00000088386.17
        ENSG00000140623.14
        ENSG00000162460.7
        ENSG00000181773.7
        ENSG00000186393.5
        
        python -m  data.geneAnnotator --sleep 1 --output ./geneAnnotationsLiver.csv genes.txt
    """
    logger.warning(f'AEDWIP !! see lines 216 and 260, I started adding pathway info. see comments at top of file')
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Input file: plain text (one ID per line) or CSV/TSV with a 'gene' column")
    ap.add_argument("-o", "--output", default="annotations.tsv", help="Output file (TSV or CSV by extension)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between Ensembl requests (politeness)")
    
    if inCommandLineArgsList is None:
        args = ap.parse_args()
    else :
        args = ap.parse_args(inCommandLineArgsList)

    inputs_raw = readGeneList(args.input)
    if not inputs_raw:
        print("No genes found in input.", file=sys.stderr); sys.exit(1)

    # Build query list with Ensembl versions stripped, and a mapping back to original
    query_ids: List[str] = []
    original_to_query: Dict[str, str] = {}
    ensembl_version_map: Dict[str, Optional[str]] = {}
    for g in inputs_raw:
        if ENSEMBL_RE.match(g):
            unv, ver = stripEnsemblVersion(g)
            query_ids.append(unv)
            original_to_query[g] = unv
            ensembl_version_map[g] = ver
        else:
            query_ids.append(g)
            original_to_query[g] = g
            ensembl_version_map[g] = None

    mg_map = queryMygene(query_ids)

    # Collect Ensembl IDs for biotype lookups (unversioned)
    ens_to_fetch = set()
    for q in query_ids:
        if ENSEMBL_RE.match(q):  # note: q is unversioned for Ensembl cases
            ens_to_fetch.add(q)
        ens_list = mg_map.get(q, {}).get("ensembl_ids") or []
        for e in ens_list:
            ens_to_fetch.add(e)

    ensembl_biotype_cache = fetchEnsemblBiotype(sorted(ens_to_fetch), pause=args.sleep)

    # Build rows aligned to original inputs
    rows = []
    for original in inputs_raw:
        q = original_to_query[original]
        ann = mg_map.get(q, {}) or {}
        ens_list = ann.get("ensembl_ids")
        ens = best_ensembl_id(ens_list) if ens_list else (q if ENSEMBL_RE.match(q) else None)
        symbol = ann.get("mygene_symbol") if ann.get("mygene_symbol") else (None if ENSEMBL_RE.match(q) else q)
        biotype = ensembl_biotype_cache.get(ens) if ens else None
        if not biotype:
            biotype = ann.get("mygene_type")

        rows.append({
            "input_id": original,
            "id_type": "ensembl" if ENSEMBL_RE.match(original) else "hgnc_symbol",
            "ensembl_id": ens,                          # unversioned ID used by APIs
            "ensembl_version": ensembl_version_map[original],  # version from input if any
            "hgnc_symbol": symbol,
            "biotype": biotype,
            "gene_name": ann.get("mygene_name"),
            #"kegg_pathways" : ann.get('kegg_pathways_full'), # full summary
            "kegg_pathways" : ann.get('kegg_names'), # names only
            "summary": ann.get("mygene_summary"),
            "sources": ";".join(s for s in [
                "MyGene.info" if ann else None,
                "Ensembl" if ens and ens in ensembl_biotype_cache else None
            ] if s)
        })

    df = pd.DataFrame(rows, columns=[
        "input_id","id_type","ensembl_id","ensembl_version","hgnc_symbol","biotype","gene_name","sources","kegg_pathways", "summary",
    ])

    out = args.output
    if out.lower().endswith(".csv"):
        df.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
    else:
        df.to_csv(out, sep="\t", index=False)
    print(f"Wrote {len(df)} rows to {out}")

###############################################################################
if __name__ == "__main__":
    main()
    
    # Uncomment the following lines to test parsing of vargs
    pwd = os.getcwd()
    # key = 'hugo'
    # key = 'ensembl'
    key = 'pure.hugo'
    testFilePath = f'{pwd}/src/python/data/test/data/test.{key}.txt'
    outDir =f'{pwd}/src/python/data/test/tmp'
    os.makedirs(outDir, exist_ok=True)
    outFilePath = f'{outDir}/geneAnnotator.{key}.debug.csv'
    # main(
    #     #inCommandLineArgsList=['-h']
    #     inCommandLineArgsList=[
    #         '--output', outFilePath,
    #         testFilePath
    #     ]
    # )
