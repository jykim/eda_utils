from __future__ import print_function

import uuid
import json
import nbformat
import io
import glob
import pandas as pd
import time

from datetime import datetime, date, timedelta
# from google.cloud import storage
from IPython.core.display import HTML
from os.path import basename
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError
from nbconvert import HTMLExporter

SEP_PAIR, SEP_KV = "|", "-"

NOTEBOOK_STYLE = """
    <style>
    div.prompt, div.input {display: none;}
    div.lev1 {padding: 0px 0px 0px 15px;}
    div.lev2 {padding: 0px 0px 0px 30px;}
    div.lev3 {padding: 0px 0px 0px 45px;}
    div.bk-root {height: 200px; !important;}
    </style>"""

# EVAL_REPORT_PROJECT = 'search-analytics'
# EVAL_REPORT_BUCKET = 'ranking-analysis'
# EVAL_REPORT_PREFIX = 'TMP/AB_NOTEBOOKS'
# EVAL_REPORT_TYPE = 'text/html'

def get_today_str(format='%Y-%m-%d'):
    return datetime.today().strftime(format)


def date_offset(datestring, offset):
    ''' Take a date string (EX: '20170401') and get a datestring N number of days
        before or after.
    '''
    t = time.strptime(datestring, '%Y%m%d')
    offset_date = date(t.tm_year, t.tm_mon, t.tm_mday) + timedelta(offset)
    return offset_date.strftime('%Y%m%d')


def dict2str(arg_dict):
    """Serialize Dict to key-value string"""
    if not arg_dict:
        return ""
    return SEP_PAIR.join([SEP_KV.join([key, str(val)]) for key, val in arg_dict.items()])


def str2dict(arg_str):
    """De-serialize key-value string to dict"""
    res = {}
    for kv in arg_str.split(SEP_PAIR):
        k,v = kv.split(SEP_KV)
        res[k] = v
    return res


def get_first_codecell(nb):
    """Find the index of first code cell"""
    for i, e in enumerate(nb['cells']):
        if e['cell_type'] == 'code':
            return i


def run_export_notebook(nb_file, params=None, skip_run=False, allow_errors=True, timeout=3600,
                        out_path='.', out_format='html', out_filename=None, html_template='full'):
    """Run notebook with given parameter & export to notebook (& html) format
    Args:
        nb_file: notebook file to export
        params: notebook execution parameter
        skip_run: export without running

    Returns filename, notebook & html source 
    """
    with io.open(nb_file, encoding='utf8') as f:
        nb = nbformat.read(f, as_version=4)
    if not out_filename:
        if params:
            out_filename = basename(nb_file).replace(".ipynb", "") + "." + dict2str(params)
        else:
            out_filename = basename(nb_file).replace(".ipynb", "")            
    if not skip_run:
        if params:
            nb['cells'][get_first_codecell(nb)]['source'] = \
                "%s\nimport json\nPAR.update(json.loads('%s'))" % \
                (nb['cells'][get_first_codecell(nb)]['source'], json.dumps(params))
        try:
            ep = ExecutePreprocessor(timeout=timeout, allow_errors=allow_errors, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': '.'}})
        except CellExecutionError as e:
            print('Error executing the notebook "%s".\n\n%s' % (nb_file, str(e)))
            return

    if out_format == 'html':
        html_exporter = HTMLExporter()
        html_exporter.template_file = html_template
        (body, resources) = html_exporter.from_notebook_node(nb)
        html_source = NOTEBOOK_STYLE + body + NOTEBOOK_STYLE
        with io.open("%s/%s.html" % (out_path, out_filename), 'wt', encoding='utf8') as f:
            f.write(html_source)
    elif out_format == 'notebook':
        with io.open("%s/%s.ipynb" % (out_path, out_filename), 'wt', encoding='utf8') as f:
            nbformat.write(nb, f)

    return out_filename, nb, html_source


def get_params_from_filename(filename):
    """
    Input: 
        ab_experiment_sizing.EFFECT_SIZE-0.01|START_DATE-20170701|END_DATE-20170705.csv
    Output:
        EFFECT_SIZE-0.01|START_DATE-20170701|END_DATE-20170705
    """
    # return ".".join(filename.split(".")[1:-1])
    return ".".join(filename.split(".")[:-1])


def import_csv_files(fileptn, parse_params = True, verbose=True, **kwargs):
    """Import multiple csv output files into DataFrame"""
    filenames = glob.glob(fileptn)
    filenames.sort()
    tbls = []
    for e in filenames:
        if verbose:
            print(e)
        tbl = pd.read_csv(e, **kwargs)
        if parse_params:
            for k,v in str2dict(get_params_from_filename(basename(e))).items():
                tbl[k] = v
        else:
            tbl['filename'] = basename(e)
        tbls.append(tbl)
    res = pd.concat(tbls)
    res.index = range(len(res)) # Make sure to re-initialize index
    return res



