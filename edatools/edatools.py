import re
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
sns.set(color_codes=True)
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.options.display.max_colwidth = 999
pd.options.display.max_seq_items = 999

from IPython.display import display, HTML, Image
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.models.glyphs import VBar
from bokeh.embed import components
from bokeh.charts import Histogram, Bar
from math import pi


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.izip_longest(*args, fillvalue=fillvalue)


def make_hyperlink(url, title = "Link"):
    return "<a href=\"{}\">{}</a>".format(url, title)
    

def load_bokeh(version):
    res = """
        <link
            href="http://cdn.pydata.org/bokeh/release/bokeh-{version}.min.css"
            rel="stylesheet" type="text/css">
        <script src="http://cdn.pydata.org/bokeh/release/bokeh-{version}.min.js"></script>
        """
    return res.format(version=version)


def sample_bq_table(query_raw, n, project_id, hash_key = "session_id", debug=False):
    # Get row count to determine sampling rate
    query_count = re.sub(r"^\s*SELECT\s+(.*?)\s+FROM", "SELECT count(1) FROM" , query_raw, re.M, re.DOTALL)
    # print query_count
    rowcount = pd.read_gbq(query_count, verbose=False, project_id=project_id).iloc[0, 0]
    sample_rate = int(rowcount / n)
    # Run sampling query
    if rowcount > n*2:
        query_sample = query_raw + " AND HASH({}) % {} == 0".format(hash_key, sample_rate)
    else:
        query_sample = query_raw
    # print query_sample 
    res = pd.read_gbq(query_sample, verbose=False, project_id=project_id)
    print("%d / %d = %d" % (rowcount, sample_rate, len(res)))
    return res


class RawTable:
    """This class performs EDA for (sampled) raw data
    """
    def __init__(self, tbl):
        self.tbl = tbl
        self.cols = tbl.columns

        self.dtypes, self.vcounts, self.ncounts = {}, {}, {}
        for c in tbl.columns:
            vcounts = self.tbl[c].value_counts()
            self.dtypes[c] = str(vcounts.index.dtype)
            self.vcounts[c] = len(vcounts)
            self.ncounts[c] = self.tbl[c].isnull().sum()

    def run_eda(self):
        """Run basic EDA with single function"""
        self.desc()
        self.corr()
        self.pairplot()

    def groupby(self, by, min_count=25):
        """Return RawTable object for each row group"""
        for gname, gtbl in self.tbl.groupby(by):
            if len(gtbl) < min_count:
                continue
            display(HTML("<h3>%s: %s</h3>" % (str(by), gname)))
            yield RawTable(gtbl)

    ### COLUMN-LEVEL ANALYSIS

    def head(self, n=10):
        return self.tbl.head(n)

    def colinfo(self, n=3):
        """Print colum metadata"""
        colnames = ["DataType", "ValueCount", "NullCount"]
        t_colinfo = pd.DataFrame([(self.dtypes[c], self.vcounts[c], self.ncounts[c]) for c in self.cols], columns = colnames, index=self.cols)
        t_head = self.tbl.head(n=3).transpose()
        return pd.concat([t_colinfo, t_head], axis=1)

    def desc(self, cols=None, outputcol=5, figsize=(250,250), max_bins=25, topk=10, proportiontocut=0):
        """Describe each column in table & plot"""
        if not cols:
            cols = [c for c in self.cols 
                if (self.dtypes[c] != "object" or self.vcounts[c] < topk) 
                and self.ncounts[c] < len(self.tbl)]
        rows = []
        for colgroup in grouper(outputcol, cols):
            colgroup = [c for c in colgroup if c]
            row = []
            for i, c in enumerate(colgroup):
                if self.dtypes[c] == "datetime64[ns]":
                    continue
                elif self.dtypes[c] == "object" and self.vcounts[c] > topk:
                    row.append(self.print_summary(c, 'vcounts', topk=topk))
                else: 
                    row.append(self.print_summary(c, 'hist', figsize=figsize, max_bins=max_bins, topk=topk, proportiontocut=proportiontocut))
            rows.append(row)
        display(HTML(ListTable(rows)._repr_html_("border:0")))

    def desc_detail(self, cols=None, output=['desc','vcounts','hist'], sort=True, return_html=True):
        # ['Description', 'Values', 'Histogram']
        if not cols:
            cols = self.cols
        rows = []
        for i, c in enumerate(cols):
            row = []
            for e in output:
                row.append(self.print_summary(c, e, figsize=(250,250), sort=sort, max_bins=25, topk=10))
            rows.append(row)
        if return_html:
            display(HTML(ListTable(rows)._repr_html_("border:0")))
        else:
            return rows

    def print_summary(self, c, output, figsize=(250,250), max_bins=25, topk=10, sort=True, proportiontocut=0):
        try:
            if output == 'summary':
                return self.tbl[c].mean().to_html()
            elif output == 'desc':
                return self.tbl[c].describe().to_frame().to_html()
            elif output == 'vcounts':
                if sort:
                    return self.tbl[c].value_counts(sort=sort).head(topk).to_frame().to_html()
                else:
                    return self.tbl[c].value_counts().sort_index().to_frame().to_html()
            elif output == 'hist':
                if self.vcounts[c] <  2:
                    return ""
                if self.dtypes[c] == "object":
                    p_input = self.tbl[c].dropna().value_counts().sort_index()
                    p = figure(x_range=p_input.index.tolist(), plot_width=figsize[0], plot_height=figsize[1], title=None,
                               toolbar_location=None, tools="")
                    p.vbar(x=p_input.index.values, top=p_input.values, width=0.5)                    
                else:
                    if self.dtypes[c] == "int64" and self.ncounts[c] > 0:
                        p_input = stats.trimboth(pd.to_numeric(self.tbl[c].dropna()), proportiontocut)
                    else:
                        p_input = stats.trimboth(self.tbl[c].dropna(), proportiontocut)
                    p = Histogram(p_input, plot_width=figsize[0], plot_height=figsize[1], 
                        bins=min(self.vcounts[c], max_bins), title=None, toolbar_location=None )
                p.yaxis.axis_label = "count"
                p.xaxis.axis_label = c
                p.xaxis.major_label_orientation = pi/4
                script, div = components(p)
                return script + div
        except Exception as e:
            print(c, e)

    ### CORRELATION ANALYSIS

    def corr(self, cols = None, method = "spearman"):
        """Correlation among Numerical Columns"""
        if not cols:
            cols = self.cols
        display(self.tbl[cols].corr(method=method).round(3).style.background_gradient(cmap='cool'))

    def corr_with(self, col, method = "spearman"):
        """Correlation with specific column"""
        ctbl = self.tbl.corr(method=method)[col][1:]
        display(ctbl.to_frame().transpose())
        ctbl.plot(kind='bar', figsize=(12,4))

    def pairplot(self, cols = None):
        if not cols:
            cols = self.cols
        tbl_full = self.tbl[cols].dropna()
        if len(tbl_full) < len(self.tbl):
            print("%d rows removed" % (len(self.tbl) - len(tbl_full)))
        sns.set(style="ticks")
        sns.pairplot(tbl_full)

    def pivot_ui(self, outfile_path="", url_prefix=""):
        from pivottablejs import pivot_ui
        return pivot_ui(self.tbl, outfile_path=outfile_path, url=(url_prefix+outfile_path))


class AggTable:
    """ TODO add analysis for aggregated data
    """
    def __init__(self, tbl):
        self.tbl = tbl
        self.cols = tbl.columns
        self.dtypes = {c: str(e) for c, e in tbl.dtypes.iteritems()}
        self.vcounts = {c : len(self.tbl[c].unique()) for c in cols}
        self.VCOUNT_MAX_CATEGORY = 20


class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in
        IPython Notebook.
    """
    def transpose(self):
        return ListTable(map(list, zip(*self)))

    def print_cell(self, arg):
        if arg:
            return arg.encode('utf-8')
        else:
            return ""

    def _repr_html_(self, style = ""):
        html = ["<table style='%s'>" % style]
        for row in self:
            html.append("<tr>")

            for col in row:
                html.append("<td>{0}</td>".format(self.print_cell(col)))

            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

