import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
sns.set(color_codes=True)
pd.options.display.max_rows = 999

from IPython.display import display, HTML, Image
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.charts import Histogram
from math import pi
output_notebook()


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.izip_longest(*args, fillvalue=fillvalue)


def get_histogram()


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

    def desc(self, cols=None, outputcol=5 , figsize=250, max_bins=25, topk=10, proportiontocut=0):
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
                try:
                    if self.vcounts[c] <= topk:
                        cdist = self.tbl[c].value_counts()
                        row.append(cdist.to_frame().to_html())
                    else:
                        if self.dtypes[c] == "datetime64[ns]":
                            continue
                            # p_input = self.tbl[c].dropna().dt.strftime("%Y%m%d:%H").value_counts().sort_index()
                            # p = figure(plot_width=figsize, plot_height=figsize)
                            # p.vbar(x=p_input.index, width=0.5, bottom=0, top=p_input.values)
                            # print p_input
                        else: 
                            if self.dtypes[c] == "int64" and self.ncounts[c] > 0:
                                p_input = stats.trimboth(pd.to_numeric(self.tbl[c].dropna()), proportiontocut)
                            else:
                                p_input = stats.trimboth(self.tbl[c].dropna(), proportiontocut)
                            p = Histogram(p_input, plot_width=figsize, plot_height=figsize, 
                                bins=min(self.vcounts[c], max_bins), title=None, toolbar_location=None )
                            p.yaxis.axis_label = "count"
                            p.xaxis.axis_label = c
                            p.xaxis.major_label_orientation = pi/4
                            script, div = components(p)
                            row.append(script + div)
                except Exception as e:
                    print c, e
            rows.append(row)
        display(HTML(ListTable(rows)._repr_html_("width:1250px;border:0")))

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
            print "%d rows removed" % (len(self.tbl) - len(tbl_full))
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

