""" EDA Tools
- This is a single-file module that makes it easy to create visualization for arbitrary table, 
by inferring column types and so on. 
- You can do the same on seaborn / bokeh, but this util makes it easier to visualize  
distributions for any subset of columns, or relationships btw. pairs of columns.
- Currently support both Seaborn and Bokeh (Bokeh support might discontinue)
"""
import base64
import bokeh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
import scipy.stats as stats

from io import BytesIO
from six import StringIO
from six.moves import urllib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from statsmodels.graphics.mosaicplot import mosaic
from IPython.display import display, HTML, Image

from bokeh.palettes import d3
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, Range1d, CategoricalColorMapper
from bokeh.models.glyphs import VBar
from bokeh.embed import components
from math import pi
try:
    from itertools import zip_longest as zip_longest
except:
    from itertools import izip_longest as zip_longest

sns.set(color_codes=True)
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.options.display.max_colwidth = 999
pd.options.display.max_seq_items = 999

SAMPLING_HASHKEY = "session_id"
PLOT_LIB = "seaborn"
BOKEH_LOADED = False


def grouper(n, iterable, fillvalue=None):
    """grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"""
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def load_bokeh():
    """Initialize Bokeh JS for notebook display"""
    output_notebook(verbose=False, hide_banner=True)
    res = """
        <link
            href="http://cdn.pydata.org/bokeh/release/bokeh-{version}.min.css"
            rel="stylesheet" type="text/css">
        <script src="http://cdn.pydata.org/bokeh/release/bokeh-{version}.min.js"></script>
        """
    BOKEH_LOADED = True
    display(HTML(res.format(version=bokeh.__version__)))


def sample_bq_table(query_raw, project_id, n=None, sample_rate=None, hash_key=SAMPLING_HASHKEY, random_seed=0, debug=False, **kwargs):
    """
    Sample n rows of BQ table by 1) getting the results set size 2) querying with appropriate sample ratio
    - If n is None, just run the query without sampling
    - query_raw assumes should give one observation per row (i.e. no GROUP BY)
      - if you need to use this for GROUP BY results, please wrap it: SELECT * FROM (...) 
    """
    # Get row count to determine sampling rate
    query_sample = query_raw
    if n or sample_rate:
        if not sample_rate:
            query_count = re.sub(r"^\s*SELECT\s+(.*?)\s+FROM",
                                 "SELECT count(1) FROM", query_raw, re.M, re.DOTALL)
            if debug:
                print(query_count)
            rowcount = pd.read_gbq(query_count, verbose=False,
                                   project_id=project_id, **kwargs).iloc[0, 0]
            sample_rate = int(rowcount / n)
        if sample_rate or rowcount > n * 2:
            if re.search(r"\sWHERE\s", query_raw):
                where_clause = "AND"
            else:
                where_clause = "WHERE"
            query_sample = query_raw + \
                " {} ABS(HASH({})) % {} == {}".format(
                    where_clause, hash_key, sample_rate, random_seed)
    if debug:
        print(query_sample)
    # Run actual query with a sampling condition
    res = pd.read_gbq(query_sample, verbose=False,
                      project_id=project_id, **kwargs)
    if n and debug:
        print("%d / %d = %d" % (rowcount, sample_rate, len(res)))
    return res


def convert_fig_to_html(fig, figsize=(5, 5)):
    """ Convert Matplotlib figure 'fig' into a <img> tag for HTML use using base64 encoding. """
    fig.set_size_inches(*figsize)
    fig.tight_layout()
    canvas = FigureCanvas(fig)
    png_output = BytesIO()
    canvas.print_png(png_output)
    data = base64.b64encode(png_output.getvalue())
    plt.clf()
    return '<img src="data:image/png;base64,{}">'.format(urllib.parse.quote(data))


def scatter_with_hover(df, x, y, hover_cols=None, marker="o", figsize=(300, 300), x_range=None, y_range=None, color=None, **kwargs):
    """
    Plots an interactive scatter plot of `x` vs `y` using bokeh, with automatic tooltips showing columns from `df`.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to be plotted
    x : str
        Name of the column to use for the x-axis values
    y : str
        Name of the column to use for the y-axis values
    hover_cols : list of str
        Columns to show in the hover tooltip (default is to show all)
    x_range : tuble of size 2
        range (min_value, max_value) of the x axis
    marker : str
        Name of marker to use for scatter plot
    **kwargs
        Any further arguments to be passed to fig.scatter
    """

    if color:
        col_color = df[color].unique().tolist()
        palette = d3['Category10'][min(max(len(col_color), 3), 10)]
        # print(col_color, palette)
        color_map = CategoricalColorMapper(factors=col_color,
                                           palette=palette)
        color_dict = {'field': color, 'transform': color_map}
    else:
        color_dict = 'black'

    source = ColumnDataSource(data=df)
    fig = figure(width=figsize[0], height=figsize[
                 1], tools=['box_zoom', 'reset'])
    fig.scatter(x, y, source=source, name='main', marker=marker,
                color=color_dict, legend=color, **kwargs)

    if x_range:
        fig.x_range = Range1d(*x_range)
    if y_range:
        fig.y_range = Range1d(*y_range)

    hover = HoverTool(names=['main'])
    if hover_cols is None:
        hover.tooltips = [(c, '@' + c) for c in df.columns]
    else:
        hover.tooltips = [(c, '@' + c) for c in hover_cols]
    fig.add_tools(hover)

    fig.yaxis.axis_label = y
    fig.xaxis.axis_label = x
    return fig


class RawTable:
    """This class performs EDA (Exploratory Data Analysis) for (sampled) raw data
    """

    def __init__(self, tbl):
        """Calculate per-column stats"""
        self.tbl = tbl
        self.cols = tbl.columns

        self.dtypes, self.vcdict, self.vcounts, self.ncounts = {}, {}, {}, {}
        for c in tbl.columns:
            self.vcdict[c] = self.tbl[c].value_counts()
            self.dtypes[c] = str(self.vcdict[c].index.dtype)
            self.vcounts[c] = len(self.vcdict[c])
            self.ncounts[c] = self.tbl[c].isnull().sum()

    def run_eda(self):
        """Run basic EDA with single function"""
        self.desc()
        self.corr()
        self.pairplot(self.tbl.columns)

    def groupby(self, by, gfilter=True, min_count=25):
        """Return RawTable object for each row group"""
        for gname, gtbl in self.tbl.groupby(by):
            if len(gtbl) < min_count:
                continue
            if gfilter and gname not in gfilter:
                continue
            display(HTML("<h3>%s: %s</h3>" % (str(by), gname)))
            yield RawTable(gtbl)

    def groupby_tbl(self, by, gfilter=None, min_count=25):
        """Generate RawTable object for each row group"""
        rows = [[], []]
        for gname, gtbl in self.tbl.groupby(by):
            if len(gtbl) < min_count:
                continue
            rows[0].append("<h3>%s: %s</h3>" % (str(by), gname))
            res = yield RawTable(gtbl)
            rows[1].append(res)
        display(HTML(ListTable(rows)._repr_html_("border:0")))

    # COLUMN-LEVEL ANALYSIS

    def head(self, n=10):
        return self.tbl.head(n)

    def colinfo(self, n=3):
        """Print colum metadata
        """
        colnames = ["DataType", "ValueCount", "NullCount"]
        t_colinfo = pd.DataFrame([(self.dtypes[c], self.vcounts[c], self.ncounts[c])
                                  for c in self.cols], columns=colnames, index=self.cols)
        t_head = self.tbl.head(n=3).transpose()
        return pd.concat([t_colinfo, t_head], axis=1)

    def desc(self, cols=None, col_ptn=None, outputcol=5, topk=10, **kwargs):
        """ Describe each column in table & plot (5 cols per row)
        """
        if col_ptn:
            cols = [c for c in self.cols if re.match(col_ptn, c)]
        elif not cols:
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
                    row.append(self.print_summary(c, 'vcounts', **kwargs))
                else:
                    row.append(self.print_summary(c, 'hist', **kwargs))
            rows.append(row)
        display(HTML(ListTable(rows)._repr_html_("border:0")))

    def desc_detail(self, cols=None, col_ptn=None, output=['desc', 'vcounts', 'hist'], return_html=True, **kwargs):
        """ Describe each column in table & plot (one row per column)
        """
        if col_ptn:
            cols = [c for c in self.cols if re.match(col_ptn, c)]
        elif not cols:
            cols = self.cols
        rows = []
        for i, c in enumerate(cols):
            row = []
            for e in output:
                row.append(self.print_summary(c, e, **kwargs))
            rows.append(row)
        if return_html:
            display(HTML(ListTable(rows)._repr_html_("border:0")))
        else:
            return rows

    def print_summary(self, c, output, sort=True, topk=10, plot_lib=PLOT_LIB, **kwargs):
        """ Summarize a column with appropriate table / visualization
        """
        try:
            if output == 'summary':
                return self.tbl[c].mean().to_html()
            elif output == 'desc':
                return self.tbl[c].describe(percentiles=[.1, .25, .5, .75, .9]).to_frame().to_html()
            elif output == 'vcounts':
                if sort:
                    return self.tbl[c].value_counts(sort=sort).head(topk).to_frame().to_html()
                else:
                    return self.tbl[c].value_counts().sort_index().to_frame().to_html()
            elif output == 'hist':
                if self.vcounts[c] < 2:
                    return ""
                elif plot_lib == "bokeh":
                    return self.print_bokeh_hist(c, **kwargs)
                elif plot_lib == "seaborn":
                    return self.print_seaborn_hist(c, **kwargs)
                else:
                    raise Exception("Invalid input!")
        except Exception as e:
            # import traceback
            # traceback.print_exc()
            print(c, e)

    def print_bokeh_hist(self, col, figsize=(250, 250), max_bins=25, proportiontocut=0):
        """ Print the histogram of a column using Boken """
        if self.dtypes[col] == "object":
            p_input = self.tbl[col].dropna().value_counts().sort_index()
            p = figure(x_range=p_input.index.tolist(), plot_width=figsize[0], plot_height=figsize[1], title=None,
                       toolbar_location=None, tools="")
            p.vbar(x=p_input.index.values, top=p_input.values,
                   width=0.5, line_color="#033649")
        else:
            if self.dtypes[col] == "int64" and self.ncounts[col] > 0:
                p_input = stats.trimboth(pd.to_numeric(
                    self.tbl[col].dropna()), proportiontocut)
            else:
                p_input = stats.trimboth(
                    self.tbl[col].dropna(), proportiontocut)
            hist, edges = np.histogram(
                p_input, density=False, bins=min(self.vcounts[col], max_bins))
            p = figure(plot_width=figsize[0], plot_height=figsize[1], title=None,
                       toolbar_location=None, tools="")
            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                   fill_color="#E05F68", line_color="#033649")
        p.yaxis.axis_label = "count"
        p.xaxis.axis_label = col
        p.xaxis.major_label_orientation = pi / 4
        script, div = components(p)
        return script + div

    def print_seaborn_hist(self, col, figsize=(5, 5), font_scale=1.2, max_bins=20, proportiontocut=0, sort_values=True):
        """ Print the histogram of a column using Seaborn """
        sns.set(font_scale=font_scale)
        if self.dtypes[col] == "object":
            p_input = self.tbl[col].dropna().value_counts()
            if sort_values:
                p_input = p_input.sort_values(ascending=False)[0:max_bins]
            else:
                p_input = p_input.sort_index()[0:max_bins]
            ax = sns.barplot(p_input.values, p_input.index.values, ci=None)                
        else:
            if self.dtypes[col] == "int64" and self.ncounts[col] > 0:
                p_input = stats.trimboth(pd.to_numeric(
                    self.tbl[col].dropna()), proportiontocut)
            else:
                p_input = stats.trimboth(
                    self.tbl[col].dropna(), proportiontocut)
            hist, edges = np.histogram(
                p_input, density=False, bins=min(self.vcounts[col], max_bins))
            edges_n = [np.round((edges[i] + edges[i + 1]) / 2, 1)
                       for i in range(len(edges) - 1)]
            ax = sns.barplot(x=edges_n, y=hist, ci=None, color="#E05F68")
            # ax = sns.distplot(p_input, axlabel=col)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title(col)
        return convert_fig_to_html(ax.get_figure(), figsize)

    # COLUMN-PAIR ANALYSIS

    def corr(self, cols=None, method="spearman"):
        """Correlation among Numerical Columns"""
        if not cols:
            cols = self.cols
        display(self.tbl[cols].corr(method=method).round(
            3).style.background_gradient(cmap='cool'))

    def corr_with(self, col, method="spearman"):
        """Correlation with specific column"""
        ctbl = self.tbl.corr(method=method)[col][1:]
        display(ctbl.to_frame().transpose())
        ctbl.plot(kind='bar', figsize=(12, 4))

    def pairplot(self, cols1=None, cols2=None, plot_lib=PLOT_LIB, **kwargs):
        """Pairplot for any data types"""
        if isinstance(cols1, str):
            cols1 = [cols1]
        if isinstance(cols2, str):
            cols2 = [cols2]
        if not cols1:
            cols1 = self.tbl.columns
        if not cols2:
            cols2 = cols1
        rows = []
        for c1 in cols1:
            row = []
            for i, c2 in enumerate(cols2):
                if c1 == c2:
                    row.append(self.print_summary(c1, 'hist', **kwargs))
                else:
                    if self.dtypes[c1] != "object" and self.dtypes[c2] != "object":
                        fig = sns.regplot(
                            self.tbl[c1], self.tbl[c2]).get_figure()
                    elif self.dtypes[c1] == "object" and self.dtypes[c2] != "object":
                        tbl_f = self.filter_rare_vals(self.tbl, c1)
                        fig = sns.swarmplot(
                            x=c2, y=c1, data=tbl_f, orient="h").get_figure()
                    elif self.dtypes[c1] != "object" and self.dtypes[c2] == "object":
                        tbl_f = self.filter_rare_vals(self.tbl, c2)
                        fig = sns.swarmplot(
                            x=c2, y=c1, data=tbl_f).get_figure()
                    else:
                        tbl_f = self.filter_rare_vals(
                            self.filter_rare_vals(self.tbl, c2), c1)
                        fig = mosaic(
                            tbl_f, index=[c2, c1], title="{} vs. {}".format(c2, c1))[0]
                    row.append(convert_fig_to_html(fig, **kwargs))
            rows.append(row)
        display(HTML(ListTable(rows)._repr_html_("border:0")))

    def pairplot_scatter_with_hover(self, cols1, cols2, **kwargs):
        """Group of scatterplot with hover & coloring (using bokeh)"""
        if not BOKEH_LOADED:
            load_bokeh()
        if isinstance(cols1, str):
            cols1 = [cols1]
        if isinstance(cols2, str):
            cols2 = [cols2]
        rows = []
        for c1 in cols1:
            row = []
            for i, c2 in enumerate(cols2):
                fig = scatter_with_hover(self.tbl, c1, c2, **kwargs)
                script, div = components(fig)
                row.append(script + div)
            rows.append(row)
        display(HTML(ListTable(rows)._repr_html_("border:0")))

    def pairplot_scatter(self, cols=None):
        """Pairplot for any numeric data types (using seaborn)"""
        if not cols:
            cols = self.cols
        tbl_full = self.tbl[cols].dropna()
        if len(tbl_full) < len(self.tbl):
            print("%d rows removed" % (len(self.tbl) - len(tbl_full)))
        sns.set(style="ticks")
        sns.pairplot(tbl_full)

    def filter_rare_vals(self, tbl, col, min_n=10):
        """Remove rare (n<=min_n) categorical values for plotting and etc."""
        return tbl[tbl[col].isin(self.vcdict[col].loc[lambda x: x >= min_n].index)]

    def pivot_ui(self, outfile_path="./tmp.html", url_prefix=""):
        """Wrpper around pivottablejs"""
        from pivottablejs import pivot_ui
        return pivot_ui(self.tbl, outfile_path=outfile_path, url=(url_prefix + outfile_path))


class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in
        IPython Notebook.
    """

    def transpose(self):
        return ListTable(map(list, zip(*self)))

    def print_cell(self, arg):
        if arg:
            return arg
        else:
            return ""

    def _repr_html_(self, style=""):
        html = ["<table style='%s'>" % style]
        for row in self:
            html.append("<tr style='%s'>" % style)

            for col in row:
                html.append(
                    "<td style='{}'>{}</td>".format(style, self.print_cell(col)))

            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)
