""" EDA Tools
- This is a single-file module that makes it easy to create visualization for arbitrary table,
by inferring column types and so on.
- You can do the same on seaborn / bokeh, but this util makes it easier to visualize
distributions for any subset of columns, or relationships btw. pairs of columns.
- Currently support both Seaborn and Bokeh (Bokeh support might discontinue)
"""
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
import scipy.stats as stats
import e3tools.eda_display_utils as edu
from pandas.api.types import CategoricalDtype

from io import BytesIO
from six import StringIO
from six.moves import urllib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from statsmodels.graphics.mosaicplot import mosaic
from IPython.display import display, HTML, Image

from math import pi
try:
    from itertools import zip_longest as zip_longest
except:
    from itertools import izip_longest as zip_longest

sns.set(color_codes=True)
sns.set(font_scale=1.2)
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.options.display.max_colwidth = 999
pd.options.display.max_seq_items = 999


def flatten_list(arg):
    return [item for sublist in arg for item in sublist]


def grouper(n, iterable, fillvalue=None):
    """grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"""
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def ensure_nested_list(arg):
    res = []
    for e in arg:
        if isinstance(e, str):
            res.append([e, e])
        else:
            res.append(e)
    return res


def add_category_dtype(df, colname, categories, ordered=True, c_suffix="_cat"):
    """Covert string column into ordered categories"""
    cdtype = CategoricalDtype(categories=categories, ordered=ordered)
    df[colname+c_suffix] = df[colname].astype(cdtype)


def add_numeric_dtype(df, colname, dtype="int"):
    """Covert string column into numbers"""
    if dtype=="int":
        df[colname+"_int"] = df[colname].str.extract("(\d+)", expand=False).astype("float").round()
    elif dtype=="double":
        df[colname+"_dbl"] = df[colname].str.extract("([\d\.]+)", expand=False).astype("double")


def _get_unique_items_from_csv_col(arg):
    """Create a list of unique values from a Series of CSV values"""
    return set(flatten_list([e.split(",") for e in arg.dropna().unique()]))


def add_feature_group_from_csv_col(df, colname, feature_list=None):
    """Covert string column with CSV values into ordered categories"""
    if not feature_list:
        feature_list = _get_unique_items_from_csv_col(df[colname])
    for f in feature_list:
        df[colname+"_"+re.sub(r'\W', "_", f)] = df[colname].str.contains(f).astype(str)


def _get_category_from_from_ptn(arg, cat_def):
    try:
        for e in ensure_nested_list(cat_def):
            if re.search(e[0], arg):
                return e[1]
        return "ETC"
    except:
        return None


def add_category_dtype_from_ptn(df, colname, cat_def):
    """Covert string column into ordered categories using the list of regex patterns
    Example:

        job_cat_def = [
            "Researcher",
            ("Engineer|Developer", "Engineer")
        ]

    """
    df[colname+"_cat"] = df[colname].apply(_get_category_from_from_ptn, cat_def=cat_def)
    add_category_dtype(df, colname+"_cat", [e[1] for e in ensure_nested_list(cat_def)], c_suffix="")


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


def convert_plotly_fig_to_html(fig, width=450, height=350):
    """ Convert Plotly figure 'fig' into a <img> tag for HTML use using base64 encoding. """

    data = base64.b64encode(fig.to_image(format='png', width=width, height=height))
    return '<img src="data:image/png;base64,{}">'.format(urllib.parse.quote(data))


def get_count(ses, vals):
    """ Count occourence of values within the series"""
    return len(ses[ses.isin(vals)])


class EDATable:
    """This class performs EDA (Exploratory Data Analysis) for (sampled) data
    """

    def __init__(self, tbl, dtypes={}):
        """Calculate per-column stats"""
        self.tbl = tbl
        self.cols = tbl.columns

        self.dtypes, self.vcdict, self.vcounts, self.ncounts = {}, {}, {}, {}
        for c in tbl.columns:
            # import pdb; pdb.set_trace()
            # print(self.tbl[c].head())

            if c in dtypes:
                if dtypes[c]['dtype'] == "datetime":
                    self.tbl[c] = pd.to_datetime(self.tbl[c], format=dtypes[c].get('format'), errors='coerce')
                    self.tbl[c+'_date'] = self.tbl[c].dt.date
                elif dtypes[c]['dtype'] == "category":
                    add_category_dtype(tbl, c, dtypes[c]['categories'], c_suffix="")
                else:
                    raise Exception("Invalid dtype! (%s)" % c)                    
            self.vcdict[c] = self.tbl[c].value_counts()
            self.dtypes[c] = str(self.vcdict[c].index.dtype)
            self.vcounts[c] = len(self.vcdict[c])
            self.ncounts[c] = self.tbl[c].isnull().sum()

    def run_eda(self):
        """Run basic EDA with single function"""
        self.desc()
        self.corr()
        self.pairplot(self.tbl.columns)

    def filter(self, query):
        return EDATable(self.tbl.query(query))

    def groupby(self, by, gfilter=None, min_count=25):
        """Return EDATable object for each row group"""
        for gname, gtbl in self.tbl.groupby(by):
            if len(gtbl) < min_count:
                continue
            if gfilter and gname not in gfilter:
                continue
            display(HTML("<h3>%s (n:%d)</h3>" % (str(gname), len(gtbl))))
            yield EDATable(gtbl)

    # COLUMN-LEVEL ANALYSIS

    def head(self, n=10):
        return self.tbl.head(n)

    def colinfo(self, n=3):
        """Print colum metadata
        """
        colnames = ["DataType", "ValueCount", "NullCount"]
        t_colinfo = pd.DataFrame([(self.dtypes[c], self.vcounts[c], self.ncounts[c])
                                  for c in self.cols], columns=colnames, index=self.cols)
        t_zero_counts = self.tbl.apply(get_count, vals=[0])
        t_zero_counts.name = "ZeroCount"
        t_head = self.tbl.head(n=3).transpose()
        print("Total: %d rows" % len(self.tbl))
        return pd.concat([t_colinfo, t_zero_counts, t_head], axis=1, sort=False)

    def desc(self, cols=None, col_ptn=None, outputcol=5, topk=10, **kwargs):
        """ Describe each column in table & plot (5 cols per row)
        """
        if col_ptn:
            cols = [c for c in self.cols if re.search(col_ptn, c)]
        elif not cols:
            cols = [c for c in self.cols
                    if (self.dtypes[c] != "object" or self.vcounts[c] < topk)
                    and self.vcounts[c] > 1 and self.ncounts[c] < len(self.tbl)]
        rows = []
        for colgroup in grouper(outputcol, cols):
            colgroup = [c for c in colgroup if c]
            row = []
            for i, c in enumerate(colgroup):
                if self.dtypes[c] in ("datetime64[ns]", "date"):
                    row.append(self.print_summary(c, 'hist', **kwargs))
                elif self.dtypes[c] == "object" and self.vcounts[c] > topk:
                    row.append(self.print_summary(c, 'vcounts', **kwargs))
                else:
                    row.append(self.print_summary(c, 'hist', **kwargs))
            rows.append(row)
        display(HTML(edu.ListTable(rows)._repr_html_("border:0")))

    def desc_ts(self, c_ts):
        """ Show column values by timestamp
        """
        pass

    def desc_detail(self, cols=None, col_ptn=None, output=['desc', 'vcounts', 'hist'], return_html=True, **kwargs):
        """ Describe each column in table & plot (one row per column)
        """
        if col_ptn:
            cols = [c for c in self.cols if re.search(col_ptn, c)]
        elif not cols:
            cols = self.cols
        rows = []
        for i, c in enumerate(cols):
            row = []
            for e in output:
                row.append(self.print_summary(c, e, **kwargs))
            rows.append(row)
        if return_html:
            display(HTML(edu.ListTable(rows)._repr_html_("border:0")))
        else:
            return rows

    def desc_group(self, group_col, cols=None, col_ptn=None, output='hist', min_count=10, return_html=True, **kwargs):
        """ Describe each column in table & plot (one row per column)
        """
        if col_ptn:
            cols = [c for c in self.cols if re.search(col_ptn, c)]
        elif not cols:
            cols = [c for c in self.cols if c != group_col]
        tbl_g = self.tbl.groupby(group_col).filter(lambda x: len(x) >= min_count).groupby(group_col)
        rows = []
        row = []
        for i, c in enumerate(cols):
            row = []
            for k,v in tbl_g:
                row.append("<b>%s: %s</b>" % (group_col, k))
            rows.append(row)
            row = []
            for k,v in tbl_g:
                row.append(EDATable(v).print_summary(c, output, **kwargs))
            rows.append(row)
        if return_html:
            display(HTML(edu.ListTable(rows)._repr_html_("border:0")))
        else:
            return rows

    def print_summary(self, c, output, sort=True, topk=10, **kwargs):
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
                return self.print_seaborn_hist(c, **kwargs)
        except Exception as e:
            print(c, e)

    def print_seaborn_hist(self, col, figsize=(5, 5), font_scale=1.2, max_bins=20, proportiontocut=0, sort_values=True):
        """ Print the histogram of a column using Seaborn 
        """
        sns.set(font_scale=font_scale)
        if self.dtypes[col] in ("datetime64[ns]", "date"):
            ax = self.tbl.groupby([self.tbl[col].dt.year, self.tbl[col].dt.month])[col].count().plot(kind='bar')
            ax.set_title(col+" (monthly)")
            ax.set_xlabel(None)
        elif (self.dtypes[col] == "object") or (self.dtypes[col] == "category"):
            p_input = self.tbl[col].dropna().value_counts()
            if sort_values:
                p_input = p_input.sort_values(ascending=False)[0:max_bins]
            else:
                p_input = p_input.sort_index()[0:max_bins]
            ax = sns.barplot(p_input.values, p_input.index.values, ci=None)
            ax.set_title(col)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
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
            ax = sns.distplot(p_input)
            ax.set_title(col + "\n(avg:%.2f p50:%.2f)" %
                         (np.mean(self.tbl[col].dropna()), np.median(self.tbl[col].dropna())))
        return convert_fig_to_html(ax.get_figure(), figsize)

    def outliers(self, cols=None, col_ptn=None, n_row=10, std_thr=5, show_all_cols=False, **kwargs):
        """ Show outliers for each column in table
        """
        if col_ptn:
            cols = [c for c in self.cols if re.search(col_ptn, c)]
        elif not cols:
            cols = self.cols
        rows = []
        target_cols = [c for c in cols if self.dtypes[c] not in ["datetime64[ns]", "object"]]

        for i, c in enumerate(target_cols):
            tbl_o = self.tbl[
                (np.abs(self.tbl[c] - self.tbl[c].mean()) > (std_thr * self.tbl[c].std()))]
            if len(tbl_o) > 0:
                edu.print_title("Column: %s" % c)
                print("%.3f +- %.1f * %.3f" %
                      (self.tbl[c].mean(), std_thr, self.tbl[c].std()))
                print("Total: %d Outlier: %d (%.3f%%)" %
                      (len(self.tbl), len(tbl_o), len(tbl_o) / len(self.tbl)))
                cm = sns.light_palette("red", as_cmap=True)
                if not show_all_cols:
                    tbl_o = tbl_o[[c] + [e for e in target_cols if e != c]]
                display(tbl_o[0:n_row].style.background_gradient(
                    subset=[c], cmap=cm))

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

    def is_empty(self, col):
        return len(self.tbl[col].dropna()) == 0

    def plot_datetime(self, dt_col, val_col, agg_unit='month', agg_func=np.mean):
        """Plot values aggregated by datetime"""
        if agg_unit == 'year':
            agg_cols = self.tbl[dt_col].dt.year
        if agg_unit == 'month':
            agg_cols = [self.tbl[dt_col].dt.year, self.tbl[dt_col].dt.month]
        if agg_unit == 'day':
            agg_cols = self.tbl[dt_col].dt.dayofyear
        elif agg_unit == None:            
            agg_cols = self.tbl[dt_col]
        return self.tbl.groupby(agg_cols).agg({val_col:agg_func}).plot(kind='bar')


    def pairplot(self, cols1=None, cols2=None, xlim=None, ylim=None, figsize=(5, 5), **kwargs):
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
                if self.is_empty(c1) or self.is_empty(c2):
                    return None
                elif c1 == c2:
                    row.append(self.print_summary(c1, 'hist', **kwargs))
                else:
                    if "datetime" in self.dtypes[c1] and self.dtypes[c2] != "object":
                        ax = self.plot_datetime(c1, c2, **kwargs)
                        fig = ax.get_figure()
                    elif "datetime" in self.dtypes[c2] and self.dtypes[c1] != "object":
                        ax = self.plot_datetime(c2, c1, **kwargs)
                        fig = ax.get_figure()
                    elif self.dtypes[c1] != "object" and self.dtypes[c2] != "object":
                        ax = sns.regplot(
                            self.tbl[c1], self.tbl[c2])
                        ax.set(xlim=xlim, ylim=ylim)
                        fig = ax.get_figure()
                    elif self.dtypes[c1] == "object" and self.dtypes[c2] != "object":
                        tbl_f = self.filter_topk_vals(self.tbl, c1)
                        ax = sns.boxplot(
                            x=c2, y=c1, data=tbl_f, orient="h")
                        ax.set(xlim=xlim)
                        fig = ax.get_figure()
                    elif self.dtypes[c1] != "object" and self.dtypes[c2] == "object":
                        tbl_f = self.filter_topk_vals(self.tbl, c2)
                        ax = sns.boxplot(
                            x=c2, y=c1, data=tbl_f)
                        ax.set(ylim=ylim)
                        fig = ax.get_figure()
                    else:
                        tbl_f = self.filter_topk_vals(
                            self.filter_topk_vals(self.tbl, c2), c1)
                        fig = mosaic(
                            tbl_f, index=[c2, c1], title="{} vs. {}".format(c2, c1))[0]
                    row.append(convert_fig_to_html(fig, figsize=figsize))
                    # fig.close()
            rows.append(row)
        display(HTML(edu.ListTable(rows)._repr_html_("border:0")))

    def pairplot_scatter_with_hover(self, cols1, cols2, **kwargs):
        """Group of scatterplot with hover & coloring (using bokeh)"""
        import e3tools.eda_display_js_utils as edju
        from bokeh.embed import components
        if isinstance(cols1, str):
            cols1 = [cols1]
        if isinstance(cols2, str):
            cols2 = [cols2]
        rows = []
        for c1 in cols1:
            row = []
            for i, c2 in enumerate(cols2):
                fig = edju.scatter_with_hover(self.tbl.dropna(
                    subset=[c1, c2]), c1, c2, **kwargs)
                script, div = components(fig)
                row.append(script + div)
            rows.append(row)
        display(HTML(edu.ListTable(rows)._repr_html_("border:0")))

    def pairplot_scatter(self, cols=None):
        """Pairplot for any numeric data types (using seaborn)"""
        if not cols:
            cols = self.cols
        tbl_full = self.tbl[cols].dropna()
        if len(tbl_full) < len(self.tbl):
            print("%d rows removed" % (len(self.tbl) - len(tbl_full)))
        sns.set(style="ticks")
        sns.pairplot(tbl_full)

    def get_topk_vals(self, col, k, ascending=True):
        """Get top-k values by count or coverage (% of rows covered)"""
        if k < 1:
            # Interpret k as coverage (0~1)
            c_sum = 0; i = 0
            total = len(self.tbl)
            for v,c in self.vcdict[col].items():
                c_sum += c; i += 1
                if c_sum / total:
                    k = i
                    break
        if ascending:
            return self.vcdict[col][0:k].index
        else:
            return self.vcdict[col][k:].index

    def filter_topk_vals(self, tbl, col, topk=5):
        """Leave only top-k categorical values for plotting and etc."""
        return tbl[tbl[col].isin(self.get_topk_vals(col, topk))]

