import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm
from IPython.display import display, HTML, Image
import plotly.graph_objects as go

import bokeh
from bokeh.palettes import d3, brewer
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, Range1d, CategoricalColorMapper, LinearColorMapper
from bokeh.models.glyphs import VBar

def print_title(title, tag='h3'):
    display(HTML("<%s>%s</%s>" % (tag, title, tag)))
    

def display_sample_sessions(tbl, col_session, col_sort, n_session=5, n_row=15, session_ids=None, cols=None):
    """Display master table rows corresponding to a few sample sessions (based on col_session) """
    if not session_ids:
        session_ids = tbl[col_session].unique()[:n_session]
        print(session_ids)
    if not cols:
        cols = tbl.columns
    for k, v in tbl[tbl[col_session].isin(session_ids)].groupby([col_session]):
        display(v.sort_values(col_sort)[cols].head(n_row))


def ensure_nested_list(l):
    return [[e, e] if type(e) == str else e for e in l]


def get_human_name(arg):
    return arg.replace('_', ' ').title()


def agg_dataframe(tbl, facets, metrics):  
    return tbl.groupby(facets).apply(  
        _calc_agg_stats, metrics).reset_index()


def _calc_agg_stats(tbl, metrics): 
    res = {'sample_size': len(tbl)} 
    for m in metrics:
        res[m + '_avg'] = tbl[m].mean() 
        res[m + '_sum'] = tbl[m].sum()  
        res[m + '_ssq'] = tbl[m].apply(lambda e: e ** 2).sum()  
        res[m + '_ci95'] = _stat_calc_ci(pd.Series(res), m)[m+'_ci'] 
    return pd.Series(res)


def _stat_calc_ci(tbl, measure, alpha=0.05, col_sample_size='sample_size'):
    """Compute CI from aggregate table
     - Assumption: the sample size for each facet can be summed to get the total sample size
    """
    sample_size = tbl[col_sample_size].sum()
    total = float(tbl[measure + '_sum'].sum())
    avg = total / sample_size
    sample_size_corrected = sample_size - 1
    var = tbl[measure + '_ssq'].sum() / sample_size_corrected - total ** 2 / sample_size / sample_size_corrected
    ci = np.sqrt(var / sample_size) * norm.ppf(1-alpha/2)
    return pd.Series(
        {'sample_size': sample_size,
         (measure + '_sum'): total,
         (measure + '_avg'): avg,
         (measure + '_sd'): np.sqrt(var),
         (measure + '_ci'): ci},
        index=[measure + '_sum', measure + '_avg', measure + '_ci', measure + '_sd', 'sample_size'])


def sum_agg_table(tbl, facets, metrics, col_ptn=None):
    """"""
    if facets == ["Overall"]:
        tbl["Overall"] = "Overall"
    agg_tbls = [tbl.groupby(facets).apply(_stat_calc_ci, m) for m in metrics]
    if col_ptn:
        return pd.concat(agg_tbls, axis=1).filter(regex=col_ptn, axis=1)
    else:
        return pd.concat(agg_tbls, axis=1).reset_index()


def plot_facet_measure(facet, metric, **kwargs):
    return plot_avg_measure(facet, metric, **kwargs)


def plot_avg_measure(facet, metric, **kwargs):
    """Plot aggregate results with error bar"""
    ax = plt.gca()
    tbl = kwargs.pop("data")
    verbose = kwargs.pop("verbose", False)
    tbl_a = tbl.groupby(facet).apply(_stat_calc_ci, metric)
    if verbose:
        display(tbl_a[[(metric + '_avg'), (metric + '_ci')]])
    tbl_a[(metric + '_avg')].plot(yerr=tbl_a[(metric + '_ci')], ax=ax, **kwargs)


def plot_total_measure(facet, metric, **kwargs):
    ax = plt.gca()
    tbl = kwargs.pop("data")
    verbose = kwargs.pop("verbose", False)
    tbl_a = tbl.groupby(facet).agg({(metric+"_sum"):np.sum})
    if verbose:
        display(tbl_a[[(metric + '_sum')]].transpose())
    tbl_a[(metric + '_sum')].plot(ax=ax, **kwargs)


def get_color_by_ratio(val):
    colorscale=[[0.0, "rgb(165,0,38)"],
                [80, "rgb(215,48,39)"],
                [90, "rgb(244,109,67)"],
                [95, "rgb(253,174,97)"],
                [98, "rgb(254,224,144)"],
                [100, "rgb(224,243,248)"],
                [102, "rgb(171,217,233)"],
                [105, "rgb(116,173,209)"],
                [110, "rgb(69,117,180)"],
                [120, "rgb(49,54,149)"]]
    for e in colorscale:
        if val < e[0]:
            return e[1]


def plot_page_transition(tbl, node_count=10, min_transition_count=10000, width=1280, height=1024, 
        title="Page Transition Diagram", col_value='sample_size', col_label=None, verbose=False
    ):
    top_pages = tbl.groupby("source_page", as_index=False).agg({col_value:"sum"}).sort_values(col_value, ascending=False).iloc[:node_count]
    top_pages.reset_index(inplace=True)
    top_page_index = {r['source_page'] : i for i,r in top_pages.iterrows()}
    tbl_f = tbl[(tbl.source_page != tbl.page) & tbl.source_page.isin(top_pages.source_page) & tbl.page.isin(top_pages.source_page) & (tbl[col_value] >= min_transition_count)]
    if verbose:
        display(tbl_f)
    sources = [top_page_index[r['source_page']] for i,r in tbl_f.iterrows()]
    targets = [top_page_index[r['page']] for i,r in tbl_f.iterrows()]
    values = [r[col_value] for i,r in tbl_f.iterrows()]
    if col_label:
        labels = [r[col_label] for i,r in tbl_f.iterrows()]
        colors = [get_color_by_ratio(r[col_label]) for i,r in tbl_f.iterrows()]
    else:
        labels = None
        colors = None

    fig = go.Figure(data=[go.Sankey(
        orientation='h',
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          # color = "#444",
          label = top_pages.source_page.values,
          color = "gray"
        ),
        link = dict(
          source = sources,
          target = targets,
          value = values,
          label = labels,
          color = colors
      ))])

    fig.update_layout(title_text=title, width=width, height=height, font_size=10)
    fig.show()


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


def scatter_with_hover(df, x, y, hover_cols=None, marker="o", color=None, color_scale='categorical',
                       title=None, figsize=(300, 300), x_range=None, y_range=None, **kwargs):
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
    color : str
        Columns to be mapped into color value
    color_scale: str
        'categorical' vs 'linear' color scale
    **kwargs
        Any further arguments to be passed to fig.scatter
    """

    if color:
        col_color = df[color].unique().tolist()
        if color_scale == 'categorical':
            palette = d3['Category10'][min(max(len(col_color), 3), 10)]
            color_map = CategoricalColorMapper(factors=col_color,
                                               palette=palette)
        elif color_scale == 'linear':
            color_map = LinearColorMapper(palette=brewer['RdYlGn'][11], low=df[
                                          color].min(), high=df[color].max())
        color_val = {'field': color, 'transform': color_map}
    else:
        color_val = 'black'
    r, r_pval = stats.pearsonr(df[x], df[y])
    r_sig = "*" if r_pval <= 0.05 else ""
    rho, rho_pval = stats.spearmanr(df[x], df[y])
    rho_sig = "*" if rho_pval <= 0.05 else ""
    if title == 'correlation':
        title = 'Correlation (r:%.3f%s / rho:%.3f%s)' % (r,
                                                         r_sig, rho, rho_sig)
    else:
        title = ""
    source = ColumnDataSource(data=df)
    fig = figure(width=figsize[0], height=figsize[1], title=title, tools=[
                 'box_zoom', 'reset', 'wheel_zoom'])
    fig.scatter(x, y, source=source, name='main', marker=marker,
                color=color_val, legend_label=color, **kwargs)

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