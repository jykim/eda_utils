import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm
from IPython.display import display, HTML, Image
import plotly.graph_objects as go

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