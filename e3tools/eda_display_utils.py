import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm
from IPython.display import display, HTML, Image

def print_title(title, tag='h3', titlize=True):
    if titlize:
        title = title.replace("_", " ").title()
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


def background_color_df(val, thresholds=[3, 1, -3, -1]):
    """ Pandas Background Color: 4 shades of red to green according to thresholds provided
    :param val: str representation of percentages
    :param thresholds: percentage thresholds
    :return: background color for pandas dataframe
    """
    nval = float(re.sub('%', '', val))
    if nval > thresholds[0]:
        return ('background-color: rgb(102, 179, 102)')
    elif nval > thresholds[1]:
        return ('background-color: rgb(153, 204, 153)')
    elif nval < thresholds[2]:
        return ('background-color: rgb(245, 102, 102)')
    elif nval < thresholds[3]:
        return ('background-color: rgb(248, 153, 153)')
    return ('')


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
    """Plot average results with error bar
    Called within sns.FacetGrid, with example below: 
    g = sns.FacetGrid(tbl, col='section', hue='os_type')
    g.map_dataframe(edu.plot_avg_measure, "item_pos", metric, kind='line')
    """
    ax = plt.gca()
    tbl = kwargs.pop("data")
    verbose = kwargs.pop("verbose", False)
    tbl_a = tbl.groupby(facet).apply(_stat_calc_ci, metric)
    if verbose:
        display(tbl_a[[(metric + '_avg'), (metric + '_ci')]])
    tbl_a[(metric + '_avg')].plot(yerr=tbl_a[(metric + '_ci')], ax=ax, **kwargs)


def plot_total_measure(facet, metric, metric_suffix="_sum", **kwargs):
    """Plot total results"""
    ax = plt.gca()
    tbl = kwargs.pop("data")
    verbose = kwargs.pop("verbose", False)
    tbl_a = tbl.groupby(facet).agg({(metric+metric_suffix):np.sum})
    if verbose:
        display(tbl_a[[(metric + metric_suffix)]].transpose())
    tbl_a[(metric + metric_suffix)].plot(ax=ax, **kwargs)

### PAGE TRANSITION SANKY PLOT

def get_color_by_ratio(val, n=20):
    colors = [
        "rgb(%d,%d,%d)" % (e[0]*254, e[1]*254, e[2]*254) 
        for e in sns.diverging_palette(10, 140, n=n)
    ]
    ranges = [i for i in range(int(100-n/2), int(100+n/2), 1)]
    colorscale = [[ranges[i], colors[i]] for i in range(n)]
    for e in colorscale:
        if val < e[0]:
            return e[1]
    return colors[-1]



def build_funnel_tbl(tbl, fdef):
    ftbl = tbl[list(fdef.values())].transpose()
    ftbl['fid'] = list(fdef.keys())
    ftbl.sort_values('fid', inplace=True)
    return ftbl


def color_sig_red(val):
    color = 'red' if val < 0.05 else 'black'
    return 'color: %s' % color


def get_ratio_change_str(ctbl, pval):
    return "%.2f%% → %.2f%% (p:%.3f)" % (ctbl[1][0]/ctbl[0][0]*100, ctbl[1][1]/ctbl[0][1]*100, pval)


def calc_user_funnel_p_value(ftbl, verbose=False):
    """performs chi-squared tests for user funnel tables

    Test the null hypothesis that the ratio (e.g. conversion rate) between the two steps in a 
    user funnel is the same in two groups (e.g. the treatment and the control).

    The result is not valid for funnel tables with event counts.
    """

    res = []
    for cp in combinations(ftbl.columns[0:-1], 2):
        for rp in combinations(ftbl.index, 2):
            chisq_input = ftbl.loc[rp, cp].values
            chisq_input[0] = chisq_input[0] - chisq_input[1]
            _, pval, _, _ = stats.chi2_contingency(chisq_input.transpose())
            res.append(["T{0} vs T{1}".format(*cp), "{1} / {0}".format(*rp), chisq_input, pval, get_ratio_change_str(ftbl.loc[rp, cp].values, pval)])
    chisq_tbl = pd.DataFrame(res, columns=['Treatments', 'Metrics', 'Values', 'Pvalue', 'RatioChange'])
    if verbose:
        display(chisq_tbl)
    display(chisq_tbl.pivot(columns='Treatments', index='Metrics', values='RatioChange'))
    display(pd.pivot_table(chisq_tbl, columns='Treatments', index='Metrics', values='Pvalue').round(3).style.applymap(color_sig_red))
    print("Table of p-values from chi sq. test (ref: https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test)")


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
