import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm
from IPython.display import display, HTML, Image

import plotly.graph_objects as go
import plotly.offline as py
import plotly.figure_factory as ff

import bokeh
from bokeh.palettes import d3, brewer
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, Range1d, CategoricalColorMapper, LinearColorMapper
from bokeh.models.glyphs import VBar
import e3tools.eda_display_utils as edu


### FUNNEL PLOTS

FUNNEL_COLORS = [
    'rgb(32,155,160)', 'rgb(253,93,124)', 'rgb(28,119,139)', 'rgb(182,231,235)'
]

output_notebook(verbose=False, hide_banner=True)
res = """
    <link
        href="http://cdn.pydata.org/bokeh/release/bokeh-{version}.min.css"
        rel="stylesheet" type="text/css">
    <script src="http://cdn.pydata.org/bokeh/release/bokeh-{version}.min.js"></script>
    """
display(HTML(res.format(version=bokeh.__version__)))


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
        colors = [edu.get_color_by_ratio(r[col_label]) for i,r in tbl_f.iterrows()]
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


def plot_funnel_chart(
        phases, values, changes=None, title="Funnel Chart", norm_by_top=True, 
        section_height=60, plot_width=1000, funnel_width_ratio=0.7, fontsize_label=15):
    """Print funnel chart 
    Adapted from https://plot.ly/python/funnel-charts/

    Params:
    - phases: name of funnel phases i.e. ['Visit', 'Sign-up', 'Selection', 'Purchase', 'Review']
    - values: value for funnel phases i.e. [13873, 10553, 5443, 3703, 1708]
    - norm_by_top: normalize by top funnel value for prob. calculation
    - plot_width: total plot width in pixel
    - funnel_width_ratio: % of funnel width relative to plot width
     """
    n_phase = len(phases)
    if changes is None:
        funnel_colors=FUNNEL_COLORS
        probs = [100]
        for i in range(n_phase):
            if norm_by_top:
                probs.append("%.2f%%" % (values[i]/values[0]*100))
            else:
                probs.append("%.2f%%" % (values[i]/values[i-1]*100))
    else:
        probs, funnel_colors = [], []
        for i in range(n_phase):
            probs.append("Δ%.2f%%" % (changes[i]*100))            
            funnel_colors.append(get_color_by_ratio((changes[i]+1)*100))

    # height of a section and difference between sections 
    section_h = section_height
    section_d = section_height/10

    # multiplication factor to calculate the width of other sections
    unit_width = plot_width * funnel_width_ratio / max(values)

    # width of each funnel section relative to the plot width
    phase_w = [int(value * unit_width) for value in values]

    # plot height based on the number of sections and the gap in between them
    plot_height = section_h * n_phase + section_d * (n_phase - 1)
    height = plot_height

    offset = 150

    # list containing all the plot shapes
    shapes = []

    # list containing the Y-axis location for each section's name and value text
    label_y = []

    for i in range(n_phase):
            if (i == n_phase-1):
                points = [phase_w[i] / 2, height, phase_w[i] / 2, height - section_h]
            else:
                points = [phase_w[i] / 2, height, phase_w[i+1] / 2, height - section_h]
            path = 'M {0} {1} L {2} {3} L -{2} {3} L -{0} {1} Z'.format(*points)
            # print(i, points, path)

            shape = {
                'type': 'path',
                'path': path,
                'fillcolor': funnel_colors[i % len(funnel_colors)],
                'line': {
                    'width': 1,
                    'color': funnel_colors[i % len(funnel_colors)]
                }
            }
            shapes.append(shape)
            
            # Y-axis location for this section's details (text)
            label_y.append(height - (section_h))
            # label_y.append(height - (section_h / 2))

            height = height - (section_h + section_d)  
            
    # For phase names
    label_trace = go.Scatter(
        x=[-500]*n_phase,
        y=label_y,
        mode='text',
        text=["<b>%s</b><br>%d (%s)" % (phases[i], v, probs[i]) for i,v in enumerate(values)],
        textposition='top right',
        hoverinfo='none',
        textfont=dict(
            color='rgb(200,200,200)',
            size=fontsize_label
        )
    )

    data = [label_trace]
     
    layout = go.Layout(
        title="<b>{}</b>".format(title),
        titlefont=dict(
            size=20,
            color='rgb(203,203,203)'
        ),
        shapes=shapes,
        height=plot_height,
        width=plot_width,
        showlegend=False,
        autosize=True,
        paper_bgcolor='rgba(44,58,71,1)',
        plot_bgcolor='rgba(44,58,71,1)',
        xaxis=dict(
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            showticklabels=False,
            zeroline=False
        ),
        margin=go.layout.Margin(
            l=10,
            r=10,
            b=25,
            t=50,
            pad=2
        )
    )
     
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)    
    return fig


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
