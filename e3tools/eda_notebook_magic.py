import re
import sys
import numpy as np
import pandas as pd
from plotnine import *
from mizani.breaks import date_breaks

from IPython.display import display, HTML
from IPython.core import magic_arguments
from IPython.core.magic import line_magic, cell_magic, line_cell_magic, Magics, magics_class

import e3tools.eda_display_utils as edu
import e3tools.eda_bq_utils as ebu
import e3tools.eda_table as et

def get_value(arglist, index):
    if index >= len(arglist):
        return None
    else:
        return arglist[index]

def parse_str_to_list(argstr):
    if argstr is None:
        return None
    return [e.strip() for e in argstr.split(",")]

def inject_vars(argstr, vdict):
    """Inject python variable to cell/line string
    Example:
    argstr = "{a} {b} {c} {c}"
    vdict = {'a':1, 'b':2, 'c':3}
    inject_vars(argstr, vdict)
    > '1 2 3 3'
    """
    str_vars = re.findall(r"\{\w+\}", argstr)
    res = argstr
    for v in set(str_vars):
        v2 = vdict[v[1:-1]]
        if isinstance(v2, list):
            v_str = ",".join(v2)
        else:
            v_str = str(v2)
        res = res.replace(v, v_str)
    return res

def plot_agg_table(agg_tbl, facets, metrics, aggfunc='sum', plot_type='bar', text_summary='sample_size', error_bars=True,
    sample_size_per_metric=False, min_sample_size=0, figsize_x=6, figsize_y=6, return_plots=False, 
    facet_scale='fixed', facet_ncol=None, labels={}, verbose=False, debug_filter=None, date_interval='7 days'):
    """ Faceted Visualization of Aggregate Statistics
    Args:
        agg_tbl
        facets: list of facets used for [x_axis, group, column, row]
        metrics: list of metrics (need the '_sum' suffix for aggfunc='sum')
        aggfunc: sum (simple summation) / avg (add error bar from 95% C.I.) / ratio (ratio of two sum metrics)
    """
    if plot_type=='bar':
        dodge = position_dodge(width=0.9)
    elif plot_type=='line':
        dodge = position_dodge(width=0)
    plots = []

    # Validate table schema for plotting
    try:
        for f in facets:
            if f and f not in agg_tbl.columns:
                raise Exception("Missing facet column: %s" % f)
        for m in metrics:
            if aggfunc=='sum' and m+'_sum' not in agg_tbl.columns:
                raise Exception("Missing metric column: %s_sum" % m)
            elif aggfunc=='avg' and m+'_ssq' not in agg_tbl.columns:
                raise Exception("Missing metric column: %s_ssq" % m)
    except Exception as e:
        print(e)
        return

    # Generate plot for each metric
    for m in metrics:
        # Prepare data 
        if aggfunc=='sum':
            tbl = agg_tbl.groupby([e for e in facets if e]).agg({(m+'_sum'):np.sum, 'sample_size':np.sum}).fillna(0)
        elif aggfunc=='ratio':
            tbl = agg_tbl.groupby([e for e in facets if e]).agg({(m[1][0]+'_sum'):np.sum, (m[1][1]+'_sum'):np.sum, 'sample_size':np.sum}).fillna(0)
            tbl[m[0]+'_ratio'] = tbl[(m[1][0]+'_sum')] / tbl[(m[1][1]+'_sum')]
            m = m[0]
        elif sample_size_per_metric:
            tbl = agg_tbl.groupby([e for e in facets if e]).apply(edu._stat_calc_ci, m, col_sample_size=(m + '_sample_size')).fillna(0)
        else:
            tbl = agg_tbl.groupby([e for e in facets if e]).apply(edu._stat_calc_ci, m).fillna(0)
            tbl = tbl.query("sample_size >= %d" % min_sample_size)

        tbl.reset_index(inplace=True)

        if verbose:
            debug_metric_suffixes = ['_sum', '_avg', '_sample_size']
            agg_tbl_f = agg_tbl[facets+[m+suffix for suffix in debug_metric_suffixes]+['sample_size']]
            if debug_filter:
                display(agg_tbl_f.query(debug_filter))
            else:
                display(tbl)

        # Basic Plot
        if len(facets) == 1:
            plot = ggplot(tbl, aes(get_value(facets, 0), ("%s_%s" % (m, aggfunc))));
        elif not get_value(facets, 1):
            plot = ggplot(tbl, aes(get_value(facets, 0), ("%s_%s" % (m, aggfunc))));            
        else:
            plot = ggplot(tbl, aes(get_value(facets, 0), ("%s_%s" % (m, aggfunc)), fill=get_value(facets, 1), color=get_value(facets, 1)));

        # Handle different chart types
        if plot_type == 'bar':
            plot += geom_bar(stat='identity', position=dodge)
        elif plot_type == 'line':
            plot += geom_line(size=1, position=dodge)

        # Set date interval for x-axis if needed
        if str(agg_tbl.dtypes.loc[get_value(facets, 0)]) in ("datetime64[ns]", "date"):
            plot += scale_x_datetime(breaks=date_breaks(date_interval)) 

        # Add faceting
        if len(facets) == 3:
            plot += facet_wrap('~%s' % facets[2], scales=facet_scale, ncol=facet_ncol, drop=True)
        elif len(facets) == 4:
            plot += facet_grid('%s~%s' % (facets[3], facets[2]), scales=facet_scale)
        if aggfunc == 'avg' and error_bars:
            plot += geom_errorbar(aes(x=get_value(facets, 0), ymax='%s_max' % m, ymin='%s_min' % m), 
                position=dodge);

        # Add summary & label & formatting
        if text_summary == 'y_value':
            label, format_string = "%s_%s" % (m, aggfunc), "{:.2f}"
        elif text_summary == 'sample_size':
            label, format_string = "sample_size", "n={:,.0f}"
        if text_summary != None:
            plot += geom_text(aes(y=-0.001, label=label), position=dodge, size=8, va='top', color='black', format_string=format_string);
        if len(labels) > 0:
            plot += labs(**labels)
        plot += theme(figure_size=(figsize_x, figsize_y));

        if return_plots:
            plots.append(plot)
            return plots
        else:
            plot.draw();

@magics_class
class EDAMagics(Magics):
    def init_eda(self, df, var_name):
        """Initialize EDA instance from $var_name$ """
        self.etbl = et.EDATable(df)
        display(self.etbl.colinfo())
        print("EDA instance initialized... (in '%s' variable)" % (var_name+'_e'))
        self.etbl = et.EDATable(df)
        self.shell.user_ns[var_name+'_e'] = self.etbl


    @cell_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('output', 
      help='The variable to return the results in'
    )
    @magic_arguments.argument('--project_id', '-p',
      help='The variable to return the results in'
    )
    @magic_arguments.argument('--sql_dialect', '-d',
      help='Sql direct (default:standard)',
      default='standard'
    )
    @magic_arguments.argument('--verbose', '-v',
        action='store_true',
        help='Whether to print the results'
    )
    def run_bq(self, line, cell):
        """Initialize EDA instance from  from BigQuery SQL"""
        args = magic_arguments.parse_argstring(self.eda_bq, inject_vars(line, self.shell.user_ns))
        self.query = inject_vars(cell, self.shell.user_ns)

        df = pd.read_gbq(self.query,
                          project_id=args.project_id, dialect=args.sql_dialect)

        self.shell.user_ns[args.output] = df
        print("BQ DataFrame initialized... (in '%s' variable)" % (args.output))


    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('input', 
        help='Pandas dataframe variable to analyze'
    )
    def eda(self, line):
        """Initialize EDA instance from  from DataFrame"""
        args = magic_arguments.parse_argstring(self.eda, line)        
        if args.input is None:
            print('[Error] No input specified!')
            return
        df = self.shell.user_ns[args.input]
        self.init_eda(df, args.input)


    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('--group', '-g',
      help='Group results by column'      
    )
    @magic_arguments.argument('--detail', '-d',
      help='Display detailed results for each column',
      action='store_true'
    )
    @magic_arguments.argument('--columns', '-c',
      help='Columns to display in output'      
    )
    def eda_distplot(self, line):
        """Show column data distribution"""
        args = magic_arguments.parse_argstring(self.eda_distplot, line)
        if args.columns:
            cols = parse_str_to_list(args.columns)
        else:
            cols = None
        if args.detail:
            self.etbl.desc_detail(cols)
        elif args.group:
            self.etbl.desc_group(args.group, cols)
        else:
            self.etbl.desc(cols, topk=50)


    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('--columns', '-c',
      help='Columns to display in output'      
    )
    def eda_pairplot(self, line):
        """Show column data distribution between columns"""
        args = magic_arguments.parse_argstring(self.eda_pairplot, line)
        if args.columns:
            cols = parse_str_to_list(args.columns)
        else:
            cols = None
        self.etbl.pairplot(cols)


    @cell_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('output', 
      help='The variable to return the results in'
    )
    @magic_arguments.argument('--project_id', '-p',
      help='The variable to return the results in'
    )
    @magic_arguments.argument('--key_column', '-k',
      help='Specify key column used for sampling'
    )
    @magic_arguments.argument('--sql_dialect', '-d',
      help='Sql direct (default:standard)',
      default='standard'
    )
    @magic_arguments.argument('--row_count', '-n',
      help='Specify target row count after sampling',
      type=int
    )
    @magic_arguments.argument('--verbose', '-v',
        action='store_true',
        help='Whether to print the results'
    )
    def eda_bq(self, line, cell):
        """Initialize EDA instance from  from BigQuery SQL"""
        args = magic_arguments.parse_argstring(self.eda_bq, inject_vars(line, self.shell.user_ns))
        # print(args)
        self.query = inject_vars(cell, self.shell.user_ns)
        df = ebu.sample_bq_table(self.query, args.project_id, 
            hash_key=args.key_column, n=args.row_count, verbose=args.verbose, dialect=args.sql_dialect)
        self.shell.user_ns[args.output] = df
        print("BQ DataFrame initialized... (in '%s' variable)" % (args.output))
        self.init_eda(df, args.output)


    @line_cell_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('output', 
      help='The variable to return the results in'
    )
    @magic_arguments.argument('facets', 
      help='Columns to group metrics by (if missing, aggregate globally)'
    )    
    @magic_arguments.argument('metrics', 
      help='Columns to compute metrics with (if missing, use all numeric columns)'
    )
    @magic_arguments.argument('--project_id', '-p',
      help='The variable to return the results in'
    )
    @magic_arguments.argument('--sql_dialect', '-d',
      help='Sql direct (default:standard)',
      default='standard'
    )
    @magic_arguments.argument('--verbose', '-v',
        help='Whether to print the results',
        action='store_true'
    )
    def agg_bq(self, line, cell=None):
        """Aggregate BQ table using facets and metrics provided"""
        # import pdb; pdb.set_trace()
        # print(line)
        # print(inject_vars(line, self.shell.user_ns))
        args = magic_arguments.parse_argstring(self.agg_bq, inject_vars(line, self.shell.user_ns))
        facets = parse_str_to_list(args.facets)
        metrics = parse_str_to_list(args.metrics)

        if cell:
            self.query = inject_vars(cell, self.shell.user_ns)
        else:
            print("Using base query (%s)" % self.query)

        self.atbl = ebu.agg_bq_table(facets, metrics, src_query=self.query, 
            project_id=args.project_id, verbose=args.verbose, dialect=args.sql_dialect)
        if args.verbose:
            display(self.atbl.head().transpose())
        self.shell.user_ns[args.output] = self.atbl


    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('facets',
      help='Columns for x-axis (additional columns used for color / column-facet / row-facet)'
    )    
    @magic_arguments.argument('metrics',
      help='Columns with metric (one plot for each metric)'
    )
    @magic_arguments.argument('--aggfunc', '-a',
      help='Aggregation function to apply (default:sum / avg)',
      default='sum'
    )
    @magic_arguments.argument('--min_sample_size', '-m',
      help='Mininum sample size per group (default: 0)',
      type=int, default=0
    )
    @magic_arguments.argument('--text_summary', '-t',
      help='Type of text summary (default: sample size)',
      default='sample_size'
    )
    @magic_arguments.argument('--figsize_x', '-x',
      help='Figure x-axis size in inches (default: 6)',
      type=float, default=6
    )
    @magic_arguments.argument('--figsize_y', '-y',
      help='Figure y-axis size in inches (default: 6)',
      type=float, default=6
    )
    def agg_plot(self, line):
        """Plot aggregation results using multiple facets (with error bar for avergage)"""
        args = magic_arguments.parse_argstring(self.agg_plot, line)
        facets = parse_str_to_list(args.facets)
        metrics = parse_str_to_list(args.metrics)
        plot_agg_table(self.atbl, facets, metrics, args.aggfunc, 'bar', text_summary=args.text_summary, figsize_x=args.figsize_x, figsize_y=args.figsize_y)


ip = get_ipython()
ip.register_magics(EDAMagics)
