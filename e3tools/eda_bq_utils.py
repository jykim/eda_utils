import re
import sqlparse
import pandas as pd
import e3tools.eda_display_utils as edu

def print_sql(sql):
    print(sqlparse.format(sql, reindent=True, keyword_case='upper'))


def list_to_str(arg):
    return ",".join(["'%s'" % e for e in arg])


def get_time_col_sql(col):
    """SQL snippet for time columns"""
    return "IF({col}>3600, 3600, IF({col}<=0, 0, IF(IS_NAN({col}), 0, {col})))".format(col=col)


def get_date_part_sql(c_input, c_output="year_month", part="MONTH"):
    return """
    CONCAT(
        CAST(EXTRACT(YEAR FROM DATE ({input})) AS STRING),
        "-", 
        FORMAT("%02d", EXTRACT({part} FROM DATE ({input})))
    ) AS {output}
    """.format(input=c_input, output=c_output, part=part)


def annotate_date_periods(d, period1, period2):
    """Annotate time periods given two periods
    # Input:
    period1 = (start_date1, end_date1)
    period1 = (start_date2, end_date2)
    """
    if d < period1[0]:
        return "Pre-P1"
    elif d <= period1[1]:
        return "P1"
    elif d < period2[0]:
        return "Btw-P1-P2"
    elif d <= period2[1]:
        return "P2"
    else:
        return "Post-P2"


def sample_bq_table(query_raw, project_id=None, hash_key=None, n=None, sample_rate=None,
                    random_seed=1, verbose=False, dialect='standard', arg_query={}, **kwargs):
    """
    Sample n rows of BQ table by two-stages:
    1) getting the results set size (N)
    2) querying with appropriate sample ratio, which is floor(N/n)
    You can skip Step 1 by providing sample_rate directly
    Sampling unit is determined by hash_key 

    # Caveat
    - If n is None, this just runs the query without sampling
    - query_raw assumes should give one observation per row (i.e. no GROUP BY)
      - if you need to use this for GROUP BY results, please wrap it: SELECT * FROM (...)
    """
    query_sample = query_raw.format(**arg_query)
    if n:
        query_count = re.sub(r"^\s*SELECT\s+(.*?)\s+FROM",
                             "SELECT count(1) FROM", query_raw.format(**arg_query), re.M, re.DOTALL)
        if verbose:
            print_sql(query_count)
        rowcount = pd.read_gbq(query_count,
                              project_id=project_id, dialect=dialect, **kwargs).iloc[0, 0]
        if rowcount > n * 2:
            sample_rate = int(rowcount / n)
            print("Sample Rate: %d / %d = %d" % (rowcount, n, sample_rate))
    if sample_rate:
        if re.search(r"\sWHERE\s", query_raw.format(**arg_query)):
            where_clause = "AND"
        else:
            where_clause = "WHERE"
        if hash_key is None:
            hash_key = "GENERATE_UUID()"
        if dialect == 'legacy':
            query_sample = query_raw.format(**arg_query) + \
                " {} ABS(HASH({})) % {} == {}".format(
                    where_clause, hash_key, sample_rate, random_seed)
        else:
            query_sample = query_raw.format(**arg_query) + \
                " {} MOD(ABS(FARM_FINGERPRINT({})), {}) = {}".format(
                    where_clause, hash_key, sample_rate, random_seed)
    if verbose:
        print_sql(query_sample.format(query_sample, reindent=True, keyword_case='upper'))

    res = pd.read_gbq(query_sample,
                      project_id=project_id, dialect=dialect, **kwargs)
    for k,v in arg_query.items():
        res[k] = v    
    return res



def agg_bq_table(facets, metrics, src_query, project_id=None, min_sample_size=0, verbose=False, arg_query={}, dialect='standard', **kwargs):
    """Build a aggregate table for given facets and metrics
    # Input 
    - facets = [[name1, def1], name2, ...]
    - metrics = [[name1, def1], name2, ...]
    # Output
    - Summary stats for all combinations of facets
    - Per-metric stats include sum / sum of sq. / avg / bsum (count in 0/1)
    Caveat:
    - Missing values in any of facets will result in omission of corresponding rows from the analysis
        - (go/ds-faq)
    """
    metric_list = edu.ensure_nested_list(metrics)
    if facets is None:
        facet_list = [["Overall", "'Overall'"]]
    else:
        facet_list = edu.ensure_nested_list(facets)
    facet_names = [e[0] for e in facet_list]
    facet_defs = ["{n} AS {m}".format(m=e[0], n=e[1])
                  for e in facet_list]
    sum_list = ["SUM({n}) AS {m}_sum".format(m=e[0], n=e[1])
                for e in metric_list]
    avg_list = ["AVG({n}) AS {m}_avg".format(m=e[0], n=e[1])
                for e in metric_list]
    bsum_list = ["SUM(IF({n}>0, 1, 0)) AS {m}_bsum".format(m=e[0], n=e[1])
                for e in metric_list]
    bavg_list = ["AVG(IF({n}>0, 1, 0)) AS {m}_bavg".format(m=e[0], n=e[1])
                for e in metric_list]
    pnull_list = ["SUM(IF({n} IS NULL, 1, 0)) AS {m}_pnull".format(m=e[0], n=e[1])
                for e in metric_list]
    sample_size_list = ["SUM(IF({n} is not null, 1, 0)) AS {m}_sample_size".format(m=e[0], n=e[1])
                for e in metric_list]
    ss_list = ["SUM(POW({n}, 2)) AS {m}_ssq".format(
        m=e[0], n=e[1]) for e in metric_list]
    sql = """
        SELECT
            {facet_defs},
            count(*) AS sample_size,
            {sum_list},
            {ss_list},
            {avg_list},
            {bsum_list},
            {bavg_list},
            {pnull_list},
            {sample_size_list}
          FROM ({src_query})
          GROUP BY
            {facet_names}
          HAVING
            sample_size >= {min_sample_size}
          ORDER BY
            {facet_names}
    """.format(
        sum_list=",\n".join(sum_list),
        ss_list=",\n".join(ss_list),
        avg_list=",\n".join(avg_list),
        bsum_list=",\n".join(bsum_list),
        bavg_list=",\n".join(bavg_list),
        pnull_list=",\n".join(pnull_list),
        sample_size_list=",\n".join(sample_size_list),
        facet_names=",\n".join(facet_names),
        facet_defs=",\n".join(facet_defs),
        project_id=project_id,
        src_query=src_query.format(**arg_query),
        min_sample_size=min_sample_size)
    if verbose:
        print_sql(sql)
    smt = pd.read_gbq(sql, project_id=project_id, dialect=dialect, **kwargs)
    for k,v in arg_query.items():
        smt[k] = v
    if len(smt.dropna()) < len(smt):
        smt_null = smt[smt.isnull().any(axis=1)]
        edu.print_title("[Warning] The results contain null values! <br>(%d out of %d; %.2f%% rows in the input table)" % \
            (smt_null.sample_size.sum(), smt.sample_size.sum(), 100 * smt_null.sample_size.sum() / smt.sample_size.sum()), 'b')
        if verbose:
            print(smt_null)
    return smt


def _get_date_range_str(start_date, end_date, interval, date_format="%Y%m%d"):
    date_range = pd.date_range(
        start_date, end_date, tz="America/Los_Angeles", freq=interval)
    return [e.strftime(date_format) for e in date_range]


