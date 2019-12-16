import re
import sqlparse
import banjo.utils.gbq as gbq
import pandas as pd
SAMPLING_HASHKEY = "session_id"
import edatools.eda_display_utils as edu


def print_sql(sql):
    print(sqlparse.format(sql, reindent=True, keyword_case='upper'))


def list_to_str(arg):
    return ",".join(["'%s'" % e for e in arg])


def sample_bq_table(query_raw, project_id, n=None, sample_rate=None, hash_key=SAMPLING_HASHKEY,
                    random_seed=0, verbose=False, dialect='legacy', **kwargs):
    """
    Sample n rows of BQ table by 1) getting the results set size 2) querying with appropriate sample ratio
    - If n is None, just run the query without sampling
    - query_raw assumes should give one observation per row (i.e. no GROUP BY)
      - if you need to use this for GROUP BY results, please wrap it: SELECT * FROM (...)
    """
    # Get row count to determine sampling rate
    query_sample = query_raw
    if n:
        query_count = re.sub(r"^\s*SELECT\s+(.*?)\s+FROM",
                             "SELECT count(1) FROM", query_raw, re.M, re.DOTALL)
        if verbose:
            print_sql(query_count)
        rowcount = pd.read_gbq(query_count,
                               project_id=project_id, dialect=dialect, **kwargs).iloc[0, 0]
        if rowcount > n * 2:
            sample_rate = int(rowcount / n)
    if sample_rate:
        if re.search(r"\sWHERE\s", query_raw):
            where_clause = "AND"
        else:
            where_clause = "WHERE"
        if dialect == 'legacy':
            query_sample = query_raw + \
                " {} ABS(HASH({})) % {} == {}".format(
                    where_clause, hash_key, sample_rate, random_seed)
        else:
            query_sample = query_raw + \
                " {} MOD(ABS(FARM_FINGERPRINT({})), {}) = {}".format(
                    where_clause, hash_key, sample_rate, random_seed)
    if verbose:
        print_sql(query_sample.format(query_sample, reindent=True, keyword_case='upper'))
    # Run actual query with a sampling condition
    res = pd.read_gbq(query_sample,
                      project_id=project_id, dialect=dialect, **kwargs)
    return res



def agg_bq_table(facets, metrics, src_query, project_id, min_sample_size=0, verbose=False, **kwargs):
    """Build a aggregate table for given facets and metrics
    - facets = [[name1, def1], name2, ...]
    - metrics = [[name1, def1], name2, ...]

    Caveat:
    - Missing values in any of facets will result in omission of corresponding rows from the analysis
        - (go/ds-faq)
    """
    metric_list = edu.ensure_nested_list(metrics)
    facet_list = edu.ensure_nested_list(facets)
    facet_names = [e[0] for e in facet_list]
    facet_defs = ["{n} AS {m}".format(m=e[0], n=e[1])
                  for e in facet_list]
    sum_list = ["SUM({n}) AS {m}_sum".format(m=e[0], n=e[1])
                for e in metric_list]
    avg_list = ["AVG({n}) AS {m}_avg".format(m=e[0], n=e[1])
                for e in metric_list]
    ss_list = ["SUM(POW({n}, 2)) AS {m}_ssq".format(
        m=e[0], n=e[1]) for e in metric_list]
    sql = """
        SELECT
            {facet_defs},
            count(*) AS sample_size,
            {sum_list},
            {ss_list},
            {avg_list}
          FROM ({src_query})
          GROUP BY
            {facet_names}
          HAVING
            sample_size >= {min_sample_size}
          ORDER BY
            {facet_names}
    """.format(
        sum_list=",\n".join(sum_list),
        avg_list=",\n".join(avg_list),
        ss_list=",\n".join(ss_list),
        facet_names=",\n".join(facet_names),
        facet_defs=",\n".join(facet_defs),
        project_id=project_id,
        src_query=src_query,
        min_sample_size=min_sample_size)
    if verbose:
        print_sql(sql)
    smt = pd.read_gbq(sql, project_id=project_id, **kwargs)
    if len(smt.dropna()) < len(smt):
        smt_null = smt[smt.isnull().any(axis=1)]
        edu.print_title("[Warning] The results contain null values! <br>(%d out of %d; %.2f%% rows in the input table)" % \
            (smt_null.sample_size.sum(), smt.sample_size.sum(), 100 * smt_null.sample_size.sum() / smt.sample_size.sum()), 'b')
        if verbose:
            print(smt_null)
    return smt


def _get_date_range_str(start_date, end_date, interval):
    date_range = pd.date_range(
        start_date, end_date, tz="America/Los_Angeles", freq=interval)
    return [e.strftime("%Y%m%d") for e in date_range]


def get_daily_master_table(sql, start_date, end_date, dest_table_prefix, dest_dataset_id, project_id,
                           write_disposition='WRITE_EMPTY', interval='1D', verbose=False):
    """ Create daily master table """
    for table_date in _get_date_range_str(start_date, end_date, interval):
        table_name = dest_table_prefix + table_date
        print(table_name)
        daily_sql = sql.format(table_date=table_date)
        if verbose:
            print(daily_sql)
        try:
            gbq.submit_sync_query(daily_sql, project_id,
                                  write_disposition=write_disposition,
                                  dest_dataset_id=dest_dataset_id,
                                  dest_table_name=table_name,
                                  dialect='standard')
        except gbq.OverwriteExistingTableError as e:
            print(e)
