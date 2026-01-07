# Setup Scripts

One-time initialization scripts. Run these before the main pipeline.

## Scripts

- `00_fetch_sec_metadata.py` - Fetch company metadata from SEC EDGAR
- `00b_merge_metadata.py` - Merge SEC metadata with filing data
- `00c_fetch_outcomes_wrds.py` - Fetch outcome variables (returns) from WRDS
- `00d_wrds_data_merge.py` - Merge WRDS outcomes with filing data
- `fetch_controls_wrds.py` - Fetch control variables from WRDS
- `setup_wrds_env.py` - Configure WRDS credentials
- `test_wrds_connection.py` - Test WRDS database connection

Most of these require WRDS access.
