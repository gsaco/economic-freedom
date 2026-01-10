# P0/P1/P2 Upgrade Checklist

## P0: Critical (Must Fix Before Production)

- [ ] **PWT Download**: The Penn World Table direct download URL may change; implement fallback to Dataverse API or manual CSV path
- [ ] **KAOPEN Excel Format**: KAOPEN Excel format varies by version; add robust column detection or version pinning
- [ ] **Rate Limiting**: Add exponential backoff for WDI API (currently basic retry only)
- [ ] **ISO3 Concordance Validation**: ~5 EFW country names may not resolve to ISO3; audit and add manual mappings

## P1: High Priority (Should Fix Soon)

- [ ] **Add IMF WEO Fallback**: Implement IMF WEO downloader for inflation/fiscal when WDI coverage is weak
- [ ] **Add V-Dem Downloader**: For institutional/democracy measures (note: large dataset)
- [ ] **Coverage-Based Variable Selection**: Automate curating "Core" vs "Secondary" variables based on coverage thresholds
- [ ] **Quinquennial Macro Coverage**: Many macro variables have weak coverage at 1970-1990 wave years; document and handle gracefully
- [ ] **Add Unit Tests**: For downloaders, harmonization, and merge logic
- [ ] **Logging**: Replace print() with proper logging module for production use
- [ ] **Config File**: Externalize indicator codes, URLs, and thresholds to `config.yaml`

## P2: Nice to Have (Future Improvements)

- [ ] **Add Maddison Project**: Long-run GDP cross-checks (1820-present)
- [ ] **Add Crisis Database**: Reinhart-Rogoff or similar for banking/currency crisis dummies
- [ ] **Interpolation Option**: For quinquennial macro (with appropriate warnings)
- [ ] **Parallel Downloads**: Use async/concurrent.futures for faster multi-source downloads
- [ ] **Interactive Dashboards**: Add Plotly/Panel for interactive exploration
- [ ] **Causal Inference Section**: Add DiD/synthetic control examples (clearly marked as illustrative)
- [ ] **Country Profiles**: Generate per-country summary reports
- [ ] **Publication-Ready LaTeX Tables**: Auto-generate regression-style tables
- [ ] **Add OECD STAN**: For more detailed government finance indicators

## Completed ✓

- [x] **WDI Downloader**: With caching, retries, and 14 core indicators
- [x] **ISO3 Concordance**: 208 country names mapped with manual overrides
- [x] **EFW Merge**: With overlap accounting and sample definition dashboard
- [x] **Annual EDA**: 2000-2023, 3,865 observations, full correlation/trend analysis
- [x] **Quinquennial EDA**: 1970-2020, convergence analysis, transition matrices
- [x] **Balanced Panel Construction**: 98 countries across all 11 waves
- [x] **Missingness Dashboard**: By variable, decade, and region
- [x] **Outlier Tables**: High inflation episodes, growth collapses
- [x] **Macro-Conditioned Analysis**: EFW change by inflation/growth quartiles
