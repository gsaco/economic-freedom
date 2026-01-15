# Research log

## Top hits by module (lowest p-values)
| module   | spec_id                                   | outcome              | term              |   horizon |        coef |       p_value |     q_value |
|:---------|:------------------------------------------|:---------------------|:------------------|----------:|------------:|--------------:|------------:|
| A        | A_linear_gdppc_growth_5y_d_efw_summary    | gdppc_growth_5y      | d_efw_summary     |         0 |  0.058655   |   1.58642e-08 | 2.68105e-06 |
| E        | E_autocracy                               | gdppc_growth_5y      | d_efw_summary     |         0 |  0.0711189  |   2.80166e-05 | 0.00217878  |
| A        | A_linear_pwt_inv_share_change_d_efw_area3 | pwt_inv_share_change | d_efw_area3       |         3 | -0.00823576 |   3.86766e-05 | 0.00217878  |
| A        | A_linear_gdppc_growth_5y_d_efw_area3      | gdppc_growth_5y      | d_efw_area3       |         0 |  0.0138926  |   0.000107514 | 0.00454245  |
| A        | A_linear_gdppc_growth_5y_d_efw_area2      | gdppc_growth_5y      | d_efw_area2       |         0 |  0.0209875  |   0.00266457  | 0.0865797   |
| A        | A_asym_gdppc_growth_5y                    | gdppc_growth_5y      | d_efw_summary_pos |         0 |  0.047828   |   0.00307384  | 0.0865797   |
| F        | F_collapse_lp                             | growth_collapse      | d_efw_summary     |         0 | -0.0732738  |   0.0130329   | 0.200233    |
| E        | E_democracy                               | gdppc_growth_5y      | d_efw_summary     |         0 |  0.0298362  |   0.0236857   | 0.307914    |
| E        | E_democracy                               | gdppc_growth_5y      | d_efw_summary     |         1 |  0.0381255  |   0.0382814   | 0.380562    |
| B        | B_legal_x_money                           | gdppc_growth_5y      | d_efw_area2       |         0 |  0.0502773  |   0.0697488   | 0.454806    |
| B        | B_legal_x_money                           | gdppc_growth_5y      | d_area2_x_area3   |         2 |  0.00609558 |   0.0791393   | 0.477662    |
| B        | B_trade_x_regulation                      | gdppc_growth_5y      | d_area4_x_area5   |         1 |  0.00445905 |   0.0852226   | 0.486823    |
| B        | B_legal_x_money                           | gdppc_growth_5y      | d_efw_area2       |         2 | -0.0392036  |   0.136177    | 0.547949    |
| B        | B_legal_x_money                           | gdppc_growth_5y      | d_efw_area2       |         3 |  0.0443402  |   0.148617    | 0.558141    |
| F        | F_collapse_lp                             | growth_collapse      | d_efw_summary     |         3 |  0.0322571  |   0.192513    | 0.650694    |
| D        | D_crisis_interaction                      | gdppc_growth_5y      | any_crisis        |         0 | -0.0623614  |   0.207555    | 0.655254    |
| E        | E_autocracy                               | gdppc_growth_5y      | d_efw_summary     |         3 | -0.0518855  |   0.209371    | 0.655254    |
| E        | E_democracy                               | gdppc_growth_5y      | d_efw_summary     |         3 | -0.0172011  |   0.306486    | 0.65859     |
| C        | C_bundles                                 | gdppc_growth_5y      | bundle_1          |         0 |  0.0775799  |   0.347858    | 0.683741    |
| F        | F_collapse_lp                             | growth_collapse      | d_efw_summary     |         2 | -0.0244326  |   0.36882     | 0.692971    |
| D        | D_crisis_interaction                      | gdppc_growth_5y      | any_crisis        |         1 | -0.0446056  |   0.400811    | 0.692971    |
| D        | D_crisis_interaction                      | gdppc_growth_5y      | crisis_x_efw      |         1 |  0.00696998 |   0.400981    | 0.692971    |
| F        | F_collapse_lp                             | growth_collapse      | d_efw_summary     |         1 | -0.0287875  |   0.4105      | 0.692971    |
| D        | D_crisis_interaction                      | gdppc_growth_5y      | crisis_x_efw      |         2 |  0.00395563 |   0.580641    | 0.810978    |
| D        | D_crisis_interaction                      | gdppc_growth_5y      | any_crisis        |         3 |  0.0231258  |   0.742901    | 0.916426    |
| C        | C_bundles                                 | gdppc_growth_5y      | bundle_3          |         0 |  0.00949354 |   0.940416    | 0.98931     |
| C        | C_bundles                                 | gdppc_growth_5y      | bundle_1          |         1 |  0.00657678 | nan           | 1           |
| C        | C_bundles                                 | gdppc_growth_5y      | bundle_0          |         2 |  0          | nan           | 1           |
| C        | C_bundles                                 | gdppc_growth_5y      | bundle_1          |         2 |  0.347883   | nan           | 1           |

## Notes
- Spec ledger stored in outputs/spec_ledger.csv with BH/FDR q-values.
- Dataset hash: 868302632f1f78e806748c2cbf7becc63dcb02d41e25254899706480a7e30feb