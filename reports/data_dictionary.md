# Diccionario de datos

| Variable | Descripcion |
|---|---|
| iso3 | Codigo ISO3 del pais. |
| year | Ano quinquenal (1970, 1975, ..., 2020). |
| country | Nombre del pais (EFW). |
| wb_region | Region WB. |
| wb_income_class | Clasificacion de ingreso WB. |
| efw_summary | Indice EFW total (0-10). |
| efw_area1..efw_area5 | Areas EFW: gobierno, sistema legal, moneda sana, comercio, regulacion. |
| efw_area2_nogender | Variante EFW area 2 sin indicador de genero (cuando aplica). |
| efw_1*..efw_5* | Subcomponentes EFW (prefijo efw_1, efw_2, ..., efw_5). |
| d_efw_* | Cambios quinquenales de EFW (prefijo d_). |
| shock_pos_* / shock_neg_* | Indicadores de shock positivo/negativo (cuantiles, SD o umbral). |
| shock_abs_d_efw_summary | Shock absoluto en cambio EFW summary (magnitud). |
| gdppc_wb | GDP per capita real (WB, USD constantes). |
| gdppc_growth_wb | Crecimiento anual GDPpc (WB, %). |
| gdppc_log | Log GDPpc (WB). |
| gdppc_growth_5y | Crecimiento quinquenal GDPpc (log diff). |
| population | Poblacion total (WB). |
| gcf_gdp | Formacion bruta de capital (% PIB, WB). |
| trade_gdp | Comercio total (% PIB, WB). |
| inflation_cpi | Inflacion CPI anual (WB, %). |
| gov_consumption_gdp | Consumo del gobierno (% PIB, WB). |
| pwt_gdppc | GDPpc PWT (PPP). |
| pwt_tfp | TFP (PWT). |
| pwt_capital | Stock de capital (PWT). |
| pwt_inv_share | Participacion de inversion (PWT). |
| hc | Indice de capital humano (PWT). |
| pwt_tfp_log / pwt_tfp_growth_5y | Log y crecimiento quinquenal de TFP. |
| pwt_capital_log / pwt_capital_growth_5y | Log y crecimiento quinquenal de capital. |
| pwt_inv_share_change | Cambio quinquenal en participacion de inversion. |
| currency / sovereign / systemic_banking | Indicadores de crisis (Laeven-Valencia). |
| sovereign_restructuring | Subtipo de crisis soberana. |
| any_crisis | Indicador agregado de crisis. |
| electoral_democracy / liberal_democracy | Indices V-Dem (quinquenal). |
| democracy / autocracy | Clasificacion binaria de regimen. |
| gini_net / gini_market | Gini neto y de mercado (SWIID). |
| gini_net_sd / gini_market_sd | Incertidumbre (SD) de Gini (SWIID). |
| country_wb / country_pwt | Etiquetas de pais por fuente (WB/PWT). |