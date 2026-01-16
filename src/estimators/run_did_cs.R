args <- commandArgs(trailingOnly = TRUE)
input_path <- args[1]
output_path <- args[2]
diag_path <- args[3]
outcome_arg <- if (length(args) >= 4) args[4] else ""

ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

ensure_pkg("did")
ensure_pkg("dplyr")
ensure_pkg("readr")

library(did)
library(dplyr)
library(readr)

panel <- read_csv(input_path, show_col_types = FALSE)

outcomes <- c(
  "efw_overall",
  "efw_delta5",
  "reform_event_03",
  "reversal_event_03",
  "hazard_rev_03",
  "delta5_log_gdp_pc",
  "delta5_log_tfp",
  "delta5_inv_share_gdp",
  "delta5_inflation_cpi",
  "delta5_inflation_volatility"
)
if (nzchar(outcome_arg)) {
  outcomes <- unlist(strsplit(outcome_arg, ","))
}

first_treat <- panel %>%
  group_by(iso3) %>%
  summarise(first_treat = ifelse(any(WTO_neg_share > 0, na.rm = TRUE), min(year[WTO_neg_share > 0], na.rm = TRUE), 0), .groups = "drop")

panel <- panel %>% left_join(first_treat, by = "iso3")

results <- list()
diags <- list()

for (outcome in outcomes) {
  if (!outcome %in% names(panel)) next

  df <- panel %>% select(iso3, year, first_treat, all_of(outcome))
  df <- df %>% filter(!is.na(.data[[outcome]]))
  df <- df %>% mutate(id = as.numeric(factor(iso3)))

  if (nrow(df) == 0) next

  att <- tryCatch(
    did::att_gt(
      yname = outcome,
      tname = "year",
      idname = "id",
      gname = "first_treat",
      data = df,
      control_group = "notyettreated",
      bstrap = TRUE,
      biters = 200,
      clustervars = "iso3"
    ),
    error = function(e) NULL
  )
  if (is.null(att)) next

  dynamic <- did::aggte(att, type = "dynamic")

  est_df <- tibble(
    anchor = "WTO",
    cohort = "all",
    outcome = outcome,
    event_time = dynamic$egt,
    att = dynamic$att,
    se = dynamic$se,
    ci_low = dynamic$att - 1.96 * dynamic$se,
    ci_high = dynamic$att + 1.96 * dynamic$se,
    spec_id = "cs_base"
  )
  results[[length(results) + 1]] <- est_df

  diags[[length(diags) + 1]] <- tibble(
    anchor = "WTO",
    outcome = outcome,
    pretrend_p = NA_real_
  )
}

if (requireNamespace("fixest", quietly = TRUE)) {
  library(fixest)
  ensure_pkg("broom")
  for (outcome in outcomes) {
    if (!outcome %in% names(panel)) next
    df <- panel %>% select(iso3, year, first_treat, all_of(outcome))
    df <- df %>% filter(!is.na(.data[[outcome]]))
    if (nrow(df) == 0) next

    model <- tryCatch(
      fixest::feols(
        as.formula(paste0(outcome, " ~ sunab(first_treat, year, ref.p = -1) | iso3 + year")),
        data = df,
        cluster = "iso3"
      ),
      error = function(e) NULL
    )
    if (is.null(model)) next

    coef_df <- broom::tidy(model)
    coef_df <- coef_df %>% filter(grepl("sunab", term))
    coef_df$event_time <- as.numeric(gsub(".*::", "", coef_df$term))

    results[[length(results) + 1]] <- tibble(
      anchor = "WTO",
      cohort = "all",
      outcome = outcome,
      event_time = coef_df$event_time,
      att = coef_df$estimate,
      se = coef_df$std.error,
      ci_low = coef_df$estimate - 1.96 * coef_df$std.error,
      ci_high = coef_df$estimate + 1.96 * coef_df$std.error,
      spec_id = "sunab"
    )
  }
}

if (length(results) > 0) {
  bind_rows(results) %>% write_csv(output_path)
} else {
  write_csv(tibble(), output_path)
}

if (length(diags) > 0) {
  bind_rows(diags) %>% write_csv(diag_path)
} else {
  write_csv(tibble(), diag_path)
}
