args <- commandArgs(trailingOnly = TRUE)
input_path <- args[1]
output_path <- args[2]

ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

ensure_pkg("dplyr")
ensure_pkg("readr")
ensure_pkg("tibble")
ensure_pkg("gsynth")

library(dplyr)
library(readr)

if (!requireNamespace("gsynth", quietly = TRUE)) {
  readr::write_csv(tibble::tibble(), output_path)
  quit(save = "no", status = 0)
}

library(gsynth)

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

cohorts <- c("2004", "2007", "2013")

results <- list()

for (cohort in cohorts) {
  treated_ids <- panel %>% filter(eu_wave == cohort) %>% pull(iso3) %>% unique()
  if (length(treated_ids) == 0) next

  treat_years <- panel %>%
    filter(iso3 %in% treated_ids, EU_neg_share > 0) %>%
    group_by(iso3) %>%
    summarise(treat_year = min(year), .groups = "drop")
  if (nrow(treat_years) == 0) next
  t0 <- min(treat_years$treat_year, na.rm = TRUE)

  panel_sub <- panel %>%
    mutate(treat = if_else(iso3 %in% treated_ids & year >= t0, 1, 0))

  for (outcome in outcomes) {
    if (!outcome %in% names(panel_sub)) next

    df <- panel_sub %>% select(iso3, year, treat, all_of(outcome))
    df <- df %>% filter(!is.na(.data[[outcome]]))
    window_years <- sort(unique(df$year[df$year >= (t0 - 10) & df$year <= (t0 + 10)]))
    if (length(window_years) == 0) next
    df <- df %>% filter(year %in% window_years)
    years <- sort(unique(df$year))
    df <- df %>% group_by(iso3) %>% filter(n() == length(years)) %>% ungroup()
    if (n_distinct(df$iso3[df$treat == 1]) == 0 || n_distinct(df$iso3[df$treat == 0]) == 0) {
      next
    }
    if (nrow(df) == 0) next

    fit <- tryCatch(
      gsynth::gsynth(
        as.formula(paste0(outcome, " ~ treat")),
        data = df,
        index = c("iso3", "year"),
        force = "two-way",
        CV = FALSE,
        r = 0:1,
        se = TRUE,
        min.T0 = 2
      ),
      error = function(e) NULL
    )
    if (is.null(fit)) next

    att <- fit$att
    event_time <- seq_along(att)

    results[[length(results) + 1]] <- tibble::tibble(
      anchor = "EU",
      cohort = cohort,
      outcome = outcome,
      event_time = event_time,
      att = att,
      se = fit$att.se,
      ci_low = att - 1.96 * fit$att.se,
      ci_high = att + 1.96 * fit$att.se,
      spec_id = "gsc"
    )
  }
}

if (length(results) > 0) {
  bind_rows(results) %>% write_csv(output_path)
} else {
  write_csv(tibble::tibble(), output_path)
}
