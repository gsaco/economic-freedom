args <- commandArgs(trailingOnly = TRUE)
input_path <- args[1]
output_path <- args[2]
outcome_arg <- if (length(args) >= 3) args[3] else ""

ensure_pkg <- function(pkg, github = NULL) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  if (!requireNamespace(pkg, quietly = TRUE) && !is.null(github)) {
    if (!requireNamespace("remotes", quietly = TRUE)) {
      install.packages("remotes", repos = "https://cloud.r-project.org")
    }
    remotes::install_github(github)
  }
}

ensure_pkg("augsynth", github = "ebenmichael/augsynth")
ensure_pkg("synthdid", github = "synth-inference/synthdid")
ensure_pkg("dplyr")
ensure_pkg("readr")
ensure_pkg("tibble")

library(augsynth)
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
    mutate(
      treat_unit = if_else(iso3 %in% treated_ids, 1, 0),
      treat = if_else(iso3 %in% treated_ids & year >= t0, 1, 0)
    )

  for (outcome in outcomes) {
    if (!outcome %in% names(panel_sub)) next

    df <- panel_sub %>% select(iso3, year, treat_unit, all_of(outcome))
    df <- df %>% filter(!is.na(.data[[outcome]]))
    window_years <- sort(unique(df$year[df$year >= (t0 - 10) & df$year <= (t0 + 10)]))
    if (length(window_years) == 0) next
    df <- df %>% filter(year %in% window_years)
    years <- sort(unique(df$year))
    df <- df %>% group_by(iso3) %>% filter(n() == length(years)) %>% ungroup()
    if (nrow(df) == 0) next

    fit <- tryCatch(
      augsynth::augsynth(
        as.formula(paste0(outcome, " ~ treat_unit")),
        unit = "iso3",
        time = "year",
        data = df,
        t_int = t0,
        progfunc = "Ridge",
        scm = TRUE
      ),
      error = function(e) NULL
    )

    if (!is.null(fit)) {
      summ <- summary(fit)
      att <- as.numeric(summ$att)
      se <- as.numeric(summ$se)

      results[[length(results) + 1]] <- tibble::tibble(
        anchor = "EU",
        cohort = cohort,
        outcome = outcome,
        event_time = 0,
        att = att,
        se = se,
        ci_low = att - 1.96 * se,
        ci_high = att + 1.96 * se,
        spec_id = "augsynth"
      )
    } else {
      df_sc <- panel_sub %>% select(iso3, year, treat, all_of(outcome)) %>% filter(!is.na(.data[[outcome]]))
      df_sc <- df_sc %>% filter(year %in% window_years)
      df_sc <- df_sc %>% group_by(iso3) %>% filter(n() == length(years)) %>% ungroup()
      if (n_distinct(df_sc$iso3[df_sc$treat == 1]) == 0 || n_distinct(df_sc$iso3[df_sc$treat == 0]) == 0) {
        next
      }
      df_sc <- as.data.frame(df_sc)
      pm <- tryCatch(
        synthdid::panel.matrices(df_sc, unit = "iso3", time = "year", outcome = outcome, treatment = "treat"),
        error = function(e) NULL
      )
      if (is.null(pm) || pm$N0 < 2 || pm$T0 < 2) next

      est <- synthdid::sc_estimate(pm$Y, pm$N0, pm$T0)
      se <- as.numeric(sqrt(stats::vcov(est)))

      setup <- attr(est, "setup")
      weights <- attr(est, "weights")
      omega <- weights$omega
      y_mat <- setup$Y
      n0 <- setup$N0
      n1 <- nrow(y_mat) - n0
      y0 <- y_mat[1:n0, , drop = FALSE]
      y1 <- y_mat[(n0 + 1):(n0 + n1), , drop = FALSE]
      treated_mean <- colMeans(y1)
      synth_mean <- as.numeric(t(omega) %*% y0)
      diff <- treated_mean - synth_mean
      time_index <- seq_len(ncol(y_mat))
      event_time <- time_index - setup$T0 - 1

      results[[length(results) + 1]] <- tibble::tibble(
        anchor = "EU",
        cohort = cohort,
        outcome = outcome,
        event_time = event_time,
        att = diff,
        se = se,
        ci_low = diff - 1.96 * se,
        ci_high = diff + 1.96 * se,
        spec_id = "scm"
      )
    }
  }
}

if (length(results) > 0) {
  bind_rows(results) %>% write_csv(output_path)
} else {
  write_csv(tibble::tibble(), output_path)
}
