# Requirements
suppressPackageStartupMessages({
  library(tidyverse)
  library(here)
})

# Folder label helper
fields_label <- function(n_fields) {
  if (n_fields == 1) "one_field"
  else if (n_fields == 5) "five_fields"
  else if (n_fields == 10) "ten_fields"
  else paste0(n_fields, "_fields")
}

# Build path to the CSV using here()
results_csv_path <- function(model, n_fields, test_id = 1,
                             base_dir = here("results", "yield_response_function_for_one_iteration")) {
  subdir <- paste0("YRF_", model, "_", fields_label(n_fields))
  file.path(base_dir, subdir, sprintf("yield_response_%s.csv", test_id))
}

# Main plotting function
# Example: plot_yield_response(model = "NO_ANN", n_fields = 5)
plot_yield_response <- function(model = "NO_ANN",
                                n_fields = 5,
                                test_id = 1,
                                n_ids = 5,
                                seed = 123,
                                fixed_scales = TRUE,
                                group_col = NULL) {
  csv_path <- results_csv_path(model, n_fields, test_id)
  
  if (!file.exists(csv_path)) {
    stop("File not found: ", csv_path, call. = FALSE)
  }
  
  df <- readr::read_csv(csv_path, show_col_types = FALSE)
  
  # x-axis: prefer N, else N_tilde
  x_col <- if ("N" %in% names(df)) "N" else if ("N_tilde" %in% names(df)) "N_tilde" else
    stop("Neither 'N' nor 'N_tilde' column found in: ", csv_path, call. = FALSE)
  
  # grouping column: user-specified or auto (aunit_id -> row_id -> cell_id -> field_id)
  if (is.null(group_col)) {
    candidates <- c("aunit_id", "row_id", "cell_id", "field_id")
    group_col <- candidates[candidates %in% names(df)][1]
    if (is.na(group_col)) stop("No suitable grouping column found in file.")
  } else if (!group_col %in% names(df)) {
    stop("Requested group_col '", group_col, "' not found in file.")
  }
  
  # pick IDs
  set.seed(seed)
  uniq_ids <- unique(df[[group_col]])
  if (!length(uniq_ids)) stop("No IDs found in ", group_col, call. = FALSE)
  pick <- sample(uniq_ids, size = min(n_ids, length(uniq_ids)))
  
  df_plot <- df %>%
    filter(.data[[group_col]] %in% pick) %>%
    arrange(.data[[group_col]], .data[[x_col]]) %>%
    drop_na(pred_yield, .data[[x_col]])
  
  # global y limits for consistency across facets
  if (fixed_scales) {
    y_min <- min(df_plot$pred_yield, na.rm = TRUE)
    y_max <- max(df_plot$pred_yield, na.rm = TRUE)
  }
  
  title_txt <- sprintf(
    "Yield response (%s: %s) — %d random %s",
    model, fields_label(n_fields), length(pick),
    if (group_col == "aunit_id") "aunit_id" else group_col
  )
  
  p <- ggplot(df_plot, aes(x = .data[[x_col]], y = pred_yield)) +
    geom_line(linewidth = 0.9) +
    facet_wrap(vars(.data[[group_col]])) +
    labs(
      title = title_txt,
      x = if (x_col == "N") "Nitrogen (N)" else "Nitrogen (N_tilde)",
      y = "Predicted yield"
    ) +
    theme_minimal()
  
  if (fixed_scales) {
    p <- p + scale_y_continuous(limits = c(y_min, y_max))
  }
  
  #print(p)
  #invisible(list(plot = p, file = csv_path, ids = pick, x_col = x_col, group_col = group_col))
  p
}
