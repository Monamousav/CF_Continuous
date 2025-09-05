# # Requirements
# suppressPackageStartupMessages({
#   library(tidyverse)
#   library(here)
# })
# 
# # Folder label helper
# fields_label <- function(n_fields) {
#   if (n_fields == 1) "one_field"
#   else if (n_fields == 5) "five_fields"
#   else if (n_fields == 10) "ten_fields"
#   else paste0(n_fields, "_fields")
# }
# 
# # Build path to the CSV using here()
# results_csv_path <- function(model, n_fields, test_id = 1,
#                              base_dir = here("results", "yield_response_function_for_one_iteration")) {
#   subdir <- paste0("YRF_", model, "_", fields_label(n_fields))
#   file.path(base_dir, subdir, sprintf("yield_response_%s.csv", test_id))
# }
# 
# # Main plotting function
# # Example: plot_yield_response(model = "NO_ANN", n_fields = 5)
# plot_yield_response <- function(model = "NO_ANN",
#                                 n_fields = 5,
#                                 test_id = 1,
#                                 n_ids = 5,
#                                 seed = 123,
#                                 fixed_scales = TRUE,
#                                 group_col = NULL) {
#   csv_path <- results_csv_path(model, n_fields, test_id)
#   
#   if (!file.exists(csv_path)) {
#     stop("File not found: ", csv_path, call. = FALSE)
#   }
#   
#   df <- readr::read_csv(csv_path, show_col_types = FALSE)
#   
#   # x-axis: prefer N, else N_tilde
#   x_col <- if ("N" %in% names(df)) "N" else if ("N_tilde" %in% names(df)) "N_tilde" else
#     stop("Neither 'N' nor 'N_tilde' column found in: ", csv_path, call. = FALSE)
#   
#   # grouping column: user-specified or auto (aunit_id -> row_id -> cell_id -> field_id)
#   if (is.null(group_col)) {
#     candidates <- c("aunit_id", "row_id", "cell_id", "field_id")
#     group_col <- candidates[candidates %in% names(df)][1]
#     if (is.na(group_col)) stop("No suitable grouping column found in file.")
#   } else if (!group_col %in% names(df)) {
#     stop("Requested group_col '", group_col, "' not found in file.")
#   }
#   
#   # pick IDs
#   set.seed(seed)
#   uniq_ids <- unique(df[[group_col]])
#   if (!length(uniq_ids)) stop("No IDs found in ", group_col, call. = FALSE)
#   pick <- sample(uniq_ids, size = min(n_ids, length(uniq_ids)))
#   
#   df_plot <- df %>%
#     filter(.data[[group_col]] %in% pick) %>%
#     arrange(.data[[group_col]], .data[[x_col]]) %>%
#     drop_na(pred_yield, .data[[x_col]])
#   
#   # global y limits for consistency across facets
#   if (fixed_scales) {
#     y_min <- min(df_plot$pred_yield, na.rm = TRUE)
#     y_max <- max(df_plot$pred_yield, na.rm = TRUE)
#   }
#   
#   title_txt <- sprintf(
#     "Yield response (%s: %s) — %d random %s",
#     model, fields_label(n_fields), length(pick),
#     if (group_col == "aunit_id") "aunit_id" else group_col
#   )
#   
#   p <- ggplot(df_plot, aes(x = .data[[x_col]], y = pred_yield)) +
#     geom_line(linewidth = 0.9) +
#     facet_wrap(vars(.data[[group_col]])) +
#     labs(
#       title = title_txt,
#       x = if (x_col == "N") "Nitrogen (N)" else "Nitrogen (N_tilde)",
#       y = "Predicted yield"
#     ) +
#     theme_minimal()
#   
#   if (fixed_scales) {
#     p <- p + scale_y_continuous(limits = c(y_min, y_max))
#   }
#   
#   #print(p)
#   #invisible(list(plot = p, file = csv_path, ids = pick, x_col = x_col, group_col = group_col))
#   p
# }




#############$%^&&&&&&&&&&&&&&&&&&& add true yrf and mean match as well


suppressPackageStartupMessages({
  library(here)
  library(readr)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(magrittr)  # %>%
})

#  folder label
fields_label <- function(n_fields) {
  if (n_fields == 1) "one_field"
  else if (n_fields == 5) "five_fields"
  else if (n_fields == 10) "ten_fields"
  else paste0(n_fields, "_fields")
}

# path to the CSV
results_csv_path <- function(model, n_fields, test_id = 1,
                             base_dir = here("results", "yield_response_function_for_one_iteration")) {
  subdir <- paste0("YRF_", model, "_", fields_label(n_fields))
  file.path(base_dir, subdir, sprintf("yield_response_%s.csv", test_id))
}

# helper so limits don't collapse if min==max
.pad_range <- function(lo, hi, frac = 0.02) {
  if (!is.finite(lo) || !is.finite(hi)) return(c(NA_real_, NA_real_))
  if (lo == hi) {
    d <- if (hi == 0) 1 else abs(hi) * frac
    return(c(lo - d, hi + d))
  }
  rng <- hi - lo
  c(lo - frac*rng, hi + frac*rng)
}

# ----------  compute global limits across multiple models ----------
compute_global_limits <- function(models,
                                  n_fields,
                                  test_id = 1,
                                  base_dir = here("results", "yield_response_function_for_one_iteration")) {
  xmins <- c(); xmaxs <- c(); ymins <- c(); ymaxs <- c()
  
  for (m in models) {
    f <- results_csv_path(m, n_fields, test_id, base_dir = base_dir)
    if (!file.exists(f)) next
    df <- suppressMessages(readr::read_csv(f, show_col_types = FALSE))
    
    # choose x column (N preferred, else N_tilde)
    x_col <- if ("N" %in% names(df)) "N" else if ("N_tilde" %in% names(df)) "N_tilde" else NULL
    if (!is.null(x_col)) {
      xmins <- c(xmins, suppressWarnings(min(df[[x_col]], na.rm = TRUE)))
      xmaxs <- c(xmaxs, suppressWarnings(max(df[[x_col]], na.rm = TRUE)))
    }
    
    # y from predicted (prefer pred_yield_mm if present) and true
    y_pred_col <- if ("pred_yield_mm" %in% names(df)) "pred_yield_mm" else "pred_yield"
    y_vec <- df[[y_pred_col]]
    if ("true_yield" %in% names(df)) y_vec <- c(y_vec, df$true_yield)
    
    ymins <- c(ymins, suppressWarnings(min(y_vec, na.rm = TRUE)))
    ymaxs <- c(ymaxs, suppressWarnings(max(y_vec, na.rm = TRUE)))
  }
  
  x_lo <- suppressWarnings(min(xmins, na.rm = TRUE))
  x_hi <- suppressWarnings(max(xmaxs, na.rm = TRUE))
  y_lo <- suppressWarnings(min(ymins, na.rm = TRUE))
  y_hi <- suppressWarnings(max(ymaxs, na.rm = TRUE))
  
  list(
    x = .pad_range(x_lo, x_hi),
    y = .pad_range(y_lo, y_hi)
  )
}

# ---------- Plot ( Predicted vs True) with  global limits ----------
plot_yield_response <- function(model = "NO_ANN",
                                n_fields = 5,
                                test_id = 1,
                                n_ids = 1,
                                seed = 123,
                                group_col = NULL,
                                specific_ids = NULL,
                                # EITHER pass explicit limits...
                                x_limits = NULL,
                                y_limits = NULL,
                                # ...OR give a set of models to compute global limits across:
                                global_models = NULL,
                                base_dir = here("results", "yield_response_function_for_one_iteration")) {
  
  csv_path <- results_csv_path(model, n_fields, test_id, base_dir = base_dir)
  if (!file.exists(csv_path)) stop("File not found: ", csv_path, call. = FALSE)
  
  df <- readr::read_csv(csv_path, show_col_types = FALSE)
  
  #  true_yield for overlay; 
  if (!"true_yield" %in% names(df)) {
    stop("The file does not contain 'true_yield'. Re-run generation to include it: ", csv_path, call. = FALSE)
  }
  
  # choose y-pred column (prefer mean-matched if present)
  y_pred_col <- if ("pred_yield_mm" %in% names(df)) "pred_yield_mm" else "pred_yield"
  pred_label <- if (y_pred_col == "pred_yield_mm") "Predicted (mm)" else "Predicted"
  
  #  global limits across models, compute them now
  if (!is.null(global_models) && (is.null(x_limits) || is.null(y_limits))) {
    lims <- compute_global_limits(models = global_models,
                                  n_fields = n_fields,
                                  test_id = test_id,
                                  base_dir = base_dir)
    if (is.null(x_limits)) x_limits <- lims$x
    if (is.null(y_limits)) y_limits <- lims$y
  }
  
  # x-axis: prefer N, else N_tilde
  x_col <- if ("N" %in% names(df)) "N" else if ("N_tilde" %in% names(df)) "N_tilde" else
    stop("Neither 'N' nor 'N_tilde' column found in: ", csv_path, call. = FALSE)
  
  # grouping column: prefer aunit_id → row_id → cell_id → field_id unless user supplies one
  if (is.null(group_col)) {
    candidates <- c("aunit_id", "row_id", "cell_id", "field_id")
    group_col <- candidates[candidates %in% names(df)][1]
    if (is.na(group_col)) stop("No suitable grouping column found in file.")
  } else if (!group_col %in% names(df)) {
    stop("Requested group_col '", group_col, "' not found in file.")
  }
  
  #  which IDs to plot (default 5)
  if (!is.null(specific_ids)) {
    pick <- intersect(specific_ids, unique(df[[group_col]]))
    if (!length(pick)) stop("None of the specified 'specific_ids' were found in the data.")
  } else {
    set.seed(seed)
    uniq_ids <- unique(df[[group_col]])
    if (!length(uniq_ids)) stop("No IDs found in ", group_col, call. = FALSE)
    pick <- sample(uniq_ids, size = min(n_ids, length(uniq_ids)))
  }
  
  df_plot <- df %>%
    filter(.data[[group_col]] %in% pick) %>%
    arrange(.data[[group_col]], .data[[x_col]]) %>%
    drop_na(.data[[y_pred_col]], true_yield, .data[[x_col]])
  
  # Long format for overlayed lines (Predicted vs True)
  df_long <- df_plot %>%
    select(all_of(c(group_col, x_col, y_pred_col, "true_yield"))) %>%
    rename(predicted_col = !!y_pred_col) %>%
    pivot_longer(c(predicted_col, true_yield),
                 names_to = "curve", values_to = "yield") %>%
    mutate(curve = recode(curve,
                          predicted_col = pred_label,
                          true_yield   = "True"))
  
  title_txt <- sprintf(
    "Yield response (%s: %s) — %d %s ( %s vs True )",
    model, fields_label(n_fields), length(pick),
    if (group_col == "aunit_id") "aunit_id" else group_col,
    pred_label
  )
  
  p <- ggplot(df_long, aes(x = .data[[x_col]], y = yield, color = curve)) +
    geom_line(linewidth = 1) +
    facet_wrap(vars(.data[[group_col]])) +
    labs(
      title = title_txt,
      x = if (x_col == "N") "Nitrogen (N)" else "Nitrogen (N_tilde)",
      y = "Yield",
      color = NULL
    ) +
    theme_minimal() +
    theme(legend.position = "top")
  
  # lock axes if provided/derived
  if (!is.null(x_limits)) p <- p + scale_x_continuous(limits = x_limits)
  if (!is.null(y_limits)) p <- p + scale_y_continuous(limits = y_limits)
  
  p
}




##### Use N_tilde for DO


# suppressPackageStartupMessages({
#   library(here)
#   library(readr)
#   library(ggplot2)
#   library(dplyr)
#   library(tidyr)
#   library(magrittr)  # %>%
# })
# 
# # Helper: folder label
# fields_label <- function(n_fields) {
#   if (n_fields == 1) "one_field"
#   else if (n_fields == 5) "five_fields"
#   else if (n_fields == 10) "ten_fields"
#   else paste0(n_fields, "_fields")
# }
# 
# # Build path to the CSV using here()
# results_csv_path <- function(model, n_fields, test_id = 1,
#                              base_dir = here("results", "yield_response_function_for_one_iteration")) {
#   subdir <- paste0("YRF_", model, "_", fields_label(n_fields))
#   file.path(base_dir, subdir, sprintf("yield_response_%s.csv", test_id))
# }
# 
# # Choose x column by model: DO -> N_tilde, others -> N
# .choose_x_by_model <- function(model, cols) {
#   want <- if (grepl("^DO", model, ignore.case = TRUE)) "N_tilde" else "N"
#   if (!(want %in% cols)) {
#     stop(
#       sprintf("Expected x column '%s' for model '%s' not found in file. Columns present: %s",
#               want, model, paste(cols, collapse = ", ")),
#       call. = FALSE
#     )
#   }
#   want
# }
# 
# # small padding helper
# .pad_range <- function(lo, hi, frac = 0.02) {
#   if (!is.finite(lo) || !is.finite(hi)) return(c(NA_real_, NA_real_))
#   if (lo == hi) {
#     d <- if (hi == 0) 1 else abs(hi) * frac
#     return(c(lo - d, hi + d))
#   }
#   rng <- hi - lo
#   c(lo - frac*rng, hi + frac*rng)
# }
# 
# # ---------- Global limits across multiple models (respects DO rule) ----------
# compute_global_limits <- function(models,
#                                   n_fields,
#                                   test_id = 1,
#                                   base_dir = here("results", "yield_response_function_for_one_iteration")) {
#   xmins <- c(); xmaxs <- c(); ymins <- c(); ymaxs <- c()
#   
#   for (m in models) {
#     f <- results_csv_path(m, n_fields, test_id, base_dir = base_dir)
#     if (!file.exists(f)) next
#     df <- suppressMessages(readr::read_csv(f, show_col_types = FALSE))
#     
#     # choose x by model (N for non-DO, N_tilde for DO)
#     x_col <- .choose_x_by_model(m, names(df))
#     xmins <- c(xmins, suppressWarnings(min(df[[x_col]], na.rm = TRUE)))
#     xmaxs <- c(xmaxs, suppressWarnings(max(df[[x_col]], na.rm = TRUE)))
#     
#     # y from both predicted and true (true_yield may be missing)
#     y_vec <- df$pred_yield
#     if ("true_yield" %in% names(df)) y_vec <- c(y_vec, df$true_yield)
#     ymins <- c(ymins, suppressWarnings(min(y_vec, na.rm = TRUE)))
#     ymaxs <- c(ymaxs, suppressWarnings(max(y_vec, na.rm = TRUE)))
#   }
#   
#   x_lo <- suppressWarnings(min(xmins, na.rm = TRUE))
#   x_hi <- suppressWarnings(max(xmaxs, na.rm = TRUE))
#   y_lo <- suppressWarnings(min(ymins, na.rm = TRUE))
#   y_hi <- suppressWarnings(max(ymaxs, na.rm = TRUE))
#   
#   list(
#     x = .pad_range(x_lo, x_hi),
#     y = .pad_range(y_lo, y_hi)
#   )
# }
# 
# # ---------- Plot (overlay Predicted vs True) with optional global limits ----------
# plot_yield_response <- function(model = "NO_ANN",
#                                 n_fields = 5,
#                                 test_id = 1,
#                                 n_ids = 1,
#                                 seed = 123,
#                                 group_col = NULL,
#                                 specific_ids = NULL,
#                                 x_limits = NULL,
#                                 y_limits = NULL,
#                                 global_models = NULL,
#                                 base_dir = here("results", "yield_response_function_for_one_iteration")) {
#   
#   csv_path <- results_csv_path(model, n_fields, test_id, base_dir = base_dir)
#   if (!file.exists(csv_path)) stop("File not found: ", csv_path, call. = FALSE)
#   
#   df <- readr::read_csv(csv_path, show_col_types = FALSE)
#   
#   if (!"true_yield" %in% names(df)) {
#     stop("The file does not contain 'true_yield'. Re-run generation to include it: ", csv_path, call. = FALSE)
#   }
#   
#   # If user asked for global limits across models, compute them first
#   if (!is.null(global_models) && (is.null(x_limits) || is.null(y_limits))) {
#     lims <- compute_global_limits(models = global_models,
#                                   n_fields = n_fields,
#                                   test_id = test_id,
#                                   base_dir = base_dir)
#     if (is.null(x_limits)) x_limits <- lims$x
#     if (is.null(y_limits)) y_limits <- lims$y
#   }
#   
#   # ----- X column selection per your rule -----
#   x_col <- .choose_x_by_model(model, names(df))
#   
#   # grouping column: prefer aunit_id → row_id → cell_id → field_id unless user supplies one
#   if (is.null(group_col)) {
#     candidates <- c("aunit_id", "row_id", "cell_id", "field_id")
#     group_col <- candidates[candidates %in% names(df)][1]
#     if (is.na(group_col)) stop("No suitable grouping column found in file.")
#   } else if (!group_col %in% names(df)) {
#     stop("Requested group_col '", group_col, "' not found in file.")
#   }
#   
#   # choose which IDs to plot
#   if (!is.null(specific_ids)) {
#     pick <- intersect(specific_ids, unique(df[[group_col]]))
#     if (!length(pick)) stop("None of the specified 'specific_ids' were found in the data.")
#   } else {
#     set.seed(seed)
#     uniq_ids <- unique(df[[group_col]])
#     if (!length(uniq_ids)) stop("No IDs found in ", group_col, call. = FALSE)
#     pick <- sample(uniq_ids, size = min(n_ids, length(uniq_ids)))
#   }
#   
#   df_plot <- df %>%
#     filter(.data[[group_col]] %in% pick) %>%
#     arrange(.data[[group_col]], .data[[x_col]]) %>%
#     drop_na(pred_yield, true_yield, .data[[x_col]])
#   
#   # Long format for overlayed lines
#   df_long <- df_plot %>%
#     select(all_of(c(group_col, x_col, "pred_yield", "true_yield"))) %>%
#     pivot_longer(c(pred_yield, true_yield),
#                  names_to = "curve", values_to = "yield") %>%
#     mutate(curve = recode(curve,
#                           pred_yield = "Predicted",
#                           true_yield = "True"))
#   
#   title_txt <- sprintf(
#     "Yield response (%s: %s) — %d %s (overlay: Predicted vs True)",
#     model, fields_label(n_fields), length(pick),
#     if (group_col == "aunit_id") "aunit_id" else group_col
#   )
#   
#   p <- ggplot(df_long, aes(x = .data[[x_col]], y = yield, color = curve)) +
#     geom_line(linewidth = 1) +
#     facet_wrap(vars(.data[[group_col]])) +
#     labs(
#       title = title_txt,
#       x = if (x_col == "N") "Nitrogen (N)" else "Nitrogen (N_tilde)",
#       y = "Yield",
#       color = NULL
#     ) +
#     theme_minimal() +
#     theme(legend.position = "top")
#   
#   if (!is.null(x_limits)) p <- p + scale_x_continuous(limits = x_limits)
#   if (!is.null(y_limits)) p <- p + scale_y_continuous(limits = y_limits)
#   
#   p
# }
# 
