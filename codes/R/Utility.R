prepare_T_mat <- function(gam_formula, data) {
  gam_setup <- gam(gam_formula, data = data)
  T_var_name <- all.vars(gam_formula)[2]
  T_sp <- predict(gam_setup, data = data, type = "lpmatrix") %>%
    .[, -1] %>%
    data.table() %>%
    setnames(names(.), paste0("T_", 1:ncol(.)))
  return(list(gam_setup = gam_setup, T_sp = T_sp, T_var_name = T_var_name))
}




get_te <- function(trained_model, test_data, x_vars, id_var = "aunit_id") {
  X_test <- test_data[, ..x_vars] %>% as.matrix()
  te_hat <- trained_model$const_marginal_effect(X_test)
  return(list(te_hat = data.table(te_hat), id_data = test_data[, ..id_var]))
}

#______________________________________________
#---- If I do not save splines use it:-------- 
#______________________________________________
find_response_semi <- function(T_seq, T_info, te_info) {
  eval_T <- data.table(T = T_seq) %>%
    setnames("T", T_info$T_var_name) %>%
    predict(T_info$gam_setup, newdata = ., type = "lpmatrix") %>%
    .[, -1] %>%
    data.table()

  curv_data <- as.matrix(te_info$te_hat) %*% t(as.matrix(eval_T)) %>%
    data.table() %>%
    setnames(names(.), as.character(T_seq)) %>%
    .[, id := 1:.N] %>%
    melt(id.var = "id") %>%
    setnames(c("variable", "value"), c("T", "est")) %>%
    .[, T := as.numeric(as.character(T))]

  id_data <- te_info$id_data[, id := 1:.N]
  final_data <- id_data[curv_data, on = "id"][, id := NULL]
  return(final_data)
}

# otherwise use it:

# find_response_semi <- function(T_seq, T_vars, train_data, te_info) {
#   # T_vars: vector of spline column names, e.g. c("T_1","T_2","T_3")
#   # train_data: training data that already has those columns
#   # te_info: output from get_te()
#   
#   # extract spline basis matrix from training data
#   T_mat <- as.matrix(train_data[, ..T_vars])
#   
#   # create spline basis interpolation function for each column
#   interp_list <- lapply(1:ncol(T_mat), function(j) {
#     approxfun(train_data$N, T_mat[, j], rule = 2)
#   })
#   
#   # evaluate the basis at T_seq values
#   eval_T <- sapply(interp_list, function(f) f(T_seq))
#   colnames(eval_T) <- T_vars
#   
#   # estimate predicted yield responses
#   curv_data <- as.matrix(te_info$te_hat) %*% t(eval_T)
#   curv_data <- data.table(curv_data)
#   setnames(curv_data, as.character(T_seq))
#   
#   curv_data <- curv_data[, id := 1:.N] %>%
#     melt(id.var = "id", variable.name = "T", value.name = "est") %>%
#     .[, T := as.numeric(as.character(T))]
#   
#   id_data <- te_info$id_data[, id := 1:.N]
#   final_data <- id_data[curv_data, on = "id"][, id := NULL]
#   return(final_data)
# }


expand_grid_df <- function(data_1, data_2) {
  data_1_ex <-
    data_1[rep(1:nrow(data_1), each = nrow(data_2)), ] %>%
    data.table() %>%
    .[, rowid := 1:nrow(.)]
  data_2_ex <-
    data_2[rep(1:nrow(data_2), nrow(data_1)), ] %>%
    data.table() %>%
    .[, rowid := 1:nrow(.)]
  expanded_data <-
    data_1_ex[data_2_ex, on = "rowid"] %>%
    .[, rowid := NULL]
  if ("tbl" %in% class(data_1)) {
    expanded_data <- as_tibble(expanded_data)
  }
  if ("rowwise_df" %in% class(data_1)) {
    expanded_data <- rowwise(expanded_data)
  }
  return(expanded_data)
}
