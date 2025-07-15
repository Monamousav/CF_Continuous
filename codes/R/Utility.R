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
