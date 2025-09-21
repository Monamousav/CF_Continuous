from RANN import run_model as run_rann
from simple_ANN import run_model as run_simpleann
from half_RANN import run_model as run_half_RANN
from half_RRF import run_model as run_halfrrf
from NO_RF import run_model as run_NO_RF
from Generate_train_test_splits import data_2nd_stage, evall_N_seq, device

if __name__ == "__main__":
    # === RANN ===
    #run_rann("RANN", 1, data_2nd_stage, evall_N_seq, device)
    #run_rann("RANN", 5, data_2nd_stage, evall_N_seq, device)
    #run_rann("RANN", 10, data_2nd_stage, evall_N_seq, device)

    # === Simple ANN ===
    #run_simpleann("SimpleANN", 1, data_2nd_stage, evall_N_seq, device)
    #run_simpleann("SimpleANN",2, data_2nd_stage, evall_N_seq, device)
    #run_simpleann("SimpleANN",3, data_2nd_stage, evall_N_seq, device)
    #run_simpleann("SimpleANN", 4, data_2nd_stage, evall_N_seq, device)
    #run_simpleann("SimpleANN", 5, data_2nd_stage, evall_N_seq, device)
    #run_simpleann("SimpleANN", 10, data_2nd_stage, evall_N_seq, device)
    run_simpleann("SimpleANN", 20, data_2nd_stage, evall_N_seq, device)


 # === half RANN ===
    #run_half_RANN("half_RANN", 1, data_2nd_stage, evall_N_seq, device)
    #run_half_RANN("half_RANN", 5, data_2nd_stage, evall_N_seq, device)
    #run_half_RANN("half_RANN", 10, data_2nd_stage, evall_N_seq, device)
    #run_half_RANN("half_RANN", 20, data_2nd_stage, evall_N_seq, device)


# === half RRF ===
    #run_halfrrf("HalfRRF", 1, data_2nd_stage, evall_N_seq, device)
    #run_halfrrf("HalfRRF", 5, data_2nd_stage, evall_N_seq, device)
    #run_halfrrf("HalfRRF", 10, data_2nd_stage, evall_N_seq, device)

   

# === NO_RF ===
    #run_NO_RF("NO_RF", 1, data_2nd_stage, evall_N_seq, device)
    #run_NO_RF("NO_RF", 5, data_2nd_stage, evall_N_seq, device)
    #run_NO_RF("NO_RF", 10, data_2nd_stage, evall_N_seq, device)


