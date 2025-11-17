from RANN import run_model as run_rann
from simple_ANN import run_model as run_simpleann
from half_RANN import run_model as run_half_RANN
from half_RRF import run_model as run_halfrrf
from NO_RF import run_model as run_NO_RF
from CF_python import run_model as run_CF_python
from CF_savedT import run_model as run_CF_savedT
from CF_spline import run_model as run_CF_spline
from CF_test import run_model as run_test
from Generate_train_test_splits import data_2nd_stage, evall_N_seq, device

if __name__ == "__main__":
    # === RANN ===
    #run_rann("RANN", 1, data_2nd_stage, evall_N_seq, device)
    #run_rann("RANN", 3, data_2nd_stage, evall_N_seq, device)
    #run_rann("RANN", 5, data_2nd_stage, evall_N_seq, device)
    #run_rann("RANN", 10, data_2nd_stage, evall_N_seq, device)

    # === Simple ANN ===
    #run_simpleann("SimpleANN", 1, data_2nd_stage, evall_N_seq, device)
    #run_simpleann("SimpleANN",2, data_2nd_stage, evall_N_seq, device)
    #run_simpleann("SimpleANN",3, data_2nd_stage, evall_N_seq, device)
    #run_simpleann("SimpleANN", 4, data_2nd_stage, evall_N_seq, device)
    #run_simpleann("SimpleANN", 5, data_2nd_stage, evall_N_seq, device)
    #run_simpleann("SimpleANN", 10, data_2nd_stage, evall_N_seq, device)
    #run_simpleann("SimpleANN", 20, data_2nd_stage, evall_N_seq, device)


 # === half RANN ===
    #run_half_RANN("half_RANN", 1, data_2nd_stage, evall_N_seq, device)
    #run_half_RANN("half_RANN", 3, data_2nd_stage, evall_N_seq, device)
    #run_half_RANN("half_RANN", 5, data_2nd_stage, evall_N_seq, device)
    #run_half_RANN("half_RANN", 10, data_2nd_stage, evall_N_seq, device)
    #run_half_RANN("half_RANN", 20, data_2nd_stage, evall_N_seq, device)


# === half RRF ===
    #run_halfrrf("HalfRRF", 1, data_2nd_stage, evall_N_seq, device)
    #run_halfrrf("HalfRRF", 3, data_2nd_stage, evall_N_seq, device)
    #run_halfrrf("HalfRRF", 5, data_2nd_stage, evall_N_seq, device)
    #run_halfrrf("HalfRRF", 10, data_2nd_stage, evall_N_seq, device)
    #run_halfrrf("HalfRRF", 20, data_2nd_stage, evall_N_seq, device)

   

# === NO_RF ===
    #run_NO_RF("NO_RF", 1, data_2nd_stage, evall_N_seq, device)
    #run_NO_RF("NO_RF", 3, data_2nd_stage, evall_N_seq, device)
    #run_NO_RF("NO_RF", 5, data_2nd_stage, evall_N_seq, device)
    #run_NO_RF("NO_RF", 10, data_2nd_stage, evall_N_seq, device)
    #run_NO_RF("NO_RF", 20, data_2nd_stage, evall_N_seq, device)


# === CF ===   
    #run_CF_python("CF_python", 1, data_2nd_stage, evall_N_seq)
    #run_CF_python("CF_python", 3, data_2nd_stage, evall_N_seq)
    #run_CF_python("CF_python", 5, data_2nd_stage, evall_N_seq)
    #run_CF_python("CF_python", 10, data_2nd_stage, evall_N_seq)


 
# === CF using saved spline columns (T_1, T_2, T_3, …) ===
    #run_CF_savedT("CF_savedT", 1, data_2nd_stage, evall_N_seq)
    #run_CF_savedT("CF_savedT", 3, data_2nd_stage, evall_N_seq)
    #run_CF_savedT("CF_savedT", 5, data_2nd_stage, evall_N_seq)
   #run_CF_savedT("CF_savedT", 10, data_2nd_stage, evall_N_seq)
    #run_CF_savedT("CF_savedT", 20, data_2nd_stage, evall_N_seq)


# === CF spline ===
   #run_CF_spline("CF_spline", 1, data_2nd_stage, evall_N_seq, device)
   #run_CF_spline("CF_spline", 3, data_2nd_stage, evall_N_seq, device)
   #run_CF_spline("CF_spline", 5, data_2nd_stage, evall_N_seq, device)
   #run_CF_spline("CF_spline", 10, data_2nd_stage, evall_N_seq, device)
   #run_CF_spline("CF_spline", 20, data_2nd_stage, evall_N_seq, device)

  # === TEST (pygam-based spline computation, mgcv equivalent) ===
   #run_test("test", 1, data_2nd_stage, evall_N_seq, device)
   # run_test("test", 3, data_2nd_stage, evall_N_seq, device)
   # run_test("test", 5, data_2nd_stage, evall_N_seq, device)
  # run_test("test", 10, data_2nd_stage, evall_N_seq, device)
  run_test("test", 20, data_2nd_stage, evall_N_seq, device)
