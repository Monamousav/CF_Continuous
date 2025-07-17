import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, GridSearchCV

# === MODEL 4: Half-RRF ===

def train_half_rf_model_from_splits(data_2nd_stage, evall_N_seq, split_csv_path, folder_name):
     # Define root results directory and build the full output path
    base_results_dir = os.path.join(os.getcwd(), 'results')
    output_folder = os.path.join(base_results_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    splits_df = pd.read_csv(split_csv_path)
    p_corn = 6.25 / 25.4
    p_N = 1 / 0.453592

    #row = splits_df[splits_df['test_id'] == 1].iloc[0:1].copy()
    #for _, row in row.iterrows():
    for _, row in splits_df.iterrows():    
        test_id = row['test_id']
        train_ids = row[[col for col in row.index if col.startswith('train_')]].values

        dataset = data_2nd_stage[data_2nd_stage['sim'].isin(train_ids)].reset_index(drop=True)
        dataset = dataset[['y_tilde', 'Nk', 'plateau', 'b0', 'N']]

        train_df = dataset.sample(frac=0.8, random_state=0)
        val_df = dataset.drop(train_df.index)

        X_train = train_df.drop('y_tilde', axis=1)
        y_train = train_df['y_tilde']
        X_val = val_df.drop('y_tilde', axis=1)
        y_val = val_df['y_tilde']

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        param_grid = {
            'max_depth': [3, 5],
            'n_estimators': [50, 250, 500, 1000],
            'max_features': [1, 2, 3]
        }
        rf = RandomForestRegressor(random_state=777)
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=777)
        grid = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
        grid.fit(X_train_scaled, y_train)
        model = grid.best_estimator_

        val_preds = model.predict(X_val_scaled)
        pd.DataFrame({'pred': val_preds, 'true': y_val.values}).to_csv(
            os.path.join(output_folder, f'validation_{test_id}.csv'), index=False
        )

        # EONR
        test_df = data_2nd_stage[data_2nd_stage['sim'] == test_id]
        features = test_df[['Nk', 'plateau', 'b0']].reset_index(drop=True)
        eval_seq = evall_N_seq[evall_N_seq['sim'] == test_id].reset_index(drop=True)
        Nseq = eval_seq['N'].to_numpy()

        estEONR = []
        for i in range(len(features)):
            base = features.iloc[[i]]
            repeated = pd.concat([base] * 100, ignore_index=True)
            full_feat = pd.concat([repeated, eval_seq[['N']]], axis=1)
            X_feat = scaler.transform(full_feat)
            preds = model.predict(X_feat)
            MP = preds * p_corn - Nseq * p_N
            estEONR.append(Nseq[np.argmax(MP)])

        pd.DataFrame({'pred': estEONR, 'true': test_df['opt_N'].to_numpy()}).to_csv(
            os.path.join(output_folder, f'EONR_{test_id}.csv'), index=False
        )

# === Wrap ===
def run_model(model_type, n_fields, data_2nd_stage, evall_N_seq, device):  # <- added device here
    split_csv_path = f'./data/train_test_split/train_test_splits_{n_fields}fields.csv'
    folder_map = {
        ("HalfRRF", 1): "HalfRRF_one_field",
        ("HalfRRF", 5): "HalfRRF_five_fields",
        ("HalfRRF", 10): "HalfRRF_ten_fields"
    }
    folder_name = folder_map.get((model_type, n_fields))
    if folder_name is None:
        raise ValueError("Unsupported model-type and field combination")

    if model_type == "HalfRRF":
        train_half_rf_model_from_splits(data_2nd_stage, evall_N_seq, split_csv_path, folder_name)
