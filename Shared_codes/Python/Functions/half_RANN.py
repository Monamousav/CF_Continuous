import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# # === MODEL 3: half__RANN ===

def train_half_rann_model_from_splits(data_2nd_stage, evall_N_seq, split_csv_path, folder_name, device):
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.net(x)

     # Define root results directory and build the full output path
    base_results_dir = os.path.join(os.getcwd(), 'Results_fixed')
    output_folder = os.path.join(base_results_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)


    splits_df = pd.read_csv(split_csv_path)
    p_corn = 6.25 / 25.4
    p_N = 1 / 0.453592

    row = splits_df[splits_df['test_id'] == 1].iloc[0:1].copy()
    for _, row in row.iterrows():
    #for _, row in splits_df.iterrows():
        test_id = row['test_id']
        train_ids = row[[col for col in row.index if col.startswith('train_')]].values

        dataset = data_2nd_stage[data_2nd_stage['sim'].isin(train_ids)].reset_index(drop=True)
        dataset = dataset[['y_tilde', 'Nk', 'plateau', 'b0', 'N']]

        train_df = dataset.sample(frac=0.8, random_state=0)
        val_df = dataset.drop(train_df.index)

        X_train = train_df.drop('y_tilde', axis=1)
        y_train = train_df['y_tilde'].to_numpy().reshape(-1, 1)
        X_val = val_df.drop('y_tilde', axis=1)
        y_val = val_df['y_tilde'].to_numpy().reshape(-1, 1)

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_ld = DataLoader(train_ds, batch_size=512, shuffle=True)

        model = MyModel().to(device)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_val = float('inf')
        patience = 10
        counter = 0
        max_epochs = 500

        for epoch in range(max_epochs):
            model.train()
            for xb, yb in train_ld:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t)
                val_loss = criterion(val_preds, y_val_t).item()

            if val_loss < best_val:
                best_val = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stop at epoch {epoch} (sim {test_id})")
                    break

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t).cpu().numpy().flatten()

        pd.DataFrame({'pred': val_out, 'true': y_val.flatten()}) \
            .to_csv(os.path.join(output_folder, f'validation_{test_id}.csv'), index=False)

        # EONR estimation
        test_df = data_2nd_stage[data_2nd_stage['sim'] == test_id]
        features = test_df[['Nk', 'plateau', 'b0']].reset_index(drop=True)
        eval_seq = evall_N_seq[evall_N_seq['sim'] == test_id].reset_index(drop=True)
        Nseq = eval_seq['N'].to_numpy()
        estEONR = []

        for i in range(len(features)):
            base = features.iloc[[i]]
            repeated = pd.concat([base] * 100, ignore_index=True)
            full_feat = pd.concat([repeated, eval_seq[['N']]], axis=1)
            X_feat = torch.tensor(scaler.transform(full_feat), dtype=torch.float32).to(device)
            with torch.no_grad():
                y_hat = model(X_feat).cpu().numpy().reshape(-1)
            MP = y_hat * p_corn - Nseq * p_N
            estEONR.append(Nseq[np.argmax(MP)])

        pd.DataFrame({'pred': estEONR, 'true': test_df['opt_N'].to_numpy()}) \
            .to_csv(os.path.join(output_folder, f'EONR_{test_id}.csv'), index=False)


def run_model(model_type, n_fields, data_2nd_stage, evall_N_seq, device):
    split_csv_path = f'./Data/train_test_split/train_test_splits_{n_fields}fields.csv'
    folder_map = {
        ("half_RANN", 1): "half_RANN_outcome_one_field",
        ("half_RANN", 5): "half_RANN_outcome_five_fields",
        ("half_RANN", 10): "half_RANN_outcome_ten_fields"
    }
    folder_name = folder_map.get((model_type, n_fields))
    if folder_name is None:
        raise ValueError("Unsupported combination of model and field count.")

    if model_type == "half_RANN":
        train_half_rann_model_from_splits(data_2nd_stage, evall_N_seq, split_csv_path, folder_name, device)
