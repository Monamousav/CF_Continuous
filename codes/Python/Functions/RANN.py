# %% [markdown]

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# # === MODEL 1: RANN ===

# %%
# === MODEL 1: RANN ===
def train_rann_model_from_splits(data_2nd_stage, evall_N_seq, split_csv_path, folder_name, device):
    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.branch1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
            self.branch2 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
            self.branch3 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, inp):
            x = inp[:, :3]
            T = inp[:, 3:]
            return self.branch1(x)*T[:,0:1] + self.branch2(x)*T[:,1:2] + self.branch3(x)*T[:,2:3]

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
        dataset = dataset[['y_tilde', 'Nk', 'plateau', 'b0', 'T_1_tilde', 'T_2_tilde', 'T_3_tilde']]
        train_dataset = dataset.sample(frac=0.8, random_state=0)
        val_dataset = dataset.drop(train_dataset.index)

        y_train = train_dataset.pop('y_tilde')
        y_val = val_dataset.pop('y_tilde')

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_dataset)
        X_val = scaler.transform(val_dataset)

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)

        model = MyModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.L1Loss()

        best_val_loss = float('inf')
        patience = 10
        counter = 0
        for epoch in range(500):
            model.train()
            for i in range(0, len(X_train), 512):
                xb = X_train[i:i+512]
                yb = y_train[i:i+512]
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), y_val).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        with torch.no_grad():
            val_preds = model(X_val).cpu().numpy().flatten()
        pd.DataFrame({'pred': val_preds, 'true': y_val.cpu().numpy().flatten()}).to_csv(
            os.path.join(output_folder, f'validation_{test_id}.csv'), index=False)

        # === EONR estimation ===
        test_df = data_2nd_stage[data_2nd_stage['sim'] == test_id]
        eval_seq = evall_N_seq[evall_N_seq['sim'] == test_id].reset_index(drop=True)
        Nseq = eval_seq['N'].values
        features = test_df[['Nk', 'plateau', 'b0']]
        estEONR = []
        for i in range(len(features)):
            base = features.iloc[[i]]
            repeated = pd.concat([base]*100, ignore_index=True)
            full_feat = pd.concat([repeated, eval_seq[['T_1', 'T_2', 'T_3']]], axis=1)
            full_feat.columns = ['Nk', 'plateau', 'b0', 'T_1_tilde', 'T_2_tilde', 'T_3_tilde']
            X_test = torch.tensor(scaler.transform(full_feat), dtype=torch.float32).to(device)
            with torch.no_grad():
                pred = model(X_test).cpu().numpy().flatten()
            MP = pred * p_corn - Nseq * p_N
            estEONR.append(Nseq[np.argmax(MP)])

        pd.DataFrame({'pred': estEONR, 'true': test_df['opt_N'].values}).to_csv(
            os.path.join(output_folder, f'EONR_{test_id}.csv'), index=False)


# === MODEL DISPATCHER ===
def run_model(model_type, n_fields, data_2nd_stage, evall_N_seq, device):
    split_csv_path = f'./data/train_test_split/train_test_splits_{n_fields}fields.csv'
    folder_map = {
        ("RANN", 1): "RANN_outcome_one_field",
        ("RANN", 5): "RANN_outcome_five_fields",
        ("RANN", 10): "RANN_outcome_ten_fields"
    }
    folder_name = folder_map.get((model_type, n_fields))
    if folder_name is None:
        raise ValueError("Unsupported combination of model and field count.")

    if model_type == "RANN":
        train_rann_model_from_splits(data_2nd_stage, evall_N_seq, split_csv_path, folder_name, device)


# %%
