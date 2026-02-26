"""
Core logic for process unlearning pipeline.
Extracted from the notebook - use target_process parameter to choose which process to unlearn.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_dataset(time_steps=1500):
    processes = [f"P{i}" for i in range(10)]
    data = []
    for process in processes:
        prev_memory = np.random.uniform(50, 150)
        regime = 0
        for t in range(time_steps):
            if t in [500, 1000]:
                regime = 1 - regime
            noise = np.random.normal(0, 3)
            if process == "P0":
                memory = prev_memory + 0.6 + noise * 0.3
            elif process == "P1":
                spike = np.random.pareto(2) * 30 if np.random.rand() < 0.15 else 0
                memory = prev_memory + spike + noise
            elif process == "P2":
                memory = prev_memory + np.random.normal(0, 8)
            elif process == "P3":
                memory = 120 + 45 * np.sin(t / 10) + noise
            elif process == "P4":
                memory = prev_memory * (1.004 if regime == 0 else 0.996)
            elif process == "P5":
                memory = 80 + 40 * np.log(t + 1)
            elif process == "P6":
                memory = prev_memory - 0.7 + noise
            elif process == "P7":
                memory = prev_memory + (25 if regime else -20) + noise
            elif process == "P8":
                memory = 60 + (t % 200) * 1.2
            elif process == "P9":
                r, x = 3.9, np.random.rand()
                chaotic = r * x * (1 - x)
                memory = prev_memory + chaotic * 15
            cpu = np.random.uniform(5, 95)
            data.append([process, t, prev_memory, cpu, memory])
            prev_memory = memory
    return pd.DataFrame(
        data,
        columns=["process_id", "time_step", "previous_memory", "cpu_usage", "memory_usage_next"],
    )


class MemoryDataset(Dataset):
    def __init__(self, df, seq_len=15):
        self.samples = []
        self.seq_len = seq_len
        for pid in df["process_id"].unique():
            proc_df = df[df["process_id"] == pid].sort_values("time_step")
            for i in range(len(proc_df) - seq_len):
                window = proc_df.iloc[i : i + seq_len]
                target = proc_df.iloc[i + seq_len]["memory_usage_next"]
                self.samples.append((window, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, target = self.samples[idx]
        proc = torch.LongTensor(window["process_encoded"].values)
        feat = torch.FloatTensor(window[["previous_memory", "cpu_usage"]].values)
        y = torch.FloatTensor([target])
        return proc, feat, y


class DeepLSTM(nn.Module):
    def __init__(self, num_processes):
        super().__init__()
        self.embedding = nn.Embedding(num_processes, 64)
        self.lstm = nn.LSTM(
            input_size=64 + 2,
            hidden_size=128,
            num_layers=3,
            dropout=0.3,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, proc, feat):
        emb = self.embedding(proc)
        x = torch.cat([emb, feat], dim=2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def evaluate_model(model, df):
    model.eval()
    results = []
    for pid in df["process_id"].unique():
        subset = df[df["process_id"] == pid]
        ds = MemoryDataset(subset)
        loader = DataLoader(ds, batch_size=64)
        preds, ys = [], []
        with torch.no_grad():
            for p, f, y in loader:
                p, f = p.to(DEVICE), f.to(DEVICE)
                out = model(p, f)
                preds.extend(out.cpu().numpy())
                ys.extend(y.numpy())
        rmse, mae, r2 = compute_metrics(ys, preds)
        results.append({"Process": pid, "RMSE": rmse, "MAE": mae, "R2": r2})
    return pd.DataFrame(results)


def train_model(model, loader, epochs=15, log_fn=print):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        grad_norm_total = 0
        for p, f, y in loader:
            p, f, y = p.to(DEVICE), f.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(p, f)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_norm = sum(
                param.grad.data.norm(2).item()
                for param in model.parameters()
                if param.grad is not None
            )
            grad_norm_total += total_norm
            optimizer.step()
            total_loss += loss.item()
        log_fn(f"Epoch {epoch:02d} | Loss {total_loss / len(loader):.4f} | GradNorm {grad_norm_total / len(loader):.4f}")
    return model


def unlearn(model, target_loader, remain_loader, log_fn=print):
    criterion = nn.MSELoss()

    # Phase 1: Erase target (gradient ascent)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    log_fn("**Phase 1 — Erase target (gradient ascent)**")
    for epoch in range(1, 3):
        total_loss = 0
        for p, f, y in target_loader:
            p, f, y = p.to(DEVICE), f.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(p, f)
            loss = criterion(pred, y)
            ascent_loss = torch.clamp(loss, max=5.0)
            (-ascent_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        log_fn(f"Epoch {epoch} | Target Loss {total_loss / len(target_loader):.4f}")

    # Phase 2: Restore remaining (gradient descent)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    log_fn("**Phase 2 — Restore remaining**")
    for epoch in range(1, 6):
        total_loss = 0
        for p, f, y in remain_loader:
            p, f, y = p.to(DEVICE), f.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(p, f)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        log_fn(f"Recovery Epoch {epoch} | Remaining Loss {total_loss / len(remain_loader):.4f}")
    return model


def run_unlearning_pipeline(target_process, log_fn=print):
    """
    Run full pipeline: generate data, train, unlearn target_process, return metrics.
    target_process: e.g. "P0", "P1", ..., "P9"
    """
    set_seed()
    log_fn(f"Device: {DEVICE}")

    df = generate_dataset()
    le = LabelEncoder()
    df["process_encoded"] = le.fit_transform(df["process_id"])
    df[["previous_memory", "cpu_usage"]] = StandardScaler().fit_transform(
        df[["previous_memory", "cpu_usage"]]
    )
    df["memory_usage_next"] = StandardScaler().fit_transform(
        df[["memory_usage_next"]]
    ).ravel()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_loader = DataLoader(MemoryDataset(train_df), batch_size=64, shuffle=True)

    model = DeepLSTM(len(le.classes_)).to(DEVICE)
    log_fn("**Training**")
    model = train_model(model, train_loader, log_fn=log_fn)

    log_fn("**Baseline metrics**")
    baseline_metrics = evaluate_model(model, test_df)

    target_loader = DataLoader(
        MemoryDataset(train_df[train_df["process_id"] == target_process]),
        batch_size=64,
        shuffle=True,
    )
    remain_loader = DataLoader(
        MemoryDataset(train_df[train_df["process_id"] != target_process]),
        batch_size=64,
        shuffle=True,
    )

    log_fn("**Unlearning**")
    model_un = DeepLSTM(len(le.classes_)).to(DEVICE)
    model_un.load_state_dict(model.state_dict())
    model_un = unlearn(model_un, target_loader, remain_loader, log_fn=log_fn)

    log_fn("**After unlearning metrics**")
    after_metrics = evaluate_model(model_un, test_df)

    before_target = baseline_metrics[baseline_metrics["Process"] == target_process]["RMSE"].values[0]
    after_target = after_metrics[after_metrics["Process"] == target_process]["RMSE"].values[0]
    certified_forgetting_score = float(after_target - before_target)

    comparison_rows = []
    for pid in baseline_metrics["Process"]:
        br = baseline_metrics[baseline_metrics["Process"] == pid]["RMSE"].values[0]
        ar = after_metrics[after_metrics["Process"] == pid]["RMSE"].values[0]
        br2 = baseline_metrics[baseline_metrics["Process"] == pid]["R2"].values[0]
        ar2 = after_metrics[after_metrics["Process"] == pid]["R2"].values[0]
        comparison_rows.append({
            "Process": pid,
            "Before RMSE": round(br, 6),
            "After RMSE": round(ar, 6),
            "Before R2": round(br2, 6),
            "After R2": round(ar2, 6),
        })
    comparison_df = pd.DataFrame(comparison_rows).sort_values("Process").reset_index(drop=True)

    return {
        "baseline_metrics": baseline_metrics,
        "after_metrics": after_metrics,
        "comparison_df": comparison_df,
        "certified_forgetting_score": certified_forgetting_score,
        "target_process": target_process,
    }
