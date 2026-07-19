"""Baseline SEM -> Depth Map regression.

Adapted from docs/[Baseline]_Simulation SEM 영상으로부터 Depth Map 생성 학습.ipynb
for local execution against data/ (2022 Samsung AI Challenge 3D Metrology data,
used here as a stand-in dataset ahead of the 2026 competition's data release).

Fixes vs. the original notebook (both needed for this to run correctly on Windows):
- glob() returns OS-native separators on Windows, so `path.split('/')[-1]` leaves a
  stray "SEM\\" prefix on output filenames; use os.path.basename() instead.
- DataLoader(num_workers>0) needs the multiprocessing spawn guard when run as a
  script (not needed inside a notebook kernel).

The simulation_data/Depth glob is deliberately globbed and concatenated with itself
before sorting: SEM has two iterations (itr0/itr1) per depth map, so duplicating
the depth list and sorting aligns each depth path with its two matching SEM paths
by position. This is the original notebook's pairing trick, not a bug -- verified
against the actual file counts (173304 SEM == 2x86652 Depth) before keeping it.
"""

import argparse
import glob
import os
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

CFG = {
    "WIDTH": 48,
    "HEIGHT": 72,
    "EPOCHS": 10,
    "LEARNING_RATE": 1e-3,
    "BATCH_SIZE": 128,
    "SEED": 41,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class CustomDataset(Dataset):
    def __init__(self, sem_path_list, depth_path_list):
        self.sem_path_list = sem_path_list
        self.depth_path_list = depth_path_list

    def __getitem__(self, index):
        sem_path = self.sem_path_list[index]
        sem_img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
        sem_img = np.expand_dims(sem_img, axis=-1).transpose(2, 0, 1)
        sem_img = sem_img / 255.0

        if self.depth_path_list is not None:
            depth_path = self.depth_path_list[index]
            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            depth_img = np.expand_dims(depth_img, axis=-1).transpose(2, 0, 1)
            depth_img = depth_img / 255.0
            return torch.Tensor(sem_img), torch.Tensor(depth_img)
        else:
            img_name = os.path.basename(sem_path)
            return torch.Tensor(sem_img), img_name

    def __len__(self):
        return len(self.sem_path_list)


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(CFG["HEIGHT"] * CFG["WIDTH"], 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, CFG["HEIGHT"] * CFG["WIDTH"]),
        )

    def forward(self, x):
        x = x.view(-1, CFG["HEIGHT"] * CFG["WIDTH"])
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, CFG["HEIGHT"], CFG["WIDTH"])
        return x


def validation(model, criterion, val_loader, device):
    model.eval()
    rmse = nn.MSELoss().to(device)

    val_loss = []
    val_rmse = []
    with torch.no_grad():
        for sem, depth in tqdm(iter(val_loader), desc="val"):
            sem = sem.float().to(device)
            depth = depth.float().to(device)

            model_pred = model(sem)
            loss = criterion(model_pred, depth)

            pred = (model_pred * 255.0).type(torch.int8).float()
            true = (depth * 255.0).type(torch.int8).float()
            b_rmse = torch.sqrt(rmse(pred, true))

            val_loss.append(loss.item())
            val_rmse.append(b_rmse.item())

    return np.mean(val_loss), np.mean(val_rmse)


def train(model, optimizer, train_loader, val_loader, device, epochs):
    model.to(device)
    criterion = nn.L1Loss().to(device)
    best_score = float("inf")
    best_model = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = []
        for sem, depth in tqdm(iter(train_loader), desc=f"epoch {epoch}"):
            sem = sem.float().to(device)
            depth = depth.float().to(device)

            optimizer.zero_grad()
            model_pred = model(sem)
            loss = criterion(model_pred, depth)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        val_loss, val_rmse = validation(model, criterion, val_loader, device)
        print(
            f"Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] "
            f"Val Loss : [{val_loss:.5f}] Val RMSE : [{val_rmse:.5f}]"
        )

        if best_score > val_rmse:
            best_score = val_rmse
            best_model = model

    return best_model


def inference(model, test_loader, device, output_dir: Path):
    model.to(device)
    model.eval()

    result_name_list = []
    result_list = []
    with torch.no_grad():
        for sem, names in tqdm(iter(test_loader), desc="infer"):
            sem = sem.float().to(device)
            model_pred = model(sem)

            for pred, img_name in zip(model_pred, names):
                pred = pred.cpu().numpy().transpose(1, 2, 0) * 255.0
                result_name_list.append(img_name)
                result_list.append(pred)

    submission_dir = output_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as submission:
        for name, pred_img in zip(result_name_list, result_list):
            img_path = submission_dir / name
            cv2.imwrite(str(img_path), pred_img)
            submission.write(img_path, arcname=name)

    return zip_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="dataset root (default: data)")
    parser.add_argument(
        "--output-dir",
        default="runtime/baseline_output",
        help="where submission.zip and intermediate files land",
    )
    parser.add_argument("--epochs", type=int, default=CFG["EPOCHS"])
    parser.add_argument("--num-workers", type=int, default=6)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")

    seed_everything(CFG["SEED"])

    simulation_sem_paths = sorted(glob.glob(str(data_dir / "simulation_data" / "SEM" / "*" / "*" / "*.png")))
    simulation_depth_paths = sorted(
        glob.glob(str(data_dir / "simulation_data" / "Depth" / "*" / "*" / "*.png"))
        + glob.glob(str(data_dir / "simulation_data" / "Depth" / "*" / "*" / "*.png"))
    )
    assert len(simulation_sem_paths) == len(simulation_depth_paths), (
        f"SEM/Depth count mismatch: {len(simulation_sem_paths)} vs {len(simulation_depth_paths)}"
    )

    data_len = len(simulation_sem_paths)
    train_sem_paths = simulation_sem_paths[: int(data_len * 0.8)]
    train_depth_paths = simulation_depth_paths[: int(data_len * 0.8)]
    val_sem_paths = simulation_sem_paths[int(data_len * 0.8) :]
    val_depth_paths = simulation_depth_paths[int(data_len * 0.8) :]

    train_dataset = CustomDataset(train_sem_paths, train_depth_paths)
    train_loader = DataLoader(
        train_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=True, num_workers=args.num_workers
    )
    val_dataset = CustomDataset(val_sem_paths, val_depth_paths)
    val_loader = DataLoader(
        val_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=False, num_workers=args.num_workers
    )

    model = BaseModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])

    infer_model = train(model, optimizer, train_loader, val_loader, device, args.epochs)

    test_sem_path_list = sorted(glob.glob(str(data_dir / "test" / "SEM" / "*.png")))
    test_dataset = CustomDataset(test_sem_path_list, None)
    test_loader = DataLoader(
        test_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=False, num_workers=args.num_workers
    )

    zip_path = inference(infer_model, test_loader, device, output_dir)
    print(f"submission written to {zip_path}")


if __name__ == "__main__":
    main()
