"""Forecast one 15-minute interval from the four preceding observations."""
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn


class FlowForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(input_size=1, hidden_size=16, batch_first=True)
        self.output = nn.Linear(16, 1)

    def forward(self, sequence):
        _, hidden = self.encoder(sequence)
        return self.output(hidden[-1]).flatten()


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    folder = Path(__file__).resolve().parent
    with (folder / "synthetic-flow.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    flow = np.array([float(row["vehicles"]) for row in rows], dtype=np.float32)
    mean, std = flow[:14 * 96].mean(), flow[:14 * 96].std()
    scaled = (flow - mean) / std
    targets = np.arange(4, len(flow))
    x = torch.tensor(np.stack([scaled[t-4:t] for t in targets]))[:, :, None]
    y = torch.from_numpy(scaled[targets])
    train = torch.where(torch.from_numpy(targets < 14 * 96))[0]
    valid = (targets >= 14 * 96) & (targets < 18 * 96)
    model = FlowForecast()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    best_loss = float("inf")
    for _ in range(25):
        for batch in train[torch.randperm(len(train))].split(64):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_loss = (model(x[valid]) - y[valid]).abs().mean().item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {name: param.detach().clone() for name, param in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = targets >= 18 * 96
    with torch.no_grad():
        prediction = np.maximum(0, model(x[test]).numpy() * std + mean)
    with (folder / "gru-predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["interval_start", "gru"])
        for target, estimate in zip(targets[test], prediction):
            writer.writerow([rows[target]["interval_start"], round(float(estimate), 3)])
    print(f"GRU test MAE = {np.abs(prediction - flow[targets[test]]).mean():.2f} vehicles / 15 min")


if __name__ == "__main__":
    main()
