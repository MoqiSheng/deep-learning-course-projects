"""Generate a teaching dataset, fit a lag model, and evaluate the held-out days."""
import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def main():
    folder = Path(__file__).resolve().parent
    rng = np.random.default_rng(42)
    intervals = np.arange(21 * 96)
    day = intervals // 96
    hour = intervals % 96 / 4
    weekday = day % 7 < 5
    rush = 74 * np.exp(-((hour - 8) / 1.8) ** 2) + 61 * np.exp(-((hour - 17.5) / 2.2) ** 2)
    midday = 21 * np.exp(-((hour - 13) / 5) ** 2)
    traffic = np.maximum(0, np.rint(15 + rush * np.where(weekday, 1, 0.63) + midday
                                    + rng.normal(0, 4, len(intervals)))).astype(int)
    stamps = [(datetime(2024, 1, 1) + timedelta(minutes=15 * int(i))).isoformat(timespec="minutes")
              for i in intervals]
    with (folder / "synthetic-flow.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["interval_start", "vehicles"])
        writer.writerows(zip(stamps, traffic))

    targets = intervals[4:]
    features = np.column_stack([np.ones(len(targets))] + [traffic[targets - k] for k in range(1, 5)])
    train = targets < 14 * 96
    weights = np.linalg.lstsq(features[train], traffic[targets[train]], rcond=None)[0]
    predictions = np.maximum(0, features @ weights)
    for name, selection in [("validation", (targets >= 14 * 96) & (targets < 18 * 96)),
                            ("test", targets >= 18 * 96)]:
        t = targets[selection]
        for method, estimate in [("last value", traffic[t - 1]), ("lag regression", predictions[selection])]:
            print(name, method, "MAE", round(np.abs(traffic[t] - estimate).mean(), 2))
    with (folder / "test-predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["interval_start", "observed", "last_value", "lag_regression"])
        for t, prediction in zip(targets, predictions):
            if t >= 18 * 96:
                writer.writerow([stamps[t], traffic[t], traffic[t - 1], round(prediction, 3)])


if __name__ == "__main__":
    main()
