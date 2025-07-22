import fire
import numpy as np

from materials_concepts.model.metrics import print_metrics, test
from materials_concepts.utils.utils import (
    load_compressed,
    load_pickle,
    save_compressed
)


def main(
    data_path: str,
    predictions_path_1: str,
    predictions_path_2: str,
    save_path: str,
    blend: list[float] = (0.6, 0.4),
    metrics_path: str | None = None,
):
    """
    Blend two sets of predictions and evaluate the result.
    """
    data = load_pickle(data_path)
    labels = data["y_test"]

    preds1 = load_compressed(predictions_path_1)
    preds2 = load_compressed(predictions_path_2)

    blended_predictions = preds1 * blend[0] + preds2 * blend[1]

    if metrics_path:
        print_metrics(labels, blended_predictions, threshold=0.5, save_path=metrics_path)

    save_compressed(blended_predictions, save_path)


if __name__ == "__main__":
    fire.Fire(main) 