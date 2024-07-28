import os
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils import create_directory, load_data
from src.draw_functions import draw_elastic, draw_elastic_gif


@hydra.main(config_name="config_hydra.yaml", config_path="config")
def main(args: DictConfig):
    with open("config_hydra.yaml", "w") as f:
        OmegaConf.save(args, f)

    output_dir = args.output_dir
    create_directory(output_dir)

    dataset = args.dataset

    output_dir_dataset = os.path.join(output_dir, dataset)
    create_directory(output_dir_dataset)

    X, y, is_classif = load_data(
        dataset_name=dataset, split=args.split, znormalize=args.znormalize
    )

    if is_classif:
        ts1 = X[y == args.class_x][
            np.random.randint(low=0, high=len(X[y == args.class_x]), size=1)[0]
        ]
        ts2 = X[y == args.class_y][
            np.random.randint(low=0, high=len(X[y == args.class_y]), size=1)[0]
        ]
    else:
        ts1 = X[np.random.randint(low=0, high=len(X), size=1)[0]]
        ts2 = X[np.random.randint(low=0, high=len(X), size=1)[0]]

    draw_elastic(
        x=ts1,
        y=ts2,
        output_dir=output_dir_dataset,
        figsize=args.figsize,
        metric=args.metric,
        metric_params=args.metric_params,
        show_warping_connections=args.show_warping,
    )

    draw_elastic_gif(
        output_dir=output_dir_dataset,
        x=ts1,
        y=ts2,
        figsize=args.figsize,
        fontsize=10,
        metric_params=args.metric_params,
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
