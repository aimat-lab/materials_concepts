from train import main
import itertools
import random
import hashlib
import json

constants = dict(
    data_path="data-v2/model/data.M.pkl",
    emb_f_train_path=False,
    emb_f_test_path=False,
    emb_c_train_path="data-v2/model/combi/hq.word-embs.2016.M.pkl.gz",
    emb_c_test_path="data-v2/model/combi/hq.word-embs.2019.M.pkl.gz",
    batch_size=1000,
    num_epochs=10000,
    log_interval=200,
    layers=[1536, 1024, 819, 10, 1],
    lr=1e-3,
    step_size=200,
    gamma=0.9,
    dropout=0.1,
    pos_ratio=0.3,
)

config = dict(
    layer_count=[2, 4],
    hidden_dim=[1024, 1200, 1536],
    layer_decrease=[0.5, 0.75, 0.9],
)


class GridSearch:
    def __init__(self, base_model, config, blacklist=[]):
        self.base_model = base_model
        self.config = config
        self.blacklist = blacklist

    def run(self, randomize=False):
        run_configs = self._generate_run_config()
        if randomize:
            random.shuffle(run_configs)

        for i, run_config in enumerate(run_configs):
            params_hash = self.hash_dict(run_config)

            if params_hash in self.blacklist:
                self._print_begin(f"Skipping run: {i + 1}/{len(run_configs)}", width=80)
                continue

            self._print_begin(f"Starting run: {i + 1}/{len(run_configs)}", width=80)

            print(f"{params_hash}".center(80))
            self._print_dict(run_config)
            print("-" * 80)

            layer_dims = self._generate_layer_dims(
                run_config["layer_count"],
                run_config["hidden_dim"],
                run_config["layer_decrease"],
            )

            layer_dims = [1536] + layer_dims + [10, 1]

            main(
                **constants,
                layers=layer_dims,
                log_file=f"logs-v2/gridsearch/{self.base_model}/{params_hash}.log",
                save_model=f"data-v2/model/{self.base_model}/gridsearch/{params_hash}.pt",
            )

    def _generate_run_config(self):
        keys = self.config.keys()
        values = (self.config[key] for key in keys)
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combination)) for combination in combinations]

    def _print_begin(self, desc, width=40):
        print("=" * width)
        print(f"{desc}".center(width))
        print("=" * width)

    def _generate_layer_dims(self, layer_count, hidden_dim, layer_decrease):
        layer_dims = [hidden_dim]
        for _ in range(layer_count - 1):
            layer_dims.append(int(layer_dims[-1] * layer_decrease))

        return layer_dims

    def _print_dict(self, d):
        for k, v in d.items():
            print(f"{k}: {v}")

    def _debug(self, **kwargs):
        self._print_dict(kwargs)

    @staticmethod
    def hash_dict(d):
        d_json = json.dumps(d, sort_keys=True)
        hash_object = hashlib.md5(d_json.encode())
        return hash_object.hexdigest()


blacklist = []  # list of config hashes to skip

if __name__ == "__main__":
    grid_search = GridSearch("pure_embs", config, blacklist=blacklist)
    grid_search.run(randomize=False)

# 6.72e+12 => OK w/ batch size 1000

# [1536, 1024, 819, 10, 1] => AUC 0.8694
