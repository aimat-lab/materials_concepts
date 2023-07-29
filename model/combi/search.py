from train import main
import itertools
import random
import hashlib
import json

constants = dict(
    data_path="data/model/data.M.pkl",
    emb_f_train_path="data/model/combi/features_2016.M.pkl.gz",
    emb_f_test_path="data/model/combi/features_2019.M.pkl.gz",
    emb_c_train_path="data/model/concept_embs/av_embs_2016.M.pkl.gz",
    emb_c_test_path="data/model/concept_embs/av_embs_2019.M.pkl.gz",
    lr=0.001,
    gamma=0.8,
    step_size=50,
    batch_size=1000,
    num_epochs=5,  # 5000,
    pos_ratio=0.3,
    log_interval=5,  # 50,
)

config = {
    "layer_count": [2, 4],
    "hidden_dim": [512, 1024, 1556],
    "layer_decrease": [0.4, 0.6, 0.8],
}


class GridSearch:
    def __init__(self, config, blacklist=[]):
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

            layer_dims = self._generate_layer_dims(
                run_config["layer_count"],
                run_config["hidden_dim"],
                run_config["layer_decrease"],
            )

            layer_dims += [10]

            # DEBUG
            # print(f"Hash: {params_hash}")
            # print("Layer dims:")
            # print(layer_dims)
            # print("Total params:")
            # print(f"{functools.reduce(lambda a, b: a * b, layer_dims):.3e}")

            main(
                **constants,
                layers=layer_dims,
                log_file=f"logs/{params_hash}.log",
                save_model=f"data/model/combi/{params_hash}.pt",
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

    @staticmethod
    def hash_dict(d):
        d_json = json.dumps(d, sort_keys=True)
        hash_object = hashlib.md5(d_json.encode())
        return hash_object.hexdigest()


blacklist = ["4e1f636becc25a9cad8f3e239faf4f7c"]  # list of config hashes to skip

if __name__ == "__main__":
    grid_search = GridSearch(config, blacklist=blacklist)
    grid_search.run(randomize=False)

# 6.719126e+12 => OK w/ batch size 1000
