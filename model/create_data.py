import numpy as np
import random
from tqdm import tqdm
import pickle
from collections import Counter
import fire
from graph import Graph


class DataGenerator:
    def __init__(
        self,
        path_to_graph: str,
        training_data: bool,
        year_start: int,
        year_delta: int,
        verbose: bool = True,
    ):
        self.training_data = training_data
        self.verbose = verbose
        self.year_start = year_start
        self.year_delta = year_delta
        year_end = year_start + year_delta

        self.graph = Graph(path_to_graph)
        self.past_graph = self.graph.get_nx_graph(until_year=year_start)
        self.future_graph = self.graph.get_nx_graph(until_year=year_end)

    def generate(self, edges_used: int, min_links: int = 1, max_v_degree: int = None):
        """
        Generates training or test data, based on chosen strategy in constructor.

        Parameters:
        edges_used: number of vertex pairs to generate
        min_links: minimum number of links (at time = year_start + delta) between vertex pairs to be counted as positive sample
        max_v_degree: maximum degree of vertices which are considered for sampling

        Returns:
        X: array of vertex pairs with vertex [0, len(self.graph.num_of_vertices)]
        y: array of labels (1 = connected, 0 = unconnected)
        """
        if self.training_data:
            X, y = self._generate_train(edges_used, min_links, max_v_degree)
        else:
            X, y = self._generate_test(edges_used, min_links, max_v_degree)

        if self.verbose:
            print(f"# {len(X)} samples")
            print(f"{Counter(y)} solution distribution")

        return X, y

    def _generate_train(self, edges_used: int, min_links: int, max_v_degree: int):
        """Generate training data by taking all positive samples and randomly drawing negative samples until the desired number of samples is reached.
        Warning: This leads to an overrepresentation of positive samples.
        """
        filtered_vertices = self.graph.get_vertices(max_degree=max_v_degree)

        pos_samples = self._get_pos_samples(filtered_vertices, min_links)
        to_draw_neg = edges_used - len(pos_samples)
        neg_samples = self._get_neg_samples(to_draw_neg, filtered_vertices)

        X = np.array(pos_samples + neg_samples)
        y = np.array([1] * len(pos_samples) + [0] * len(neg_samples))
        return self.shuffle(X, y)

    def _generate_test(self, edges_used: int, min_links: int, max_v_degree: int):
        """Generate test data by drawing vertex pairs randomly until the desired number of samples is reached.
        Drawn sample resembles the real distribution of (un)connected vertex pairs.
        """
        filtered_vertices = self.graph.get_vertices(
            max_degree=max_v_degree
        )  # TODO: For testing necessary as well?

        X, y = self._get_samples(edges_used, filtered_vertices, min_links)

        return self.shuffle(X, y)

    def _get_pos_samples(self, filtered_vertices, min_links):
        """Positive samples of vertex pairs: {year_start} unconnected, {year_start + delta} connected

        Aproach:
        All edges which exist at {year_start + delta} and didn't exist at {year_start} are candidates.
        Filter out all edges which have less than {min_links} links at {year_start + delta}
        and check if the vertices are in the filtered set of vertices.
        """
        pos_samples = set(self.future_graph.edges()) - set(self.past_graph.edges())

        lookup = {elem: True for elem in filtered_vertices}
        pos_samples = [
            (v1, v2) for v1, v2 in pos_samples if lookup.get(v1) and lookup.get(v2)
        ]

        pos_samples = [
            (v1, v2)
            for v1, v2 in pos_samples
            if self.future_graph.edges[v1, v2]["links"] >= min_links
        ]

        return pos_samples

    def _get_neg_samples(self, to_draw, vertices):
        """Negative samples of vertex pairs: {year_start} unconnected, {year_start + delta} unconnected"""
        X = []

        draw_sample = self.get_draw_sample(len(vertices))
        pbar = tqdm(total=to_draw)
        while len(X) < to_draw:
            i1, i2 = draw_sample()
            v1, v2 = vertices[i1], vertices[i2]

            if (
                v1 != v2
                and not self.past_graph.has_edge(v1, v2)
                and not self.future_graph.has_edge(v1, v2)
                # order of vertices in call doesn't matter as networkx Graph is undirected
            ):
                X.append((v1, v2))
                pbar.update(1)

        pbar.close()

        return X

    def _get_samples(self, to_draw, vertices, min_links):
        X, y = [], []

        draw_sample = self.get_draw_sample(len(vertices))
        pbar = tqdm(total=to_draw)
        while len(X) < to_draw:
            i1, i2 = draw_sample()
            v1, v2 = vertices[i1], vertices[i2]

            if v1 != v2 and not self.past_graph.has_edge(v1, v2):
                X.append((v1, v2))

                is_pos_sample = (
                    self.future_graph.has_edge(v1, v2)
                    and self.future_graph.edges[v1, v2]["links"] >= min_links
                )
                y.append(int(is_pos_sample))
                pbar.update(1)

        pbar.close()

        return np.array(X), np.array(y)

    @staticmethod
    def get_draw_sample(n):
        bag = range(n)
        drawn = set()

        def draw_sample():
            while (comb := frozenset(random.sample(bag, 2))) in drawn:
                pass

            drawn.add(comb)
            return comb

        return draw_sample

    @staticmethod
    def shuffle(X, y):
        """Shuffle X and y in unison"""
        assert len(X) == len(y)
        p = np.random.permutation(len(X))
        return X[p], y[p]


def main(
    graph_path="graph/edges.pkl",
    data_path="model/data.pkl",
    year_start_train=2016,
    year_start_test=2019,
    year_delta=3,
    edges_used_train=4_000_000,
    edges_used_test=1_000_000,
    min_links=1,
    max_v_degree=None,
    verbose=True,
):
    train_generator = DataGenerator(
        graph_path,
        training_data=True,
        year_start=year_start_train,
        year_delta=year_delta,
        verbose=verbose,
    )

    test_generator = DataGenerator(
        graph_path,
        training_data=False,
        year_start=year_start_test,
        year_delta=year_delta,
        verbose=verbose,
    )

    X_train, y_train = train_generator.generate(
        edges_used=edges_used_train, min_links=min_links, max_v_degree=max_v_degree
    )

    X_test, y_test = test_generator.generate(
        edges_used=edges_used_test, min_links=min_links, max_v_degree=max_v_degree
    )

    storage = {
        "year_train": year_start_train,
        "year_test": year_start_test,
        "year_delta": year_delta,
        "min_links": min_links,
        "max_v_degree": max_v_degree,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }

    with open(data_path, "wb") as f:
        pickle.dump(storage, f)


if __name__ == "__main__":
    fire.Fire(main)
