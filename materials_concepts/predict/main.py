import configparser
import asyncio
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from materials_concepts.predict.predict import Predictor
from materials_concepts.model.graph import Graph
from materials_concepts.predict.utils import load_lookup


class Config:
    def __init__(self, config_path="predictor_config.ini"):
        self.config_path = Path(config_path)
        self.config = configparser.ConfigParser()
        if self.config_path.exists():
            self.config.read(self.config_path)

    def get(self, section, option, default=None):
        return self.config.get(section, option, fallback=default)

    def set(self, section, option, value):
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, option, value)

    def save(self):
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)


def prompt_for_path(prompt_text, default_path):
    user_input = input(f"{prompt_text} [{default_path}]: ").strip()
    return user_input if user_input else default_path


def get_paths_from_user(config):
    paths = {
        "feature_embeddings": prompt_for_path("Enter the path to the feature embeddings", config.get("paths", "feature_embeddings", "test-data/baseline/features.2022.binary.M.pkl.gz")),
        "concept_embeddings": prompt_for_path("Enter the path to the concept embeddings", config.get("paths", "concept_embeddings", "test-data/pure_embs/features.concept-embs.2022.M.pkl.gz")),
        "graph": prompt_for_path("Enter the path to the graph data", config.get("paths", "graph", "test-data/edges.M.pkl")),
        "model_baseline": prompt_for_path("Enter the path to the baseline model", config.get("paths", "model_baseline", "test-data/baseline/model.pt")),
        "model_pure_embs": prompt_for_path("Enter the path to the pure embeddings model", config.get("paths", "model_pure_embs", "test-data/pure_embs/model.pt")),
        "lookup": prompt_for_path("Enter the path to the lookup file", config.get("paths", "lookup", "test-data/lookup.M.csv")),
        "report": prompt_for_path("Enter the path to the report file", config.get("paths", "report", "report.md")),
    }
    if input("Save these paths? (y/n): ").lower() == "y":
        for key, value in paths.items():
            config.set("paths", key, value)
        config.save()
    return paths


async def main(k_concepts: int = 15, use_min_depth_of_threshold: bool = False):
    config = Config()
    if not config.config_path.exists():
        paths = get_paths_from_user(config)
    else:
        logger.info(f"Loading config from {config.config_path}")
        paths = {
            "feature_embeddings": config.get("paths", "feature_embeddings"),
            "concept_embeddings": config.get("paths", "concept_embeddings"),
            "graph": config.get("paths", "graph"),
            "model_baseline": config.get("paths", "model_baseline"),
            "model_pure_embs": config.get("paths", "model_pure_embs"),
            "lookup": config.get("paths", "lookup"),
            "report": config.get("paths", "report"),
        }

    # PREDICTION
    SINCE=2023

    # MODEL
    FEATURES=["True,False","False,True"]
    LAYERS=["20,300,180,108,64,10,1","1536,1024,819,10,1"] 
    BLENDING=[0.6, 0.4]

    concepts_input = input("Enter concepts to predict, separated by commas: ")
    concepts = [c.strip() for c in concepts_input.split(',')]

    if not Predictor.verify_concepts(concepts, paths['lookup']):
        return
    logger.info(f"Concepts verified")

    logger.info(f"Loading graph from '{paths['graph']}' (this may take a while)")
    G = Graph.from_path(paths['graph'])

    predictor = Predictor(
        logger=logger,
        lookup=paths['lookup'],
        feature_embeddings=paths['feature_embeddings'],
        concept_embeddings=paths['concept_embeddings'],
        graph=G,
        since=int(SINCE),
        layers=LAYERS,
        model=[paths['model_baseline'], paths['model_pure_embs']],
        features=FEATURES,
        blending=BLENDING,
    )

    for concept in tqdm(concepts):
        results = await predictor.predict(concept, k=k_concepts, min_depth=3 if use_min_depth_of_threshold else None)
        predictor.save_result(concept, results, paths['report'])


if __name__ == "__main__":
    asyncio.run(main())