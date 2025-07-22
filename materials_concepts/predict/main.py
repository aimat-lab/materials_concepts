import fire
from tqdm import tqdm

from materials_concepts.predict.predict import Predictor
from materials_concepts.model.graph import Graph
from materials_concepts.predict.utils import load_lookup
from loguru import logger

# EMBEDDINGS
FEATURE_EMBEDDINGS="test-data/baseline/features.2022.binary.M.pkl.gz"
CONCEPT_EMBEDDINGS="test-data/pure_embs/features.concept-embs.2022.M.pkl.gz"

# GRAPH
GRAPH="test-data/edges.M.pkl"

# PREDICTION
SINCE=2023

# MODEL
MODEL=["test-data/baseline/model.pt","test-data/pure_embs/model.pt"]
FEATURES=["True,False","False,True"]
LAYERS=["20,300,180,108,64,10,1","1536,1024,819,10,1"] 
BLENDING=[0.6, 0.4]

# LOOKUP
LOOKUP="test-data/lookup.M.csv"

def verify_concepts(concepts: list[str]):
    lookup = load_lookup(LOOKUP)
    available_concepts = set(lookup["concept"].tolist())
    for concept in concepts:
        if concept not in available_concepts:
            print(f"Concept '{concept}' is not contained in our database. Please check the contents of {LOOKUP} to see all available concepts.")
            return False
    return True

def load_predictor():
    logger.info(f"Loading graph from '{GRAPH}'")
    G = Graph.from_path(GRAPH)

    predictor = Predictor(
        logger=logger,
        lookup=LOOKUP,
        feature_embeddings=FEATURE_EMBEDDINGS,
        concept_embeddings=CONCEPT_EMBEDDINGS,
        graph=G,
        since=int(SINCE),
        layers=LAYERS,
        model=MODEL,
        features=FEATURES,
        blending=BLENDING,
    )
    return predictor

def save_result(concept: str, result: list[dict], filename: str):
    # save as markdown:
    # heading ## keyword, then list with concept: score (rounded to 4 decimal places)
    with open(filename, "a") as f:
        f.write(f"## {concept}\n")
        for r in result:
            f.write(f"- {r['concept']}: *{r['score']:.4f}*\n")
        f.write("\n")

async def main(concepts: list[str], report_path: str = "report.md"):
    if not verify_concepts(concepts):
        return
    logger.info(f"Concepts verified: {len(concepts)}")
    
    predictor = load_predictor()
    for concept in tqdm(concepts):
        results = await predictor.predict(concept, k=15)
        save_result(concept, results, report_path)


if __name__ == "__main__":
    fire.Fire(main)