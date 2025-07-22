import asyncio

from materials_concepts.predict.predict import Predictor
from materials_concepts.model.graph import Graph
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

async def main():
    results = await predictor.predict("dna concentration", k=15)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())