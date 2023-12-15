from advanced_analysis import Analyser
import pandas as pd
from pathlib import Path
from fire import Fire


def main(
    embeddings: str = "evolution/transformed.1990.pkl.gz",
    output: str = "evolution/1990.png",
):
    analyser = Analyser(
        pd.DataFrame(columns=["source", "llama_concepts"]),
        source="<non-existent>",
        all_embeddings=Path(embeddings),
        predictions="mock.pkl",
        occurance_filter=2,
    )

    analyser.save_map_with_important_landmarks(to=Path(output))


if __name__ == "__main__":
    Fire(main)
