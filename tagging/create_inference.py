import fire
import pandas as pd


def main(
    base_file="data/materials-science.elements.works.csv",
    ground_truth_file="tagging/out.csv",
    output_file="preparation/hpc/data-inference/untagged.csv",
    n_samples=500,
):
    all_works = pd.read_csv(base_file)
    tagged_works = pd.read_csv(ground_truth_file)

    untagged_works = all_works[~all_works.id.isin(tagged_works.id)]
    print(f"Found {len(untagged_works)} untagged works. From total of {len(all_works)}")

    sample = untagged_works.sample(n_samples)
    sample.to_csv(output_file, index=False)


if __name__ == "__main__":
    fire.Fire(main)
