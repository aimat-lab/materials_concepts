import pandas as pd
import textwrap

max_line_length = 80


def create_abstracts(
    input_file, output_file, max_line_length, max_abstracts=None, random=False
):
    df = pd.read_csv(input_file)

    if random:
        df = df.sample(frac=1).reset_index(drop=True)

    if max_abstracts is not None:
        df = df.head(max_abstracts)

    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            wrapped_text = textwrap.fill(row.abstract, width=max_line_length)

            f.write(row.id + "\n\n")  # include ID to search abstracts

            f.write(wrapped_text)

            f.write(
                "\n\n" + "=" * max_line_length + "\n\n"
            )  # Add a newline between text blocks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a .txt file with line-wrapped abstracts."
    )

    parser.add_argument("works_file", help="Path to the works file")
    parser.add_argument("abstracts_file", help="Path to the output file")
    parser.add_argument("--line-length", help="Maximum characters per line", default=80)
    parser.add_argument(
        "--max-abstracts",
        help="Maximum number of abstracts to include",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--random",
        help="Randomize the order of abstracts",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    create_abstracts(
        args.works_file,
        args.abstracts_file,
        args.line_length,
        args.max_abstracts,
        args.random,
    )
