import pandas as pd

TAG_DELIMITER = "§"


def extract_tags(lines):
    start_index = lines.index(TAG_DELIMITER) + 1
    end_index = lines.index(TAG_DELIMITER, start_index)
    return lines[start_index:end_index]


def main(input_file, output_file, abstract_file, separator_length=80):
    with open(input_file) as f:
        text = f.read()

    works_data = text.split("=" * separator_length)

    data = []

    for work in works_data:
        lines = [
            line.strip().lower() for line in work.split("\n") if line.strip()
        ]  # remove empty lines, clean, lower case

        if len(lines) < 3:
            continue

        work_id = lines[0].upper()  # ID has W19393939 format
        try:
            tags = extract_tags(lines)
        except ValueError:
            tags = []
        data.append(dict(id=work_id, tags=",".join(tags)))

    tags = pd.DataFrame(data)
    abstracts = pd.read_csv(abstract_file)[["id", "abstract"]]
    df = tags.join(abstracts.set_index("id"), on="id", how="left")
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a .csv file mapping works to their concepts."
    )

    parser.add_argument("input_file", help="Path to the tagged abstracts file")
    parser.add_argument("output_file", help="Path to the .csv output file")
    parser.add_argument(
        "--separator-length", help="Length of the separator", default=80
    )
    parser.add_argument(
        "--abstract_file",
        help="Path to the abstract file",
        default="data/materials-science.elements.works.csv",
    )

    args = parser.parse_args()

    main(args.input_file, args.output_file, args.abstract_file, args.separator_length)
