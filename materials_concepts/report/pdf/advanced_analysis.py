import zipfile
import numpy as np
import pandas as pd
import random
from ast import literal_eval
from collections import Counter, defaultdict
from typing import Literal
from loguru import logger
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import gzip
import pickle
from pylatex import (
    Document,
    Section,
    Subsection,
    Subsubsection,
    Command,
    NoEscape,
    Itemize,
    Figure,
)
from pylatex.utils import italic, bold
from tqdm import tqdm
from typing import Union
import os


def load(path: str):
    return load_pickle(path) if path.endswith(".pkl") else load_compressed(path)


def load_compressed(path: str):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class Analyser:
    def __init__(
        self,
        base_dir: Path,
        _df: pd.DataFrame,
        source: str,
        predictions: Path,
        # degrees: dict[str, int],
        all_embeddings: Path,
        occurance_filter: int = 2,
    ):
        self.source = source

        self.all_embeddings: dict = load(str(all_embeddings))

        lookup = pd.read_csv(base_dir / "lookup.M.new.csv")
        self.degrees = dict(zip(lookup["concept"], lookup["count"]))

        self.occurance_filter = occurance_filter
        self.df = _df[_df["source"] == source]  # relevant data for source
        self.all_concepts = Counter(
            item
            for sublist in self.df["llama_concepts"].values
            for item in sublist
            if item in self.all_embeddings  # only consider concepts with embeddings
        )
        logger.info(f"Found {len(self.all_concepts)} concepts in {source}")
        self.interesting_concepts = {
            concept: count
            for concept, count in self.all_concepts.items()
            if count >= occurance_filter
        }
        logger.info(f"Found {len(self.interesting_concepts)} interesting concepts")

        self.predictions: dict = load_compressed(predictions)
        logger.info(f"Loaded {len(self.predictions)} predictions")

        logger.info("Loading own and other predictions")
        (
            self.own_predictions,
            self.other_predictions,
        ) = self._suggestions_to_own_and_other_concepts()
        logger.info(f"Loaded {len(self.own_predictions):,} predictions to own concepts")
        logger.info(
            f"Loaded {len(self.other_predictions):,} predictions to other concepts"
        )

    def save_world_cloud(
        self, to: Path, mode: Literal["all", "interesting"] = "interesting"
    ):
        data_dict = {
            "all": self.all_concepts,
            "interesting": self.interesting_concepts,
        }

        wordcloud = WordCloud(
            width=1200, height=1200, background_color="white", min_font_size=10
        ).generate_from_frequencies(data_dict[mode])
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)

        plt.savefig(to)

    # UNUSED
    def legacy_print_2d_word_embeddings_map(self, to: Path):
        embeddings = {
            key: self.all_embeddings[key] for key in self.all_embeddings.keys()
        }

        # Extracting x and y coordinates and terms
        x = [point[0] for point in embeddings.values()]
        y = [point[1] for point in embeddings.values()]
        terms = list(embeddings.keys())

        # Creating the scatter plot
        fig = px.scatter(x=x, y=y, text=terms)

        # Customizing the plot
        fig.update_traces(textposition="top center")

        # Showing the plot
        fig.update_layout(
            xaxis_range=[-12, 12],  # Replace xmin and xmax with your desired values
            yaxis_range=[-12, 12],  # Replace ymin and ymax with your desired values
        )

        fig.write_image(str(to), width=1200, height=1200)

    def get_grid(self, panel_size: int = 2):
        """For each panel, return a list of concepts that are in that panel.
        Splits the 2D space into a grid of squares with side length panel_size."""

        grid = defaultdict(list)

        for concept, point in tqdm(self.all_embeddings.items()):
            grid_x = int(point[0] // panel_size)
            grid_y = int(point[1] // panel_size)
            grid[(grid_x, grid_y)].append(concept)

        return grid

    def _add_background_trace(
        self,
        fig: go.Figure,
        max_concepts_for_total_inclusion: int = 300,
        inclusion_rate: float = 0.2,
        color: str = "#BEBADA",
    ):
        included_subset = []
        for _, concepts in self.get_grid(panel_size=1).items():  # Iter over all panels
            if len(concepts) > max_concepts_for_total_inclusion:
                include_amt = round(inclusion_rate * len(concepts))
                included_subset.extend(random.sample(concepts, include_amt))
            else:
                included_subset.extend(concepts)

        background = {
            "x": [self.all_embeddings[concept][0] for concept in included_subset],
            "y": [self.all_embeddings[concept][1] for concept in included_subset],
        }

        fig.add_trace(
            go.Scatter(
                x=background["x"],
                y=background["y"],
                mode="markers",
                name=f"All embeddings (inclusion: {inclusion_rate * 100}%)",
                text=[],
                textposition="top center",
                marker=dict(color=color),
            )
        )

    def _add_concepts_trace(
        self,
        fig: go.Figure,
        concepts: list[str],
        name: str,
        color: str,
        display_text: bool = True,
    ):
        fig.add_trace(
            go.Scatter(
                x=[self.all_embeddings[concept][0] for concept in concepts],
                y=[self.all_embeddings[concept][1] for concept in concepts],
                mode="markers+text",
                name=name,
                text=concepts if display_text else [],
                textposition="top center",
                marker=dict(color=color),
            )
        )

    def generate_map(
        self,
        include_concepts: dict[str, list[str]],
        colors: Union[dict[str], None] = None,
        background_concepts: bool = True,
        max_concepts_for_total_inclusion: int = 300,
        background_inclusion_rate: float = 0.2,
        background_color: str = "#BEBADA",
    ):
        fig = go.Figure()

        if background_concepts:
            self._add_background_trace(
                fig,
                max_concepts_for_total_inclusion,
                background_inclusion_rate,
                background_color,
            )

        default_colors = [
            "#FF0000",
            "#f5d742",
            "#0000FF",
            "#00FF00",
            "#FF00FF",
            "#00FFFF",
        ]
        for index, (name, concepts) in enumerate(include_concepts.items()):
            logger.info(f"Adding {name} concepts trace to map")
            color_default = default_colors[index]
            color = colors.get(name, color_default) if colors else color_default
            self._add_concepts_trace(fig, concepts, name, color, display_text=True)

        return fig

    def save_map_with_interesting_concepts(self, to: Path, fig_size: int = 1200):
        fig = self.generate_map(
            include_concepts={"Interesting": list(self.interesting_concepts.keys())}
        )
        logger.info(f"Saving whole embeddings map w/ interesting concepts {to}")
        fig.write_image(str(to), width=fig_size, height=fig_size)

    def save_map_with_important_landmarks(
        self, to: Path, panel_size=1, fig_size: int = 1200
    ):
        landmarks = {}

        for panel, panel_concepts in tqdm(self.get_grid(panel_size=panel_size).items()):
            degrees = {concept: self.degrees[concept] for concept in panel_concepts}
            index_of_max_degree = np.argmax(list(degrees.values()))
            landmarks[panel] = panel_concepts[index_of_max_degree]

        fig = self.generate_map(
            include_concepts={"Landmarks": list(landmarks.values())},
            colors={"Landmarks": "#FF0000"},
        )

        fig.update_layout(title="Map of Materials Science")

        logger.info(f"Saving whole embeddings map w/ landmarks {to}")
        # as_html = fig.to_html(
        #     full_html=True,
        #     include_plotlyjs=True,
        #     default_height="100vh",
        #     default_width="80vw",
        # )
        # with open("map.html", "w") as f:
        #     f.write(as_html)
        fig.write_image(str(to), width=fig_size, height=fig_size)

    def _iter_suggestions(self, combined_with: Literal["own", "other"]):
        for own_concept, all_predictions in self.predictions.items():
            for prediction in all_predictions:
                if (
                    combined_with == "own"
                    and prediction["concept"] in self.all_concepts
                ) or (
                    combined_with == "other"
                    and prediction["concept"] not in self.all_concepts
                ):
                    yield own_concept, prediction["concept"], prediction["score"]

    def _suggestions_to_own_and_other_concepts(self):
        sort_by_score = lambda x: x[2]
        return (
            sorted(
                self._iter_suggestions(combined_with="own"),
                key=sort_by_score,
                reverse=True,
            ),
            sorted(
                self._iter_suggestions(combined_with="other"),
                key=sort_by_score,
                reverse=True,
            ),
        )

    def get_highly_connective_predictions(self, threshold=0.995):
        logger.info(f"Retrieving predictions with score > {threshold}")
        all_predictions = self.own_predictions + self.other_predictions

        highly_scored_predictions = [
            (x, y, score) for x, y, score in all_predictions if score > threshold
        ]
        logger.info(
            f"Found {len(highly_scored_predictions)} predictions (score > {threshold})"
        )

        suggested_new_keywords = defaultdict(list)
        for own_concept, new_concept, score in highly_scored_predictions:
            suggested_new_keywords[new_concept].append((own_concept, score))

        new_keyword_matches = {  # how many of the own concepts are very likely to be combined with the new concept
            key: len(value) for key, value in suggested_new_keywords.items()
        }
        interesting_new_keywords = sorted(
            new_keyword_matches.items(), key=lambda x: x[1], reverse=True
        )

        return {
            new_keyword: suggested_new_keywords[new_keyword]
            for new_keyword, _ in interesting_new_keywords
        }

    def get_potentially_interesting_predictions(
        self,
        max_other_degree: int = 100,
        min_distance: float = 3,
        max_distance: float = 6,
    ) -> list[tuple[str, str, float, float]]:
        all_predictions = self.own_predictions + self.other_predictions

        logger.info(f"Retrieving predictions with degree <= {max_other_degree}")
        after_degree_filter = [
            (x, y, score)
            for x, y, score in all_predictions
            if self.degrees[y] <= max_other_degree
        ]
        logger.info(f"Found {len(after_degree_filter):,} predictions")

        logger.info(
            f"Retrieving predictions with {min_distance} <= distance <= {max_distance}"
        )
        after_distance_filter = [
            (x, y, score, concept_dist)
            for x, y, score in after_degree_filter
            if min_distance <= (concept_dist := self.emb_distance(x, y)) <= max_distance
        ]
        logger.info(f"Found {len(after_distance_filter):,} predictions")

        ordered_combinations = sorted(
            after_distance_filter, key=lambda x: x[2], reverse=True
        )

        # fig = self.generate_map(
        #     include_concepts={
        #         "Own Concepts": [x for x, _, _, _ in ordered_combinations],
        #         "Other Concepts": [y for _, y, _, _ in ordered_combinations],
        #     },
        # )
        # fig.write_image(
        #     "potentially_interesting_predictions.png", width=1200, height=1200
        # )

        return ordered_combinations

    def emb_distance(self, concept_1: str, concept_2: str):
        try:
            emb_1 = self.all_embeddings[concept_1]
            emb_2 = self.all_embeddings[concept_2]
        except KeyError:
            return -1  # If there is no embedding, an impossible distance is returned
        return ((emb_1[0] - emb_2[0]) ** 2 + (emb_1[1] - emb_2[1]) ** 2) ** 0.5


class Report:
    def __init__(self, base_dir: Path, author_name: str, analyser: Analyser):
        logger.info(f"Initializing Report at {base_dir}")
        if not base_dir.exists():
            os.makedirs(base_dir)

        self.base_dir = base_dir
        self.analyser = analyser

        _doc = Document(str(this_reports_base))
        _doc.preamble.append(Command("title", "Research Keywords Analysis"))
        _doc.preamble.append(Command("author", author_name))
        _doc.preamble.append(Command("date", Command("today")))
        _doc.packages.append(NoEscape(r"\usepackage{graphicx}"))
        _doc.packages.append(NoEscape(r"\usepackage{float}"))
        _doc.append(NoEscape(r"\maketitle"))

        self.doc = _doc

    def add_introduction(self):
        # TODO: Link to online tool
        # TODO: Describe project
        with self.doc.create(Section("Introduction")):
            self.doc.append(
                "This document contains suggestions on how to combine concepts that appeared in your work"
                + " with other concepts such that the combination is new, i.e. the two concepts forming the combination"
                + " have not been mentioned together in an abstract."
            )

    def add_image(self, path: Path, caption: str):
        with self.doc.create(Figure(position="H")) as image_figure:
            image_figure.add_image(str(path.name))
            image_figure.add_caption(caption)

    def add_keywords_section(self):
        logger.info("Adding Keywords Section")
        with self.doc.create(Section("Keyword Analysis")):
            #### WORDCLOUD
            wordcloud = Path(self.base_dir / "wordcloud.png")
            self.analyser.save_world_cloud(to=wordcloud, mode="interesting")
            self.add_image(wordcloud, "Wordcloud of interesting concepts (count > 2).")

            #### EMBEDDINGS MAP
            embeddings_map = Path(self.base_dir / "embeddings_map.pdf")
            self.analyser.save_map_with_interesting_concepts(
                to=embeddings_map, fig_size=1600
            )
            self.add_image(
                embeddings_map, "2D Embeddings Map of your interesting concepts."
            )

            #### LIST of interesting keywords
            self.doc.append(
                "The following concepts were extracted using a fine-tuned version of 'LlaMa-2-13B' from abstracts of your published papers (count > 2):\n"
            )

            interesting_concepts = sorted(
                analyser.interesting_concepts.items(), key=lambda x: x[1], reverse=True
            )
            with self.doc.create(Itemize()) as itemize:
                for keyword, count in interesting_concepts:
                    itemize.add_item(f"{keyword}: ({count})")

    def add_suggestions_section(self, top_k: int = 50):
        logger.info("Adding Suggestions Section")
        with self.doc.create(Section("Suggestions of Research Directions")):
            with self.doc.create(
                Subsection(f"Own Concepts Combinations (Best {top_k})")
            ):
                with self.doc.create(Itemize()) as itemize:
                    for concept_1, concept_2, score in self.analyser.own_predictions[
                        :top_k
                    ]:
                        itemize.add_item(
                            NoEscape(
                                f"\\textbf{{{concept_1}}} and {concept_2}: {round(score, 4)}"
                            )
                        )

            with self.doc.create(
                Subsection(f"Other Concepts Combinations (Best {top_k})")
            ):
                with self.doc.create(Itemize()) as itemize:
                    for concept_1, concept_2, score in analyser.other_predictions[
                        :top_k
                    ]:
                        itemize.add_item(
                            NoEscape(
                                f"\\textbf{{{concept_1}}} and {concept_2}: {round(score, 4)}"
                            )
                        )

    def add_highly_connective_predictions(self, top_k: int, threshold: float):
        logger.info("Adding Highly Connective Predictions Section")
        with self.doc.create(
            Section(
                f"Top {top_k} Concepts with most highest scored connections to your own concepts (score >= {threshold})"
            )
        ):
            highly_connective_predictions = list(
                analyser.get_highly_connective_predictions(threshold=threshold).items()
            )[:top_k]
            for concept, suggestions in highly_connective_predictions:
                with self.doc.create(Subsection(f"Recommendation: {concept}")):
                    self.doc.append("To be combined with your work on:")
                    with self.doc.create(Itemize()) as itemize:
                        for own_concept, score in suggestions:
                            itemize.add_item(
                                NoEscape(f"{own_concept}: {round(score, 4)}")
                            )

    def add_potentially_interesting_predictions(
        self, top_k: int, max_degree, min_dist, max_dist
    ):
        logger.info(
            f"Adding Potentially Interesting Predictions Section for {max_degree} max degree and {min_dist} <= distance <= {max_dist}"
        )

        potentially_interesting_connections = (
            analyser.get_potentially_interesting_predictions(
                max_degree, min_dist, max_dist
            )[:top_k]
        )

        with self.doc.create(
            Section(
                f"Top {top_k} Concepts that have other concept with degree <= {max_degree} and {min_dist} <= distance <= {max_dist}"
            )
        ):
            with self.doc.create(Itemize()) as itemize:
                for (
                    concept_own,
                    concept_other,
                    score,
                    concept_dist,
                ) in potentially_interesting_connections:
                    itemize.add_item(
                        NoEscape(
                            f"\\textbf{{{concept_own}}} and {concept_other}: {round(score, 4)} (distance: {round(concept_dist, 1)})"
                        )
                    )

    def add_sources(self):
        logger.info("Adding Sources Section")
        with self.doc.create(Section("Used Papers")):
            self.doc.append(
                "The following papers were considered for concept extraction:"
            )

            all_titles = self.analyser.df["title"].tolist()

            with self.doc.create(Itemize()) as itemize:
                for title in all_titles:
                    only_ascii = "".join([char for char in title if ord(char) < 128])
                    itemize.add_item(only_ascii)

    def add_map_of_materials_science_map(self):
        logger.info("Adding Map of Materials Science Section")

        map_of_materials_science = Path(self.base_dir / "map_of_materials_science.pdf")
        self.analyser.save_map_with_important_landmarks(
            to=map_of_materials_science, panel_size=1, fig_size=1600
        )

        with self.doc.create(Section("Map of Materials Science")):
            self.add_image(
                map_of_materials_science,
                "Map of Materials Science with important landmarks",
            )

    def generate_pdf(self, output: Path, clean_tex: bool = False):
        logger.info(f"Generating PDF to {output}")
        self.doc.generate_pdf(str(output), clean_tex=clean_tex)


generation_base = Path("report/pdf/generation")
prediction_base = generation_base / "predictions"
if __name__ == "__main__":
    df = pd.read_csv(generation_base / "jasmin_jens_concepts.csv")
    df["llama_concepts"] = df["llama_concepts"].apply(literal_eval)  # make usable

    sources = sorted(set(df["source"].tolist()))  # you can also use manual mode
    for source in sources:
        raw_name = source.split(".")[0]
        author_name = " ".join(name.capitalize() for name in raw_name.split("_"))
        logger.info(f"Generating report for '{author_name}'")
        this_reports_base = generation_base / raw_name
        if this_reports_base.exists():
            logger.warning(f"Skipping {author_name}")
            continue

        predictions = prediction_base / f"{raw_name}.pkl.gz"

        analyser = Analyser(
            generation_base,
            df,
            source,
            predictions=predictions,
            all_embeddings=generation_base / "transformed.pkl.gz",
            occurance_filter=2,
        )
        # analyser.save_map_with_important_landmarks("x.png", panel_size=1)

        r = Report(this_reports_base, author_name, analyser=analyser)
        r.add_introduction()
        r.add_keywords_section()
        r.add_suggestions_section(top_k=25)
        r.add_highly_connective_predictions(top_k=20, threshold=0.999)
        r.add_potentially_interesting_predictions(
            top_k=50, max_degree=100, min_dist=3, max_dist=6
        )
        r.add_map_of_materials_science_map()
        r.add_sources()
        r.generate_pdf(this_reports_base / raw_name, clean_tex=False)

        logger.info(f"Finished report for '{author_name}'\n\n\n")

    # add all reports to a zip file
    with zipfile.ZipFile(generation_base / "all_reports.zip", "w") as zipf:
        for source in sources:
            raw_name = source.split(".")[0]
            this_reports_base = generation_base / raw_name
            pdf_report = this_reports_base / f"{raw_name}.pdf"
            logger.info(f"Adding {pdf_report} to zip file")
            zipf.write(pdf_report, arcname=pdf_report.name)
