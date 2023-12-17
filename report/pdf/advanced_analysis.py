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


def load_compressed(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class Analyser:
    def __init__(
        self,
        _df: pd.DataFrame,
        source: str,
        predictions: Path,
        # degrees: dict[str, int],
        all_embeddings: Path,
        occurance_filter: int = 2,
    ):
        self.source = source

        self.all_embeddings: dict = load_compressed(str(all_embeddings))

        lookup = pd.read_csv("lookup.M.new.csv")
        self.degrees = dict(zip(lookup["concept"], lookup["count"]))

        self.occurance_filter = occurance_filter
        self.df = _df[_df["source"] == source]  # relevant data for source
        self.all_concepts = Counter(
            item for sublist in self.df["llama_concepts"].values for item in sublist
        )
        self.interesting_concepts = {
            concept: count
            for concept, count in self.all_concepts.items()
            if count >= occurance_filter
        }
        logger.info(f"Found {len(self.interesting_concepts)} interesting concepts")

        self.predictions: dict = load_pickle(predictions)
        logger.info(f"Loaded {len(self.predictions)} predictions")

        logger.info("Loading own and other predictions")
        (
            self.own_predictions,
            self.other_predictions,
        ) = self.suggestions_to_own_and_other_concepts()

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

        plt.savefig(to, format="png")

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

    # TODO: Refactor into a function that takes a list of concepts
    # that are highlighted (or dict with colors and lists as values)
    def save_whole_2d_word_embeddings_map(self, to: Path, fig_size: int = 1200):
        fig = go.Figure()

        # Add all embeddings as a gray scatter plot
        # You can extract the points for the gray embeddings as lists, similar to how you did before
        gray_x = [
            point[0]
            for key, point in self.all_embeddings.items()
            if key not in self.interesting_concepts
        ]
        gray_y = [
            point[1]
            for key, point in self.all_embeddings.items()
            if key not in self.interesting_concepts
        ]

        fig.add_trace(
            go.Scatter(
                x=gray_x,
                y=gray_y,
                mode="markers",
                name="Other embeddings",
                marker=dict(color="#BEBADA"),
                textposition="top center",
            )
        )

        # Now add the interesting embeddings on top with text annotations
        interesting_x = [
            point[0]
            for key, point in self.all_embeddings.items()
            if key in self.interesting_concepts
        ]
        interesting_y = [
            point[1]
            for key, point in self.all_embeddings.items()
            if key in self.interesting_concepts
        ]
        interesting_text = [
            key
            for key in self.all_embeddings.keys()
            if key in self.interesting_concepts
        ]

        fig.add_trace(
            go.Scatter(
                x=interesting_x,
                y=interesting_y,
                mode="markers+text",
                name="Interesting embeddings",
                text=interesting_text,
                textposition="top center",
            )
        )

        logger.info("Saving whole embeddings map")
        fig.write_image(str(to), width=fig_size, height=fig_size)

    def save_map_with_important_landmarks(
        self, to: Path, panel_size=2, fig_size: int = 1200
    ):
        winners = {}
        non_winners = defaultdict(list)

        # Process the coordinates and find the winners
        for concept, point in tqdm(self.all_embeddings.items()):
            # Calculate the grid square's indices
            grid_x = int(point[0] // panel_size)
            grid_y = int(point[1] // panel_size)
            grid_key = (grid_x, grid_y)

            # Check if we already have a winner for this square and if this point has a higher degree
            if grid_key not in winners or self.degrees[concept] > self.degrees.get(
                winners.get(grid_key, "nonexistant"), -1
            ):
                winners[grid_key] = concept
            else:
                non_winners[grid_key].append(concept)

        ## PRINT ##
        max_x = max(key[0] for key in non_winners) + 1
        max_y = max(key[1] for key in non_winners) + 1
        for x in range(max_x):
            for y in range(max_y):
                print(f"{len(non_winners.get((x, y), ' ')): 5d}", end=" ")
            print()
        ## END PRINT ##

        # Now that we have the winners, let's separate the points
        winner_points = {
            concept: self.all_embeddings[concept] for concept in winners.values()
        }

        # non_winner_points = {
        #     concept: self.all_embeddings[concept]
        #     for concept in self.all_embeddings
        #     if concept not in winner_points
        # }

        subset_non_winners = []
        for key in tqdm(non_winners):
            current_points = non_winners[key]
            if len(current_points) > 300:
                subset_non_winners.extend(
                    random.sample(current_points, round(0.2 * len(current_points)))
                )
            else:
                subset_non_winners.extend(current_points)

        logger.info(
            f"Found {len(subset_non_winners)} non-winners: {subset_non_winners[:10]}"
        )

        non_winner_points = {
            concept: self.all_embeddings[concept] for concept in subset_non_winners
        }
        logger.info(f"Found {len(non_winner_points)} non-winners")

        # Prepare traces for the Plotly figure
        winner_x = [point[0] for point in winner_points.values()]
        winner_y = [point[1] for point in winner_points.values()]
        winner_text = [concept for concept in winner_points.keys()]

        non_winner_x = [point[0] for point in non_winner_points.values()]
        non_winner_y = [point[1] for point in non_winner_points.values()]

        # Create the figure
        fig = go.Figure()

        # Add non-winners as a basic colored trace
        fig.add_trace(
            go.Scatter(
                x=non_winner_x,
                y=non_winner_y,
                mode="markers",
                name="Non-winners",
                hovertext=list(non_winner_points.keys()),
                marker=dict(color="#BEBADA"),
            )
        )

        # Add winners with red color and text
        fig.add_trace(
            go.Scatter(
                x=winner_x,
                y=winner_y,
                mode="markers+text",
                name="Winners",
                text=winner_text,
                textposition="top center",
                hovertext=winner_text,
                hoverinfo="text",
                marker=dict(color="red"),
            )
        )

        fig.update_layout(
            title=f"Map of Materials Science (Reduced to {len(non_winner_points) + len(winner_points)} / {len(self.all_embeddings)} concepts )",
        )

        logger.info("Saving whole embeddings map w/ landmarks")
        as_html = fig.to_html(
            full_html=True,
            include_plotlyjs=True,
            default_height="100vh",
            default_width="80vw",
        )
        # with open("map.html", "w") as f:
        #     f.write(as_html)
        fig.write_image(str(to), width=fig_size, height=fig_size)

    def _suggestions_to_own_concepts(self):
        for concept, predictions in self.predictions.items():
            for prediction in predictions:
                if prediction["concept"] in self.all_concepts:
                    yield concept, prediction["concept"], prediction["score"]

    def _suggestions_to_other_concepts(self):
        for concept, predictions in self.predictions.items():
            for prediction in predictions:
                if prediction["concept"] not in self.all_concepts:
                    yield concept, prediction["concept"], prediction["score"]

    def suggestions_to_own_and_other_concepts(self):
        sort_by_score = lambda x: x[2]
        return (
            sorted(
                self._suggestions_to_own_concepts(), key=sort_by_score, reverse=True
            ),
            sorted(
                self._suggestions_to_other_concepts(), key=sort_by_score, reverse=True
            ),
        )

    def get_highly_connective_predictions(self, threshold=0.995):
        logger.info(f"Retrieving predictions with score > {threshold}")
        all_predictions = self.own_predictions + self.other_predictions

        all_predictions = [
            (x, y, score) for x, y, score in all_predictions if score > threshold
        ]
        logger.info(f"Found {len(all_predictions)} predictions (score > {threshold})")

        suggested_new_keywords = defaultdict(list)
        for own_concept, new_concept, score in all_predictions:
            suggested_new_keywords[new_concept].append((own_concept, score))

        new_keyword_matches = {
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
        self, max_other_degree, min_distance, max_distance
    ) -> list[tuple[str, str, float, float]]:
        all_predictions = self.own_predictions + self.other_predictions

        # pred = set(other for _, other, _ in all_predictions)
        # logger.info(f"Found {len(pred)} unique predictions")
        # embedded_concepts = set(self.all_embeddings.keys())
        # logger.info(f"Found {len(embedded_concepts)} embedded concepts")
        # remaining = pred - embedded_concepts
        # logger.info(f"Found {len(remaining)} remaining concepts: {remaining}")
        # TODO: type heterojunction is missing

        after_degree_filter = [
            (x, y, score)
            for x, y, score in all_predictions
            if self.degrees[y] <= max_other_degree
        ]

        after_distance_filter = [
            (x, y, score, concept_dist)
            for x, y, score in after_degree_filter
            if min_distance <= (concept_dist := self.emb_distance(x, y)) <= max_distance
        ]

        return sorted(after_distance_filter, key=lambda x: x[2], reverse=True)

    def emb_distance(self, concept_1: str, concept_2: str):
        try:
            emb_1 = self.all_embeddings[concept_1]
            emb_2 = self.all_embeddings[concept_2]
        except KeyError:
            return -1
        return ((emb_1[0] - emb_2[0]) ** 2 + (emb_1[1] - emb_2[1]) ** 2) ** 0.5


class Report:
    def __init__(self, author_name: str, analyser: Analyser):
        logger.info("Initializing Report")
        self.analyser = analyser

        _doc = Document("basic")
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
                "This document presents a comprehensive analysis of research concepts."
            )

    def add_image(self, path: Path, caption: str):
        with self.doc.create(Figure(position="H")) as image_figure:
            image_figure.add_image(str(path))
            image_figure.add_caption(caption)

    def add_keywords_section(self):
        logger.info("Adding Keywords Section")
        with self.doc.create(Section("Keyword Analysis")):
            #### WORDCLOUD
            wordcloud = Path("wordcloud.png")
            self.analyser.save_world_cloud(to=wordcloud, mode="interesting")
            self.add_image(wordcloud, "Wordcloud of interesting concepts (count > 2).")

            #### EMBEDDINGS MAP
            embeddings_map = Path("embeddings_map.pdf")
            self.analyser.save_whole_2d_word_embeddings_map(
                to=embeddings_map, fig_size=1600
            )
            self.add_image(embeddings_map, "2D Embeddings Map of interesting concepts.")

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
                    itemize.add_item(title)

    def add_map_of_materials_science_map(self):
        logger.info("Adding Map of Materials Science Section")

        map_of_materials_science = Path("map_of_materials_science.pdf")
        self.analyser.save_map_with_important_landmarks(
            to=map_of_materials_science, panel_size=1, fig_size=1600
        )

        with self.doc.create(Section("Map of Materials Science")):
            self.add_image(
                map_of_materials_science,
                "Map of Materials Science with important landmarks",
            )

    def generate_pdf(self, output: Path, clean_tex: bool = False):
        self.doc.generate_pdf(str(output), clean_tex=clean_tex)


if __name__ == "__main__":
    author_name = "Pascal Friederich"
    df = pd.read_csv("fixed_concepts.csv")
    df["llama_concepts"] = df["llama_concepts"].apply(literal_eval)  # make usable
    analyser = Analyser(
        df,
        "pascal_friederich.txt",
        predictions=Path("friederich.pickle"),
        all_embeddings=Path("transformed.pkl.gz"),
        occurance_filter=2,
    )
    # analyser.save_map_with_important_landmarks("x.png", panel_size=1)

    r = Report("Pascal Friederich", analyser=analyser)
    # r.add_introduction()
    # r.add_keywords_section()
    # r.add_suggestions_section(top_k=25)
    # r.add_highly_connective_predictions(top_k=20, threshold=0.999)
    # r.add_map_of_materials_science_map()
    r.add_potentially_interesting_predictions(
        top_k=30, max_degree=100, min_dist=3, max_dist=6
    )
    # r.add_sources()
    r.generate_pdf(Path("research_keyword_analysis"), clean_tex=True)
