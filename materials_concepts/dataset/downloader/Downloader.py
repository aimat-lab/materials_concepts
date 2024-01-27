import requests
import csv
import pandas as pd
from tqdm import tqdm


class Converter:
    def __init__(self, converter_funcs):
        self.converter_funcs = converter_funcs

    def _convert_row(self, row):
        new_row = row.copy()
        for attr, conv_func in self.converter_funcs.items():
            new_row[attr] = conv_func(new_row[attr])
        return new_row

    def apply(self, data):
        return [self._convert_row(row) for row in data]


class FileHandler:
    def __init__(self, filename, fields, converter, flush_threshold=1000):
        self.filename = filename
        self.fields = fields
        self.converter = converter
        self.flush_threshold = flush_threshold

    def __enter__(self):
        self.file = open(self.filename, "w")
        self.flush_counter = 0
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
        self.writer.writeheader()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def handle(self, data):
        conv_data = self.converter.apply(data)
        self.writer.writerows(conv_data)

        self.flush_counter += len(conv_data)
        if self.flush_counter >= self.flush_threshold:
            self.file.flush()
            self.flush_counter = 0


class InMemoryHandler:
    def __init__(self):
        self.data = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Nothing needs to be cleaned up since the data
        # should be available after exiting the with
        pass

    def handle(self, data):
        self.data.extend(data)

    def to_csv(self, filename, converter=None):
        self.to_df(converter).to_csv(filename, index=False)

    def to_df(self, converter):
        if converter is None:
            return pd.DataFrame(self.data)

        conv_data = converter.apply(self.data)
        return pd.DataFrame(conv_data)


class OADownloader:
    CURSOR_START = "*"  # init value for OpenAlex

    def __init__(
        self, url, fields, handler=None, per_page=200, fetch_limit=None, filter=None
    ) -> None:
        self._cache = []
        self.url = url
        self.fields = fields
        self.per_page = per_page
        if handler is None:
            handler = InMemoryHandler()
        self.handler = handler
        self.fetch_limit = fetch_limit
        self.filter = filter
        self.cursor = self.CURSOR_START

    def _get_params(self, cursor):
        params = {
            "select": ",".join(self.fields),
            "per-page": self.per_page,
            "cursor": cursor,
        }
        if self.filter is not None:
            params["filter"] = self.filter

        return params

    def perform_request(self, cursor):
        params = self._get_params(cursor)
        response = requests.get(self.url, params=params)
        response.raise_for_status()
        response = response.json()
        return response

    def get(self):
        with tqdm(total=self.fetch_limit) as pbar:
            with self.handler as handler:
                while self.cursor:
                    req_data = self.perform_request(self.cursor)

                    # handle data (e.g. store in memory, write to csv, etc.)
                    handler.handle(req_data["results"])

                    # update progress bar
                    count = len(req_data["results"])
                    pbar.update(count)

                    # update cursor
                    next_cursor = req_data["meta"]["next_cursor"]
                    self.cursor = next_cursor

        return self.handler
