import requests
import csv
import pandas as pd


class OADownloader:
    CURSOR_START = "*"

    def __init__(self, url, fields, per_page, fetch_limit=None, filter=None) -> None:
        self._cache = []
        self.url = url
        self.fields = fields
        self.per_page = per_page
        self.fetch_limit = fetch_limit
        self.filter = filter
        self.cursor = self.CURSOR_START  # init value for OpenAlex

    def _get(self, cursor):
        params = {
            "select": ",".join(self.fields),
            "per-page": self.per_page,
            "cursor": cursor,
        }
        if self.filter is not None:
            params["filter"] = self.filter

        response = requests.get(self.url, params=params)
        response.raise_for_status()
        response = response.json()
        next_cursor = response["meta"]["next_cursor"]
        self.cursor = next_cursor

        if next_cursor:
            return response["results"] + self._get(next_cursor)

        return response["results"]

    def get(self):  # TODO lookup fetch: metadata => build progressbar (tqdm)
        if len(self._cache) == 0:
            self._cache = self._get(self.CURSOR_START)
        return self

    def to_csv(self, filename, converters={}):
        self.to_df(converters).to_csv(filename, index=False)

    def to_df(self, converters={}):
        def _convert_row(row):
            new_row = row.copy()
            for attr, conv_func in converters.items():
                new_row[attr] = conv_func(new_row[attr])
            return new_row

        data = [_convert_row(row) for row in self._cache]
        return pd.DataFrame(data)
