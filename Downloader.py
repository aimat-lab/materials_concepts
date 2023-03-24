import requests
import csv
import pandas as pd


class OADownloader:
    start_page = 1

    def __init__(self, url, fields, per_page, fetch_limit=None, filter=None) -> None:
        self._cache = []
        self.url = url
        self.fields = fields
        self.per_page = per_page
        self.fetch_limit = fetch_limit
        self.filter = filter

    def _get(self, page):
        params = {
            "select": ",".join(self.fields),
            "per-page": self.per_page,
            "page": page,
        }
        if self.filter is not None:
            params["filter"] = self.filter

        response = requests.get(self.url, params=params)
        response.raise_for_status()
        response = response.json()
        count = response["meta"]["count"]

        # page * self.per_page => number of fetched items
        # if number < count, there is more data to fetch
        UPPER_LIMIT = self.fetch_limit or count
        if page * self.per_page < UPPER_LIMIT:  # continue fetching?
            return response["results"] + self._get(page + 1)

        return response["results"]

    def get(self):  # TODO lookup fetch: metadata => build progressbar (tqdm)
        if len(self._cache) == 0:
            self._cache = self._get(self.start_page)
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
