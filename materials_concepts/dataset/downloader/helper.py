import datetime


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = datetime.datetime.now()
        return self

    def __exit__(self, *args):
        print(
            f"{self.name} time elapsed: {(datetime.datetime.now() - self.start).seconds} seconds..."
        )


def remove_non_ascii(text):
    """
    Removes non-ASCII characters from a string.
    """
    return "".join([char for char in text if ord(char) < 128])


def make_valid_filename(filename):
    """
    Removes invalid characters from a string to create a valid filename.
    """

    filename = remove_non_ascii(filename)

    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "")

    # Remove leading/trailing whitespace and dots
    filename = filename.strip(". ")

    # Truncate the filename to 255 characters (max filename length on most file systems)
    filename = filename[:255]

    filename = filename.replace(" ", "-")

    return filename
