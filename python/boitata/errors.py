class BoitataError(Exception):
    """Base class of every error raised by boitata."""


class InvalidInput(BoitataError, ValueError):
    pass


class FileError(BoitataError, OSError):
    pass


class MissingColumn(BoitataError, KeyError):
    """A column name that the table or container does not have; the message lists those it has."""

    __str__ = Exception.__str__
