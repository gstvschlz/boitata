class CeresError(Exception):
    """Base class of every error raised by ceres."""


class InvalidInput(CeresError, ValueError):
    pass


class FileError(CeresError, OSError):
    pass


class MissingColumn(CeresError, KeyError):
    """A column name that the table or container does not have; the message lists those it has."""

    __str__ = Exception.__str__
