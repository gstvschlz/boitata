class CeresError(Exception):
    """Base class of every error raised by ceres."""


class InvalidInput(CeresError, ValueError):
    pass


class FileError(CeresError, OSError):
    pass
