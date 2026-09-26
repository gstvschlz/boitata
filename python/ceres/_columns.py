import numpy as np

from ceres.errors import InvalidInput, MissingColumn


def names(data):
    table = getattr(data, "attributes", data)
    return list(table.column_names) if hasattr(table, "column_names") else list(table.keys())


def column(data, arg, what="values"):
    """`arg`, or the column of `data` it names."""
    if not isinstance(arg, str):
        return arg
    if data is None:
        raise InvalidInput(f'{what} names column "{arg}"; that needs a container')
    try:
        return data[arg]
    except KeyError:
        raise MissingColumn(f'no column "{arg}"; columns: {", ".join(names(data))}') from None


def stack(data, labels=None, columns=None):
    """`data`, an array or named columns (those in `columns` when given), as an ``(n, d)`` float array and labels."""
    if columns is not None:
        data = {c: column(data, c, "columns") for c in columns}
    elif hasattr(data, "attributes"):
        data = data.attributes
    if hasattr(data, "column_names") or hasattr(data, "keys"):
        keys = names(data)
        labels = keys if labels is None else labels
        data = np.column_stack([np.asarray(data[k], dtype=float) for k in keys])
    data = np.array(data, dtype=float)
    labels = [str(j) for j in range(data.shape[1])] if labels is None else list(labels)
    return data, labels
