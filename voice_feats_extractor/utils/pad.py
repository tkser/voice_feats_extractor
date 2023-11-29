import numpy as np


def pad_1d(
    x: np.ndarray | list[np.ndarray],
    pad: float = 0.0,
) -> np.ndarray:
    """Pad 1D array(s) with a constant value.

    Args:
        x (np.ndarray | list[np.ndarray]): 1D array(s) to pad.
        pad (float): Constant value to pad.

    Returns:
        np.ndarray: Padded 1D array(s).
    """

    def pad_data(x: np.ndarray, length: int) -> np.ndarray:
        x_padded = np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=pad)
        return x_padded

    max_len = max(len(x) for x in x)
    padded = np.stack([pad_data(x, max_len) for x in x])

    return padded


def pad_2d(
    x: np.ndarray | list[np.ndarray],
    max_len: int | None = None,
    pad: float = 0.0,
) -> np.ndarray:
    """Pad 2D array(s) with a constant value.

    Args:
        x (np.ndarray | list[np.ndarray]): 2D array(s) to pad.
        max_len (int | None): Maximum length to pad. If None, the maximum length of input arrays is used.
        pad (float): Constant value to pad.

    Returns:
        np.ndarray: Padded 2D array(s).
    """

    def pad_data(x: np.ndarray, _max_len: int) -> np.ndarray:
        if np.shape(x)[0] > _max_len:
            msg = f"Input array length ({np.shape(x)[0]}) is greater than max_len ({_max_len})."
            raise ValueError(msg)
        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, _max_len - np.shape(x)[0]), mode="constant", constant_values=pad)
        return x_padded[:, :s]

    if max_len:
        output = np.stack([pad_data(x, max_len) for x in x])
    else:
        max_len = max(np.shape(x)[0] for x in x)
        output = np.stack([pad_data(x, max_len) for x in x])

    return output
