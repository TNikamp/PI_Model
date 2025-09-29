import numpy as np

def sigmoid(
    x: float | np.ndarray,
    a: float = 0.0,
    b: float = 1.0,
    c: float = 1.0,
    d: float = 1.0,
    e: float = 0.0,
) -> float | np.ndarray:
    """Function to calculate the sigmoid function.
    The base is a + b / (1 + c * np.exp(d * -(x - e))). When only x is given, the
    function will return the sigmoid function with the default values of a=0, b=1, c=1,
    d=1, e=0 resulting in the sigmoid function 1 / (1 + exp(-x)).

    Args:
        x (float | np.ndarray): input value(s) for the sigmoid function
        a (float, optional): The first parameter. Defaults to 0.
        b (float, optional): The second parameter. Defaults to 1.
        c (float, optional): The third parameter. Defaults to 1.
        d (float, optional): The fourth parameter. Defaults to 1.
        e (float, optional): The fifth parameter. Defaults to 0.

    Returns:
        float | np.ndarray: The sigmoid function value(s) for the input value(s) x
    """
    in_exp = np.clip(d * -(x - e), a_min=-700, a_max=700)
    return a + (b / (1 + c * np.exp(in_exp)))
