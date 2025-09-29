import numpy as np
import pandas as pd
from scipy.constants import g


def calculate_metabolic_power(df, players, framerate=25, eccr=3.6, percentile_cap=99, smooth=True, window=25, silent=True):
    """
    Calculate the frame-wise metabolic power for each player based on their velocity 
    and acceleration using the di Prampero model.

    This function implements the metabolic power model from di Prampero & Osgnach (2018) 
    and Minetti & Parvei (2018), estimating energy expenditure over time using 
    spatiotemporal data (velocity and acceleration). The result is optionally smoothed 
    using a moving average.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing velocity and acceleration time series for players.
        Each player's data must be in columns named like '<player>_velocity' and 
        '<player>_acceleration'.

    players : list of str
        A list of player identifiers corresponding to the prefixes of column names in 
        the DataFrame.

    framerate : int, optional (default=25)
        The sampling rate of the data in Hz (frames per second). This is used to scale 
        cumulative metrics (not currently returned).

    eccr : float, optional (default=3.6)
        Energy cost of constant-speed running (J/kg·m). Can be adjusted to account 
        for different surfaces or calibration needs.

    smooth : bool, optional (default=True)
        If True, applies a moving average smoothing to the metabolic power time series.

    window : int, optional (default=25)
        The size of the moving average window used for smoothing (in frames). Only 
        relevant if `smooth=True`.

    Returns
    -------
    results : dict
        A dictionary mapping each player name to another dictionary containing:
            - 'metabolic_power': np.ndarray of frame-wise metabolic power (W/kg).
        If data is missing for a player, the value will be `None`.

    Notes
    -----
    This implementation follows the original di Prampero metabolic model, including:
        - Equivalent slope (ES) and equivalent mass (EM) calculations
        - Polynomial-based energy cost of walking (ECW)
        - Slope-dependent energy cost of running (ECR)
        - A walk/run transition based on slope
    """
    # Constants
    K = 0.0037
    RUNNING_TRANSITION_COEFF = np.array([-107.05, 113.13, -1.13, -15.84, -1.7, 2.27])
    ECW_ES_CUTOFFS = np.array([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4])
    ECW_POLY_COEFF = np.array([
        [0.28, -1.66, 3.81, -3.96, 4.01],
        [0.03, -0.15, 0.98, -2.25, 3.14],
        [0.69, -3.21, 5.94, -5.07, 2.79],
        [1.25, -6.57, 13.14, -11.15, 5.35],
        [0.68, -4.17, 10.17, -10.31, 8.66],
        [3.80, -14.91, 22.94, -14.53, 11.24],
        [44.95, -122.88, 126.94, -57.46, 21.39],
        [94.62, -213.94, 184.43, -68.49, 25.04],
    ])
     
    def calc_es(vel, acc):
        return (acc / g) + ((K * vel**2) / g)

    def calc_em(es):
        return np.sqrt(es**2 + 1)

    def calc_v_trans(es):
        es_power = np.stack(
            [es**5, es**4, es**3, es**2, es, np.ones_like(es)], axis=-1
        )
        return np.matmul(es_power, RUNNING_TRANSITION_COEFF)

    def is_running(vel, es):
        v_trans = calc_v_trans(es)
        return (vel >= v_trans) | (vel > 2.5)

    def get_interpolation_weights(es):
        T = len(es)
        W = np.zeros((T, len(ECW_ES_CUTOFFS)))
        idxs = ECW_ES_CUTOFFS.searchsorted(es)
        mask = (idxs > 0) & (idxs < 8)

        W[np.arange(T)[mask], idxs[mask] - 1] = (ECW_ES_CUTOFFS[idxs[mask]] - es[mask]) * 10
        W[np.arange(T)[mask], idxs[mask]] = (es[mask] - ECW_ES_CUTOFFS[idxs[mask] - 1]) * 10

        W[idxs == 0, 0] = 1
        W[idxs == 8, 7] = 1
        return W

    def calc_ecw(es, vel, em):
        W = get_interpolation_weights(es)
        WC = np.matmul(W, ECW_POLY_COEFF)
        V = np.stack([vel**4, vel**3, vel**2, vel, np.ones_like(vel)], axis=-1)
        return np.sum(WC * V, axis=-1) * em

    def calc_ecr(es, em):
        cng = lambda x: -8.34 * x + eccr * np.exp(13 * x)
        cpg = lambda x: 39.5 * x + eccr * np.exp(-4 * x)
        cost = np.where(es < 0, cng(es), cpg(es))
        return cost * em

    def calc_ecl(es, vel, em):
        running = is_running(vel, es)
        ecl = calc_ecw(es, vel, em)
        ecl[running] = calc_ecr(es[running], em[running])
        return ecl

    def calc_metabolic_power(vel, acc):
        es = calc_es(vel, acc)
        em = calc_em(es)
        ecl = calc_ecl(es, vel, em)
        metp = ecl * vel
    
        # Suppress unrealistic power at very low speeds
        metp[vel < 0.5] = 0
    
        # Clipping with custom percentile cap
        cap = np.percentile(metp, percentile_cap)
        if not silent:
            print(f"{percentile_cap}th percentile metabolic power cap: {cap:.2f} W/kg")
        metp = np.clip(metp, 0, cap)
    
        return metp

    def smooth_metabolic_power(values, window=25):
        return pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().to_numpy()

    results = {}
    for player in players:
        v_col = f"{player}_velocity"
        a_col = f"{player}_acceleration"
        if v_col in df.columns and a_col in df.columns:
            vel = df[v_col].values
            acc = df[a_col].values
            met_power = calc_metabolic_power(vel, acc)

            if smooth:
                met_power = smooth_metabolic_power(met_power, window=window)
            
            results[player] = {
                "metabolic_power": met_power,
                }
        else:
            results[player] = None  # Missing data

    return results
