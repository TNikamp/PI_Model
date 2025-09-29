import numpy as np
import pandas as pd

def pti_model(series):
    """
    Compute the Pressing/Interception Intensity (PTI) for attacking players against defenders.

    This function implements an enhanced version of the pressing intensity model by 
    Joris Bekkers, which estimates the probability that a defending player can 
    intercept or apply pressure to an attacking player. Enhancements added to the 
    original model include:
        - Sideline pressure: adds additional pressure when attacking players are near 
          the pitch boundaries.
        - Non-moving defender penalty: defenders below a certain speed threshold are 
          considered less effective and their PTI contributions are set to zero.

    Args:
        series (pd.Series): A single-frame tracking data series with the following:
            - "player_possession" (str): Identifier of the player currently in possession 
              (e.g., "home_10").
            - Player x/y positions: "{team}_{jersey}_x", "{team}_{jersey}_y"
            - Player velocities: "{team}_{jersey}_vx", "{team}_{jersey}_vy"
              for all home and away players on the pitch.

    Returns:
        pd.DataFrame: A dataframe containing the PTI values for all attacking players:
            - Columns are defending players’ jersey numbers.
            - Rows are attacking players’ jersey numbers.
            - "Sideline": maximum PTI contribution from proximity to the top or bottom 
              sideline (additional pressure).
            - "Total_Pressure": sum of all PTI contributions including sideline effect.

    Notes:
        - Reaction time, maximum object speed, and other thresholds are set internally.
        - Defenders moving slower than a threshold speed are considered ineffective 
          for pressing (PTI set to 0 for those defenders).
        - Sideline contribution is only added for attacking players near the top or 
          bottom boundaries and moving faster than a minimum speed.
        - PTI computation relies on helper functions `time_to_intercept` and 
          `probability_to_intercept` for estimating interception probabilities.
        - Player identifiers are automatically extracted from column names for both 
          possession and defending teams.

    References:
        Bekkers, J., "Quantifying pressing intensity in soccer using tracking data", 2021.
    """
    
    # Determine which team is in possession and which is defending
    player_possession = series['player_possession']
    possession_team = player_possession.split('_')[0]
    defending_team = "away" if possession_team == "home" else 'home'
    
    # Helper function to extract base column names for players (e.g., 'home_10')
    def get_column_ids(columns, team_prefix):
        return sorted(set(
            col.rsplit("_", 1)[0]
            for col in columns
            if col.startswith(f"{team_prefix}_") and col.rsplit("_", 1)[-1] in {"x", "y"}
        ))
    
    # Get initial lists of player IDs
    possession_col_ids = get_column_ids(series.index, team_prefix=possession_team)
    defending_col_ids = get_column_ids(series.index, team_prefix=defending_team)
    
    # Filter to only players with valid 'x' and 'y' data (on the pitch)
    def valid_player_ids(col_ids):
        return [
            col_id for col_id in col_ids
            if not pd.isnull(series.get(f"{col_id}_x")) and not pd.isnull(series.get(f"{col_id}_y"))
        ]
    
    possession_col_ids = valid_player_ids(possession_col_ids)
    defending_col_ids = valid_player_ids(defending_col_ids)
    
    # Extract positions and velocities
    def extract_player_data(col_ids, suffixes):
        return np.stack([
            series[[f"{col_id}_{sfx}" for col_id in col_ids]].values
            for sfx in suffixes
        ]).T

    # Extract positoins (x,y) and velocities (vx,vy) for both teams
    p1 = extract_player_data(defending_col_ids, ["x", "y"])
    p2 = extract_player_data(possession_col_ids, ["x", "y"])
    v1 = extract_player_data(defending_col_ids, ["vx", "vy"])
    v2 = extract_player_data(possession_col_ids, ["vx", "vy"])

    # Ensure numpy arrays with float64 for numerical stability
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    # Model parameters
    reaction_time = 0.7             # seconds
    max_object_speed = 13.0         # m/s
    speed_threshold = 2.5           # minimum defender speed to contribute
    movement_threshold = 1.5        # minimum attacker speed for sideline pressure
    sideline_threshold = 6.0        # distance to sideline for extra pressure
    tti_sigma = 0.45                # standard deviation for probability calculation
    tti_time_threshold = 1.5        # seconds for PTI probability calculation

    # Compute speed of defenders
    v1_speed = np.linalg.norm(v1, axis=1)
    low_speed_mask = v1_speed < speed_threshold # Mask for inactive defenders

    # Compute time-to-intercept (TTI) matrix
    tti = time_to_intercept(p1, p2, v1, v2, reaction_time, max_object_speed)

    # Convert TTI to probability to intercept (PTI)
    pti = probability_to_intercept(tti, tti_sigma, tti_time_threshold)

    # Extract jersey numbers from column IDs
    def extract_jersey_numbers(col_ids):
        return [int(col_id.split("_")[1]) for col_id in col_ids if f"{col_id}_x" in series and not pd.isnull(series[f"{col_id}_x"])]

    defending_jerseys = extract_jersey_numbers(defending_col_ids)
    attacking_jerseys = extract_jersey_numbers(possession_col_ids)

    # Ensure PTI shape matches the number of attacking and defending players
    assert pti.shape == (len(attacking_jerseys), len(defending_jerseys)), \
        f"PTI shape {pti.shape} doesn't match attacking {len(attacking_jerseys)} and defending {len(defending_jerseys)}"

    # Set PTI of slow-moving defenders to zero
    pti[:, low_speed_mask] = 0.0
    
    # Build dataframe with attacking players as rows and defending players as columns
    pti_df = pd.DataFrame(pti, columns=defending_jerseys, index=attacking_jerseys).round(4)

    # Sideline pressure contribution
    top_sideline = np.stack([p2[:, 0], np.full_like(p2[:, 0], 34)], axis=1)
    bottom_sideline = np.stack([p2[:, 0], np.full_like(p2[:, 0], -34)], axis=1)
    zero_vel = np.zeros_like(p2)

    # Compute TTI and PTI for sideline pressure
    tti_top = time_to_intercept(top_sideline, p2, zero_vel, v2, reaction_time, max_object_speed)
    tti_bot = time_to_intercept(bottom_sideline, p2, zero_vel, v2, reaction_time, max_object_speed)

    pti_top = probability_to_intercept(tti_top, tti_sigma, tti_time_threshold)
    pti_bot = probability_to_intercept(tti_bot, tti_sigma, tti_time_threshold)

    # Maximum sideline pressure per attacking player
    pti_sideline_top = pti_top.max(axis=1)
    pti_sideline_bot = pti_bot.max(axis=1)

    # Apply minimum movement threshold: stationary attackers get no sideline pressure
    v2_speed = np.linalg.norm(v2, axis=1)
    moving_mask = v2_speed >= movement_threshold
    pti_sideline_top[~moving_mask] = 0.0
    pti_sideline_bot[~moving_mask] = 0.0

     # Apply proximity threshold: only add pressure if near sideline
    top_proximity_mask = p2[:, 1] > 34 - sideline_threshold
    bottom_proximity_mask = p2[:, 1] < -34 + sideline_threshold
    pti_sideline_top[~top_proximity_mask] = 0.0
    pti_sideline_bot[~bottom_proximity_mask] = 0.0

    # Combine sideline pressure contributions (top or bottom)
    pti_sideline_combined = np.maximum(pti_sideline_top, pti_sideline_bot)

    # Add sideline and total pressure columns
    pti_df["Sideline"] = pti_sideline_combined.round(4)
    pti_df["Total_Pressure"] = pti_df.sum(axis=1)

    return pti_df

def probability_to_intercept(
    time_to_intercept: np.ndarray, tti_sigma: float, tti_time_threshold: float
):
    exponent = (
        -np.pi / np.sqrt(3.0) / tti_sigma * (tti_time_threshold - time_to_intercept)
    )
    # we take the below step to avoid Overflow errors, np.exp does not like values above ~700.
    # exp(25) should already result in p ~ 0.000%
    exponent = np.clip(exponent, -700, 700)
    p = 1 / (1.0 + np.exp(exponent))
    return p


def time_to_intercept(
    p1: np.ndarray,
    p2: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    reaction_time: float,
    max_object_speed: float,
) -> np.ndarray:
    """
    BSD 3-Clause License

    Copyright (c) 2025 [UnravelSports]

    See: https://opensource.org/licenses/BSD-3-Clause

    This project includes code and contributions from:
        - Joris Bekkers (UnravelSports)

    Permission is hereby granted to redistribute this software under the BSD 3-Clause License, with proper attribution
    ----------

    Calculate the Time-to-Intercept (TTI) pressing intensity for a group of players.

    This function estimates the time required for Player 1 to press Player 2 based on their
    positions, velocities, reaction times, and maximum running speed. It calculates an
    interception time matrix for all possible pairings of players.

    Parameters
    ----------
    p1 : ndarray
        An array of shape (n, 2) representing the positions of Pressing Players.
        Each row corresponds to a player's position as (x, y) coordinates.

    p2 : ndarray
        An array of shape (m, 2) representing the positions of Players on the In Possession Team (potentially including the ball location)
        Each row corresponds to a player's position as (x, y) coordinates.

    v1 : ndarray
        An array of shape (n, 2) representing the velocities corresponding to v1. Each row corresponds
        to a player's velocity as (vx, vy).

    v2 : ndarray
        An array of shape (m, 2) representing the velocities corresponding to p2. Each row corresponds
        to a player's velocity as (vx, vy).

    reaction_time : float
        The reaction time of p1'ss (in seconds) before they start moving towards p2's.

    max_velocity : float
        The maximum running velocity of Player 1 (in meters per second).

    Returns
    -------
    t : ndarray
        A 2D array of shape (m, n) where t[i, j] represents the time required for Player 1[j]
        to press Player 2[i].
    """
    u = (p1 + v1) - p1  # Adjusted velocity of Pressing Players
    d2 = p2 + v2  # Destination of Players Under Pressure

    v = (
        d2[:, None, :] - p1[None, :, :]
    )  # Relative motion vector between Pressing Players and Players Under Pressure

    u_mag = np.linalg.norm(u, axis=-1)  # Magnitude of Pressing Players velocity
    v_mag = np.linalg.norm(v, axis=-1)  # Magnitude of relative motion vector
    dot_product = np.sum(u * v, axis=-1)

    epsilon = 1e-10  # We add epsilon to avoid dividing by zero (which throws a warning)
    angle = np.arccos(dot_product / (u_mag * v_mag + epsilon))

    r_reaction = (
        p1 + v1 * reaction_time
    )  # Adjusted position of Pressing Players after reaction time
    d = d2[:, None, :] - r_reaction[None, :, :]  # Distance vector after reaction time

    t = (
        u_mag * angle / np.pi  # Time contribution from angular adjustment
        + reaction_time  # Add reaction time
        + np.linalg.norm(d, axis=-1) / max_object_speed
    )  # Time contribution from running

    return t