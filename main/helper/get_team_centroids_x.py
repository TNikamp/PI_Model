import pandas as pd
import numpy as np  # Ensure this is imported at the top
from helper.get_column_ids import *

def classify_defensive_block(df):
    """
    Classify the defensive block height of both teams based on tracking data.

    The function calculates the centroid of each team’s outfield players 
    (excluding the deepest player, typically the goalkeeper or last defender),
    and then determines how far up the pitch the defensive block is set. 
    Distances are then classified into categorical defensive block types:
        - 0 = Low block (≤ 20m from own goal)
        - 1 = Mid block (> 20m and ≤ 60m)
        - 2 = High block (> 60m)
        - "Unknown" = Could not compute due to missing data

    Args:
        df (pd.DataFrame): Tracking data containing player x-coordinates. 
            Expected structure:
            - Columns for each player's x-coordinate (e.g., "home_1_x", "away_5_x").
            - A helper function `get_column_ids` is assumed to extract player IDs 
              for "home" and "away".

    Returns:
        pd.DataFrame: Original dataframe with two additional columns:
            - "defensive_block_home" (int or str): Defensive block classification 
              for the home team (0, 1, 2, or "Unknown").
            - "defensive_block_away" (int or str): Defensive block classification 
              for the away team (0, 1, 2, or "Unknown").
    """

    # Standard values, replace with metadata values if available
    pitch_length = 105
    home_goal_x = -pitch_length / 2
    away_goal_x = pitch_length / 2

    # Get player IDs for home and away teams
    home_players = get_column_ids(df, column="home", team_prefix="home")
    away_players = get_column_ids(df, column="away", team_prefix="away")

    # Create lists of x-coordinate column names for each team
    home_x_cols = [f"{pid}_x" for pid in home_players]
    away_x_cols = [f"{pid}_x" for pid in away_players]
    
    # Function to compute team centroid excluding the deepest player
    def compute_centroid_excluding_deepest(row, x_cols, team_side='home'):
        # Extract positions for all valid players
        positions = {col: row[col] for col in x_cols if pd.notna(row[col])}
        if not positions:
            return float('nan')  # Return NaN if no valid positions
        
        # Determine deepest player
        if team_side == 'home':
            deepest = min(positions.items(), key=lambda item: item[1])[0]
        else:
            deepest = max(positions.items(), key=lambda item: item[1])[0]
        
        # Compute median of remaining players
        valid_positions = [v for k, v in positions.items() if k != deepest and pd.notna(v)]
        return np.median(valid_positions) if valid_positions else float('nan')

    # Compute intermediate columns
    home_centroid_x = df.apply(lambda row: compute_centroid_excluding_deepest(row, home_x_cols, 'home'), axis=1)
    away_centroid_x = df.apply(lambda row: compute_centroid_excluding_deepest(row, away_x_cols, 'away'), axis=1)

    # Compute distance of defensive block from own goal (52.5 is half pitch)
    home_block_distance = home_centroid_x + 52.5
    away_block_distance = 52.5 - away_centroid_x

    # Function to classify defensive block into categories
    def classify_block(d):
        if pd.isna(d):
            return 'Unknown'
        elif d <= 20:
            return 0
        elif d <= 60:
            return 1
        else:
            return 2

    # Apply classification to each team
    home_defensive_block = home_block_distance.apply(classify_block)
    away_defensive_block = away_block_distance.apply(classify_block)

    # Concatenate all new columns at once to avoid fragmentation
    new_cols = pd.concat([
        home_defensive_block.rename('defensive_block_home'),
        away_defensive_block.rename('defensive_block_away'),
    ], axis=1)

    df = pd.concat([df, new_cols], axis=1).copy()  # de-fragment

    return df



