def get_column_ids(df, column, team_prefix='home'):
    """Get the column ids for a specific team ('home', 'away', or 'ball').

    Args:
        df (pd.DataFrame): DataFrame with player and ball positions.
        team (str): Team type ('home', 'away', or 'ball').

    Returns:
        list: List of column ids corresponding to the team or ball.
    """
    if column not in ['home', 'away', 'ball']:
        raise ValueError("team must be one of 'home', 'away', or 'ball'")

    x_cols = {col[:-2] for col in df.columns if col.endswith("_x") and col.startswith(column)}
    y_cols = {col[:-2] for col in df.columns if col.endswith("_y") and col.startswith(column)}
    
    return sorted(x_cols & y_cols)
