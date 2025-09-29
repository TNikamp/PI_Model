import pandas as pd

def get_deliberate_possessions(df, min_players=2, min_frames=125):
    """
    Identifies deliberate possessions where:
    - The same team retains possession for at least `min_frames` frames.
    - At least `min_players` distinct players from that team are involved during that span.
    """
    deliberate_possessions = []
    alive_df = df[df["ball_status"] == "alive"].reset_index(drop=True)
    n = len(alive_df)
    i = min_frames

    while i < n:
        # Check if the last `min_frames` had consistent last_touch
        recent_touch_slice = alive_df.iloc[i - min_frames:i]
        team = alive_df.loc[i, 'last_touch']

        if team is None or not all(recent_touch_slice['last_touch'] == team):
            i += 1
            continue

        # Check player_possession validity at start
        player = alive_df.loc[i, 'player_possession']
        if not isinstance(player, str) or not player.startswith(team):
            i += 1
            continue

        # Start tracking possession from this index
        start_idx = i
        current_players = set()
        j = i

        # Extend possession until last_touch changes
        while j < n and alive_df.loc[j, 'last_touch'] == team:
            p = alive_df.loc[j, 'player_possession']
            if isinstance(p, str) and p.startswith(team):
                current_players.add(p)
            j += 1

        possession_length = j - start_idx

        if possession_length >= min_frames and len(current_players) >= min_players:
            deliberate_possessions.append({
                'start_frame': alive_df.loc[start_idx, 'global_frame_id'],
                'end_frame': alive_df.loc[j - 1, 'global_frame_id'],
                'team': team,
                'players': current_players.copy(),
                'period': alive_df.loc[start_idx, 'period']
            })

        # Move index forward
        i = j
    return deliberate_possessions