import pandas as pd
from helper import *
from my_utils import *
from tqdm import tqdm
import math
# Pre processing

# Loading in files
filepath = # Insert file location/ method of importing here
eventpath = # Insert file location/ method of importing here

tracking_df = pd.read_csv(filepath)
event_df = pd.read_csv(eventpath)

# Replace team names in event data with those of tracking data to prevent running into name comparing issues
event_df['home_team_name'] = tracking_df['home_team_id'].iloc[0]
event_df['away_team_name'] = tracking_df['away_team_id'].iloc[0]

# Mapping ball status and filtering ball columns
tracking_df["ball_status"] = tracking_df["ball_status"].map({1: "alive", 0: "dead"})
ball_cols = get_column_ids(tracking_df, 'ball')
tracking_df = filter_tracking_data(tracking_df, column_ids=ball_cols)

# Calculating velocities and accelerations
tracking_df = calc_velocities(tracking_df, frame_rate=25, max_velocity=13)
tracking_df = calc_accelerations(tracking_df, frame_rate=25, max_acceleration=7, filter_type='savitzky_golay')
tracking_df = calc_decelerations(tracking_df, frame_rate=25)

# Individual player possesion and deliberate possessions algorithms
get_individual_player_possession(tracking_df, inplace=True)
possessions = get_deliberate_possessions(tracking_df, min_players=2, min_frames=125)

# Get block heights and add to tracking_df
tracking_df = classify_defensive_block(tracking_df)

#%% Pressing events model

pressing_events = []
# Parameters for pressing model
max_x_range = 20 # x range for contributing players
individual_threshold = 0.55 # PTI value
team_threshold = 0.3 # PTI value

for possession in tqdm(possessions, desc="Processing possessions", unit="possession"):
    team_in_possession = possession['team']
    team_out_of_possession = 'away' if team_in_possession == 'home' else 'home'   

    possession_frames = tracking_df[
        (tracking_df['global_frame_id'] >= possession['start_frame']) &
        (tracking_df['global_frame_id'] <= possession['end_frame'])
    ]
    
    event_extracted = False

    for frame_idx in possession_frames.index:
        if event_extracted:
            break
        
        series = tracking_df.loc[frame_idx]
        player_possession = series['player_possession']
               
        # Check if not None/NaN and starts with team in possession
        if pd.notna(player_possession) and str(player_possession).startswith(team_in_possession):
            
            # Get index within possession_frames to check last 5 frames
            pos_idx = possession_frames.index.get_loc(frame_idx)
            if pos_idx >= 4:
                last_five = possession_frames.iloc[pos_idx - 4:pos_idx + 1]['player_possession'].fillna('')
                if not all(str(p) == str(player_possession) for p in last_five):
                    continue  # Skip if not stable for 5 frames

            try:
                possession_jersey = int(player_possession.split('_')[1])
            except IndexError:
                print(f"IndexError: Could not split player_possession value '{player_possession}'")
                continue
            except ValueError:
                print(f"ValueError: Could not convert to int from '{player_possession.split('_')[1]}' in '{player_possession}'")
                continue
            
            # Calculate TTI Matrix with PTI values per slice of tracking data
            pti_frame = pti_model(series)
            defender_columns = [col for col in pti_frame.columns if isinstance(col, int)]
            defender_columns_sideline = [col for col in pti_frame.columns if isinstance(col, int) or col == "Sideline"]

            pressure_values = pti_frame.loc[possession_jersey][defender_columns_sideline].astype(float)
            single_pressure_values = pti_frame.loc[possession_jersey][defender_columns].astype(float)

            # Keep only defenders applying significant pressure
            high_single_pressure = single_pressure_values[single_pressure_values > individual_threshold]
        
            # Filter defenders by whether they are moving toward the ball
            ball_pos = np.array([series['ball_x'], series['ball_y']])
            valid_on_field_players = set()
            valid_high_pressure = {}
        
            # Iterate over all defenders (not just the high-pressure ones)
            for defender_jersey in pti_frame.columns:
                if isinstance(defender_jersey, int):  # Check if it's a defender's jersey number
                    defender_key = f"{team_out_of_possession}_{defender_jersey}"
                    dx = series.get(f"{defender_key}_x")
                    dy = series.get(f"{defender_key}_y")
                    vx = series.get(f"{defender_key}_vx")
                    vy = series.get(f"{defender_key}_vy")
                    
                    if pd.notna(dx):
                        # Define the x-range of the ball carrier
                        min_x = ball_pos[0] - max_x_range  
                        max_x = ball_pos[0] + max_x_range
                        
                        # Check if defender is within the x-range of the ball carrier
                        if min_x <= dx <= max_x:
                            valid_on_field_players.add(defender_jersey)
            
                    if pd.notna(dx) and pd.notna(dy) and pd.notna(vx) and pd.notna(vy):
                        # Check if the player is applying high pressure
                        single_pressure_value = single_pressure_values.get(defender_jersey, 0)
            
                        if single_pressure_value > individual_threshold:
                            # Check if defender is moving toward the ball
                            pos = np.array([dx, dy])
                            vel = np.array([vx, vy])
                    
                            if is_moving_towards_ball(pos, vel, ball_pos):
                                valid_high_pressure[defender_jersey] = single_pressure_value
                        
            # Team pressure                        
            team_pressure_matrix = pti_frame[defender_columns].astype(float)
            
            # Initialize the counter for team pressure contributors
            team_pressure_contributors = []
            sideline_contribution = []
            
            # Iterate over valid on-field players
            for defender_jersey in valid_on_field_players:
                # Get the pressure value for the current defender from the team pressure matrix
                defender_pressure_value = team_pressure_matrix[defender_jersey]
            
                # Check if the defender's pressure is >= 0.3 on any
                if (defender_pressure_value >= 0.3).any():
                    defender_key = f"{team_out_of_possession}_{defender_jersey}"
                    team_pressure_contributors.append(defender_key)            

            # Sideline setup
            sideline_series = pti_frame['Sideline']
            if (sideline_series >= 0.3).any():
                sideline_contribution.append("Sideline")
            
            total_contribution = len(team_pressure_contributors) + len(sideline_contribution)

            # Condition 1: Check if there is at least one player applying high pressure
            # Condition 2: At least 3 other players + sideline contributing (making 4 total players)
            if len(valid_high_pressure) >= 1 and total_contribution >= 3:
                # Check if player in possession is inside the 16m box
                possessor_x = series.get(f"{player_possession}_x")
                possessor_y = series.get(f"{player_possession}_y")
                
                in_16m_box = (
                    abs(possessor_x) >= 52.5 - 16.5 and
                    abs(possessor_y) <= 20.15
                )
                
                if in_16m_box:
                    continue  # Skip if player is in the 16m box
                
                block = series.get(f'defensive_block_{team_out_of_possession}')
                total_pressure = pressure_values.sum()
                
                pressing_events.append({
                    "frame": series['global_frame_id'],
                    "period": series['period'],
                    "pressing_team": team_out_of_possession,
                    'attacking_team': team_in_possession,
                    "block_height": block,
                    "player_possession": player_possession,
                    "total_pressure_on_carrier": total_pressure,
                    "team_pressure_contributors": team_pressure_contributors,
                    "sideline_contributors": sideline_contribution
                })
                
                event_extracted = True
                
            pass
        else:
            continue

# %% Event outcomes


def evaluate_pressing_success(df, event, frame_rate=25, min_backward_pass_distance=11):

    """
    Evaluate whether a pressing attempt in football (soccer) was successful or not.  
    
    The function uses tracking and event data to analyze what happened after a pressing trigger:  
      - Checks if the press fails (e.g., opponent creates a shot on target or reaches penalty box).  
      - Checks if the press succeeds (e.g., possession regained, forced backward pass, pass to goalkeeper, foul, or ball out of play).  
      - Returns detailed outcome information including type of success/failure, relevant frames, and contextual info about the pressing team.  
    
    Parameters
    ----------
    df : pd.DataFrame
        Tracking data for the match (player positions, ball position, possessions, etc.).
    event : dict
        Dictionary describing the pressing event, containing metadata such as pressing team, frame of press start,
        block height, contributing players, etc.
    frame_rate : int, optional (default=25)
        Number of tracking frames per second.
    min_backward_pass_distance : int, optional (default=11)
        Minimum distance (in meters) a backward pass must travel to count as a "forced backward pass".
    
    Returns
    -------
    dict
        Dictionary with keys describing the outcome:
        - successful (bool): Whether the press was successful
        - success_type / failure_type (str): What type of outcome occurred
        - success_frame / failure_frame (int): Frame offset (relative to press start)
        - start_frame, end_frame (int): Time window of press
        - time_to_failure_sec (float): If failed, time until failure in seconds
        - pressing_team (str): Team name
        - pressing_players (list): Pressing contributors
        - block_height (str): 'low', 'mid', or 'high'
    """

    def find_goalkeeper(tracking_df):
        '''
        Find the goalkeeper in team, identifying him/her as the player closest to goal at kick off
        ''' 
        x_columns_home = [c for c in tracking_df.columns if c[-2:].lower()=='_x' and c[:4] in ['home']]
        GK_home = tracking_df.iloc[[0]][x_columns_home].abs().idxmax(axis=1)
        
        x_columns_away = [c for c in tracking_df.columns if c[-2:].lower()=='_x' and c[:4] in ['away']]
        GK_away = tracking_df.iloc[[0]][x_columns_away].abs().idxmax(axis=1)
        
        return GK_home.iloc[0][:-2], GK_away.iloc[0][:-2]
    
    home_gk_id, away_gk_id = find_goalkeeper(tracking_df)
    
    def is_pass_in_window(frame, pass_event, tolerance=5):
        """Check if a pass event exists within ±tolerance frames of a given frame."""
        return not pass_event[
            (pass_event['tracking_frame'] >= frame - tolerance) &
            (pass_event['tracking_frame'] <= frame + tolerance)
        ].empty
    
    # Get pressing and attacking team context
    pressing_team = event['pressing_team']
    pressing_team_name = tracking_df[f"{event['pressing_team']}_team_id"][0]
    
    attacking_team = 'home' if pressing_team == 'away' else 'away'
    attacking_team_name = tracking_df[f"{event['attacking_team']}_team_id"][0]
    
    # Press block height info
    defensive_block_height = event['block_height']
    block_height_label = {0: 'low', 1: 'mid', 2: 'high'}.get(defensive_block_height, 'unknown')
    
    # Frame window to analyze
    start_frame = event['frame'] #end frame not needed, if succes/false before end timewindow --> thats endframe, otherwise start + 250  
    
    df_window = tracking_df[(tracking_df['global_frame_id'] >= start_frame) & (tracking_df['global_frame_id'] <= end_frame)].copy()
    event_window = event_df[(event_df['tracking_frame'] >= start_frame) & (event_df['tracking_frame'] <= end_frame)].copy()
    
    pressing_players = event.get('team_pressure_contributors')
    
    # If no tracking data is found in the window --> automatic failure
    if df_window.empty:
       print('Warning, no tracking data found!')
       return {
        'successful': False,
        'success_type': None,
        'success_frame': None,
        'start_frame': start_frame,
        'end_frame': (start_frame + (success_frame if pd.notna(success_frame) else 250)),
        'pressing_team': pressing_team_name,
        'pressing_players': pressing_players,
        'block_height': block_height_label
    }

    # Initial conditions at start of pressing
    initial_ball_x = df_window.iloc[0]['ball_x']
    initial_possesor = df_window.iloc[0]['player_possession']
    
    success_type = None
    success_frame = None
    
    # Identify goalkeeper for defending team
    defending_gk_id = home_gk_id if pressing_team == 'home' else away_gk_id
        
    # Extract shots and passes in the event_window
    shot_event = event_window[
        (event_window['team_name'] == attacking_team_name) &
        (event_window['base_type_name'] == 'SHOT') &
        (event_window['shot_type_name'].isin(['ON_TARGET', 'POST']))
    ]

    pass_event = event_window[
        (event_window['team_name'] == attacking_team_name) &
        (event_window['base_type_name'] == 'PASS') &
        (event_window['result_id'] == 1)]
    
    # Shot on goal by opponent = automatic failure
    if not shot_event.empty:
        first_shot = shot_event.iloc[0]
        failure_type = 'shot_on_goal'
        failure_frame = first_shot['tracking_frame'] - start_frame
        return({
            'successful': False,
            'failure_type': failure_type,
            'failure_frame': failure_frame,
            'start_frame': start_frame,
            'end_frame': (start_frame + (failure_frame if pd.notna(failure_frame) else 250)),
            'time_to_failure_sec': failure_frame / frame_rate,
            'pressing_team': pressing_team_name,
            'pressing_players': pressing_players,
            'block_height': defensive_block_height
        })    

    for i, (idx, row) in enumerate(df_window.iterrows()):
        current_possessor = row['player_possession']
    
        # First check if the pressing might have failed because the opponnent reached the penalty box
        if current_possessor and current_possessor.startswith(attacking_team):
            try:
                x = row[f"{current_possessor}_x"]
                y = row[f"{current_possessor}_y"]
                
                if pressing_team == 'home':
                    if x <= -52.5 + 16.5 and abs(y) <= 40.32 / 2:
                        failure_type = 'penalty_box_reached'
                        failure_frame = (row['global_frame_id'] - start_frame)
                        return {
                            'successful': False,
                            'failure_type': failure_type,
                            'failure_frame': failure_frame,
                            'start_frame': start_frame,
                            'end_frame': (start_frame + (failure_frame if pd.notna(failure_frame) else 250)),
                            'time_to_failure_sec': failure_frame / frame_rate,
                            'pressing_team': pressing_team_name,
                            'pressing_players': pressing_players,
                            'block_height': block_height_label
                        }
                else:  # pressing_team == 'away'
                    if x >= 52.5 - 16.5 and abs(y) <= 40.32 / 2:
                        failure_type = 'penalty_box_reached'
                        failure_frame = (row['global_frame_id'] - start_frame)
                        return {
                            'successful': False,
                            'failure_type': failure_type,
                            'failure_frame': failure_frame,
                            'start_frame': start_frame,
                            'end_frame': (start_frame + (failure_frame if pd.notna(failure_frame) else 250)),
                            'time_to_failure_sec': failure_frame / frame_rate,
                            'pressing_team': pressing_team_name,
                            'pressing_players': pressing_players,
                            'block_height': block_height_label
                        }
            except KeyError:
                pass
            
        # Logic for success: Ball goes dead (foul or out of play)
        if row.get('ball_status', 'alive') == 'dead':
            ball_x = row.get('ball_x', 0)
            ball_y = row.get('ball_y', 0)
            
            # If ball within the pitch boundaries --> Foul
            if -52.5 <= ball_x <= 52.5 and -34 <= ball_y <= 34:
                success_type = 'foul'
                success_frame = (row['global_frame_id'] - start_frame)
                break
            # Else ball outside the pitch boundaries --> ball out of play
            else:
                success_type = 'ball_out_of_play'
                success_frame = (row['global_frame_id'] - start_frame)
                break
                
        
        # Passing (backwards or towards goalkeeper)
        if pd.isna(current_possessor):
            continue

        # Check if pass went to same team goalkeeper
        initial_team = 'home' if initial_possesor.startswith('home') else 'away'
        initial_team_gk_id = home_gk_id if initial_team == 'home' else away_gk_id
    
        if current_possessor == initial_team_gk_id and initial_possesor != initial_team_gk_id:
            if is_pass_in_window(row['global_frame_id'], pass_event):
                success_type = 'pass_to_goalkeeper'
                success_frame = (row['global_frame_id'] - start_frame)
                break
            
        # Forced backward pass
        ball_x = row['ball_x']
        if (
            (pressing_team == 'home' and ball_x - initial_ball_x >= min_backward_pass_distance) or
            (pressing_team == 'away' and initial_ball_x - ball_x >= min_backward_pass_distance)
        ):
            # Check event data if a pass occured within this window
            if is_pass_in_window(row['global_frame_id'], pass_event):
                success_type = 'forced_backward_pass'
                success_frame = (row['global_frame_id'] - start_frame)
                break

        # Look back at last 5 frames of possession for possible possession gains (true possession, not just some random assignment because the ball was near a player long enough)
        if i >= 4:
            last_five = df_window.iloc[i - 4:i + 1]['player_possession'].fillna('')
            if all(p.startswith(pressing_team) for p in last_five):
                success_type = 'possession_gain'
                success_frame = (row['global_frame_id'] - start_frame)
                break
    
    return {
        'successful': success_type is not None,
        'success_type': success_type,
        'success_frame': success_frame,
        'start_frame': start_frame,
        'end_frame': (start_frame + (success_frame if pd.notna(success_frame) else 250)),
        'pressing_team': pressing_team_name,
        'pressing_players': pressing_players,
        'block_height': block_height_label
    }

outcomes_list = []

# Loop over the entire set of pressings, determining the outcomes
for event in pressing_events:
    start_frame = event['frame']
    end_frame = start_frame + 250  # 10 seconds at 25 FPS

    if tracking_df['global_frame_id'].max() < end_frame:
        print(f"Skipping pressing event at frame {start_frame} — not enough data after event to evaluate (requires 10s).")
        continue

    outcome = evaluate_pressing_success(tracking_df, event)
    outcomes_list.append(outcome)
    
#%% External load parameter calculations for each pressing event

# Constants
framerate = 25
dt = 1 / framerate
eccr = 3.6  # energy cost of constant running

# Time windows in frames
pre_frames = 3 * framerate  # 75
post_frames = 10 * framerate  # 250

# Speed zones (in m/s)
speed_zones = {
    "walking": (0.16, 1.97),
    "jogging": (1.98, 3.98),
    "running": (3.98, 5.47),
    "high_speed_running": (5.48, 7.0),
    "sprinting": (7.0, np.inf),
}

# Define zones (W/kg)
power_zones = {
    "low_power": (0.0, 10.0),
    "intermediate_power": (10.0, 20.0),
    "high_power": (20.0, 35.0),
    "elevated_power": (35.0, 55.0),
    "maximal_power": (55.0, np.inf),  # > 55 W/kg
}

# --- Helper: get base player column names (e.g., "home_1", "away_7") ---

def get_player_bases(tracking_df, side_prefix):
    """Return sorted base column names for players on one side.

    Args:
        tracking_df (pd.DataFrame): Tracking data with per-player x/y/velocity columns.
        side_prefix (str): Prefix identifying the team side (e.g., 'home' or 'away').
            We look for columns like f"{side_prefix}_<id>_x" and strip the trailing suffix.

    Returns:
        list[str]: Sorted unique base names like 'home_1', 'away_7'.
    """
    bases = []
    for c in tracking_df.columns:
        # Columns come in groups like '<base>_x', '<base>_y', '<base>_velocity', etc.
        if c.startswith(side_prefix) and c.endswith('_x'):
            bases.append(c[:-2])  # strip "_x" to get the base, e.g., 'home_1'
    return sorted(set(bases))


# Identify the home/away player base names once (re-used later)
home_players = get_player_bases(tracking_df, 'home')
away_players = get_player_bases(tracking_df, 'away')


# --- Helper: active players (non-NaN velocity at start_frame) ---

def get_active_players(player_bases, frame_idx):
    """Return players that are active (have a non-NaN velocity) at a given frame.

    Args:
        player_bases (list[str]): Base names such as 'home_1', 'away_5'.
        frame_idx (int): Frame index at which to check activity.

    Returns:
        list[str]: Subset of player_bases that have a valid velocity at frame_idx.
    """
    active = []
    for p in player_bases:
        v_col = f"{p}_velocity"
        if v_col in tracking_df.columns:
            val = tracking_df.iloc[frame_idx][v_col]
            if pd.notna(val):  # consider active if velocity is present (not NaN)
                active.append(p)
    return active


# --- Core: compute external load for a window & list of players ---

def compute_external_load_for_window(start_frame, end_frame, players):
    """Compute external load metrics for a set of players over a frame window.

    This function aggregates distance/time in speed zones, counts accel/decel efforts,
    and computes metabolic power stats in the given window.

    Args:
        start_frame (int): Start frame (inclusive) of the analysis window.
        end_frame (int): End frame (exclusive) of the analysis window.
        players (list[str]): Player base names to process (e.g., 'home_3').

    Requires (from outer scope):
        - tracking_df (pd.DataFrame) with per-frame player metrics.
        - pre_frames (int): Frames to include before start_frame for context.
        - dt (float): Seconds per frame (1 / framerate).
        - framerate (float): Sampling rate (frames per second).
        - speed_zones (dict[str, tuple[float, float]]): Named speed ranges (m/s).
        - power_zones (dict[str, tuple[float, float]]): Named power ranges (W/kg).
        - calculate_metabolic_power (callable): Function computing power per player.

    Returns:
        list[dict]: Per-player metric dictionaries for the window.
    """
    results = []

    # Expand the window backwards by 'pre_frames' to capture ramps into the event
    start = max(int(start_frame) - pre_frames, 0)
    end = min(int(end_frame), len(tracking_df))

    if end <= start:
        # Empty or invalid window → nothing to compute
        return results

    df_window = tracking_df.iloc[start:end].copy()
    time_window = (end - start) * dt  # window duration in seconds

    def get_effort_stats(mask):
        """Count continuous efforts and time proportion given a boolean mask.

        An 'effort' is a continuous run of True values lasting at least 0.7 s.

        Args:
            mask (np.ndarray[bool]): Per-frame boolean mask (e.g., accel > threshold).

        Returns:
            tuple[int, float]: (effort_count, time_prop) where time_prop is the
                proportion of frames within the window contributing to efforts.
        """
        effort_frames = 0
        effort_count = 0
        in_effort = False
        current_duration = 0

        for i in range(len(mask)):
            if mask[i]:
                current_duration += 1
                in_effort = True
            else:
                # If we exit an effort, check if it met the minimum duration
                if in_effort and current_duration >= 0.7 * framerate:
                    effort_count += 1
                    effort_frames += current_duration
                in_effort = False
                current_duration = 0

        # Handle trailing effort that reaches the end of the window
        if in_effort and current_duration >= 0.7 * framerate:
            effort_count += 1
            effort_frames += current_duration

        time_prop = effort_frames / len(mask) if len(mask) > 0 else 0.0
        return effort_count, time_prop

    for player in players:
        v_col = f"{player}_velocity"
        a_col = f"{player}_acceleration"
        d_col = f"{player}_decel_signed"

        # Skip if any required column is missing for this player
        if v_col not in df_window.columns or a_col not in df_window.columns or d_col not in df_window.columns:
            continue

        # --- Velocity & distance ---
        velocity = df_window[v_col].values  # m/s per frame
        total_distance = np.sum(velocity * dt)  # integrate speed over time

        # --- Speed zones (vectorized for efficiency) ---
        zone_masks = {
            zone: (velocity >= lo) & (velocity < hi)
            for zone, (lo, hi) in speed_zones.items()
        }

        # Time spent in each speed zone (seconds)
        speed_zone_times = {
            zone: mask.sum() * dt
            for zone, mask in zone_masks.items()
        }

        # Proportion of total event time in each speed zone
        speed_zone_time_props = {
            f"time_prop_{zone}": (mask.sum() * dt) / time_window if time_window > 0 else 0.0
            for zone, mask in zone_masks.items()
        }

        # Distance covered in each speed zone (meters), using the actual speeds
        speed_zone_distances = {
            zone: np.sum(velocity[mask]) * dt
            for zone, mask in zone_masks.items()
        }

        # --- Acceleration / deceleration efforts ---
        acceleration = df_window[a_col].values
        deceleration = df_window[d_col].values

        # Thresholds (m/s^2) for defining high-intensity efforts; tune as needed
        accel_mask = acceleration > 2.0
        decel_mask = deceleration < -2.0

        # Count continuous efforts (≥ 0.7 s by default)
        accel_count, accel_time_prop = get_effort_stats(accel_mask)
        decel_count, decel_time_prop = get_effort_stats(decel_mask)

        # --- Metabolic power (W/kg) ---
        power_result = calculate_metabolic_power(df_window, [player], framerate=framerate)

        # Support both {player: [values]} and {player: {"metabolic_power": [...]}} shapes
        if isinstance(power_result.get(player, None), dict):
            power_values = power_result[player].get("metabolic_power", [])
        else:
            power_values = power_result.get(player, [])

        # Replace NaNs/Infs to avoid propagating them into stats
        power_values = np.nan_to_num(np.asarray(power_values))

        # --- Align, clean arrays (safety against length mismatches) ---
        pv = np.asarray(power_values, dtype=float)
        vel = np.asarray(velocity, dtype=float)
        n = min(pv.size, vel.size)
        pv = np.nan_to_num(pv[:n], nan=0.0)
        vel = np.nan_to_num(vel[:n], nan=0.0)

        # --- Power zones ---
        # Build masks per power zone (inclusive lower bound, exclusive upper bound),
        # except for open-ended upper bounds where we use pv >= lo.
        power_zone_masks = {}
        for name, (lo, hi) in power_zones.items():
            if np.isfinite(hi):
                mask = (pv >= lo) & (pv < hi)
            else:
                mask = (pv >= lo)
            power_zone_masks[name] = mask

        # Distance per power zone (meters) using aligned velocity
        dist_power_zones = {
            f"dist_power_{name}_m": float(np.sum(vel[mask]) * dt)
            for name, mask in power_zone_masks.items()
        }

        # Aggregate scalar power metrics
        mean_power = float(np.mean(power_values)) if power_values.size else 0.0
        peak_power = float(np.max(power_values)) if power_values.size else 0.0
        total_work = float(np.sum(power_values) * dt)  # J/kg (since power is W/kg)

        # Collect per-player results
        results.append({
            "player": player,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "event_duration_s": float(time_window),
            "total_distance_m": total_distance,
            **{f"dist_{z}_m": d for z, d in speed_zone_distances.items()},
            **{f"time_{z}_s": t for z, t in speed_zone_times.items()},
            **speed_zone_time_props,
            "accel_count": int(accel_count),
            "decel_count": int(decel_count),
            "accel_time_prop": float(accel_time_prop),
            "decel_time_prop": float(decel_time_prop),
            "met_power_mean": float(mean_power),
            "met_power_peak": float(peak_power),
            "met_work_total_j_per_kg": float(total_work),
            **dist_power_zones,
        })

    return results


# --- Orchestrate over all pressing outcomes ---

# Extract identifiers (assumes constant across the match)
home_id  = str(tracking_df['home_team_id'].iloc[0])
away_id  = str(tracking_df['away_team_id'].iloc[0])
match_id = str(tracking_df['match_id'].iloc[0]) if 'match_id' in tracking_df.columns else "unknown"

# Containers for summary and detailed per-press/per-player data
pressing_summary_rows = []
frames = []   # per-pressing, per-player DataFrames
keys   = []   # index keys → multiple levels in result

for pressing_number, outcome in enumerate(outcomes_list, start=1):
    # Unpack outcome metadata (with fallbacks)
    start_frame   = int(outcome['start_frame'])
    end_frame     = outcome.get('end_frame')
    pressing_team = outcome.get("pressing_team")
    successful    = bool(outcome.get("successful", False))
    block_height  = outcome.get("block_height")
    pressing_players = outcome.get('pressing_players')
    success_type = outcome.get('success_type')
    failure_type = outcome.get('failure_type')

    # Derive an outcome label with sensible default
    if success_type:
        outcome_type = success_type
    elif failure_type:
        outcome_type = failure_type
    else:
        outcome_type = 'time_limit_reached'

    # Determine active players at the start of this pressing (both teams)
    active_home = get_active_players(home_players, start_frame)
    active_away = get_active_players(away_players, start_frame)

    # Compute per-player external load for all active players in the window
    per_player = []
    for side_players in (active_home, active_away):
        per_player.extend(
            compute_external_load_for_window(start_frame, end_frame, side_players)
        )

    # Build a compact per-pressing player DataFrame for downstream analysis/exports
    player_rows = []
    for p in per_player:
        row = {
            "player": p["player"],
            "event_duration_s": p["event_duration_s"],
            "total_distance_m": p["total_distance_m"],
            "accel_count": p["accel_count"],
            "decel_count": p["decel_count"],
            "accel_time_prop": p["accel_time_prop"],
            "decel_time_prop": p["decel_time_prop"],
            "met_power_mean": p["met_power_mean"],
            "met_power_peak": p["met_power_peak"],
            "met_work_total_j_per_kg": p["met_work_total_j_per_kg"],
        }
        # Dynamically include any dist_*/time_* fields from zones without hard-coding names
        for k, v in p.items():
            if k.startswith("dist_") or k.startswith("time_"):
                row[k] = v
        player_rows.append(row)

    press_df = pd.DataFrame(player_rows)

    if not press_df.empty:
        # Index by player for easier slicing, and store with a multi-index key (match, press no.)
        press_df = press_df.set_index("player")
        press_df.index.name = "player"
        frames.append(press_df)
        keys.append((match_id, pressing_number))

    # Append an outer summary row for this pressing instance
    pressing_summary_rows.append({
        "pressing_id": f"{home_id}_{away_id}_{match_id}_{pressing_number}_{pressing_team}",
        "pressing_number": pressing_number,
        "pressing_team": pressing_team,
        "pressing_players": pressing_players,
        "successful": successful,
        "outcome_type": outcome_type,
        "block_height": block_height,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "n_players": len(pressing_players),
    })

# --- Build outer and nested DataFrames ---

# High-level, one row per pressing event
pressing_summary_df = pd.DataFrame(pressing_summary_rows)

# Nested details: multi-index (match_id, pressing_number) → per-player metrics
if frames:
    details_df = pd.concat(
        frames,
        keys=keys,
        names=["match_id", "pressing_number"]
    )
    # Optional: sort the multi-index for nicer browsing/exports
    details_df = details_df.sort_index()
else:
    details_df = pd.DataFrame()  # no data case














