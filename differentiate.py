import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def calc_velocities(
    df: pd.DataFrame,
    frame_rate: float = 25,
    filter_type: str = None,
    window_length: int = 7,
    polyorder: int = 2,
    max_velocity: float = np.inf,
    inplace: bool = False,
    column_ids: list = None,
    ball_filter_type: str = "moving_average",
    ball_window_length: int = 12,
    ball_polyorder: int = 2,
    ball_max_velocity: float = 50.0,
) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

    dt = 1.0 / frame_rate

    x_cols = {col[:-2] for col in df.columns if col.endswith("_x")}
    y_cols = {col[:-2] for col in df.columns if col.endswith("_y")}
    entity_ids = sorted(x_cols & y_cols)

    if column_ids is not None:
        entity_ids = [eid for eid in entity_ids if eid in column_ids]
        
    new_data = {}

    for entity in entity_ids:
        x_col, y_col = f"{entity}_x", f"{entity}_y"
        vx = np.gradient(df[x_col].values, dt)
        vy = np.gradient(df[y_col].values, dt)

        is_ball = entity == "ball"
        if is_ball and ball_filter_type:
            if ball_filter_type == "savitzky_golay":
                w = ball_window_length + 1 if ball_window_length % 2 == 0 else ball_window_length
                vx = savgol_filter(vx, w, ball_polyorder)
                vy = savgol_filter(vy, w, ball_polyorder)
            elif ball_filter_type == "moving_average":
                vx = pd.Series(vx).rolling(ball_window_length, center=True, min_periods=1).mean().values
                vy = pd.Series(vy).rolling(ball_window_length, center=True, min_periods=1).mean().values
        elif filter_type:
            if filter_type == "savitzky_golay":
                w = window_length + 1 if window_length % 2 == 0 else window_length
                vx = savgol_filter(vx, w, polyorder)
                vy = savgol_filter(vy, w, polyorder)

        speed = np.linalg.norm([vx, vy], axis=0)

        vmax = ball_max_velocity if is_ball else max_velocity
        if np.isfinite(vmax):
            mask = speed > vmax
            if np.any(mask):
                scale = vmax / speed[mask]
                vx[mask] *= scale
                vy[mask] *= scale
                speed[mask] = vmax

        new_data[f"{entity}_vx"] = vx
        new_data[f"{entity}_vy"] = vy
        new_data[f"{entity}_velocity"] = speed

    df = pd.concat([df, pd.DataFrame(new_data, index=df.index)], axis=1)
    return df


def calc_accelerations(
    df: pd.DataFrame,
    frame_rate: float = 25,
    filter_type: str = None,
    window_length: int = 7,
    polyorder: int = 2,
    max_acceleration: float = np.inf,
    inplace: bool = False,
    column_ids: list = None,
    ball_filter_type: str = "savitzky_golay",
    ball_window_length: int = 35,
    ball_polyorder: int = 2,
    ball_max_acceleration: float = 20.0,
) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

    dt = 1.0 / frame_rate

    vx_cols = {col[:-3] for col in df.columns if col.endswith("_vx")}
    vy_cols = {col[:-3] for col in df.columns if col.endswith("_vy")}
    entity_ids = sorted(vx_cols & vy_cols)

    if column_ids is not None:
        entity_ids = [eid for eid in entity_ids if eid in column_ids]

    for entity in entity_ids:
        vx = df[f"{entity}_vx"].values
        vy = df[f"{entity}_vy"].values

        ax = np.gradient(vx, dt)
        ay = np.gradient(vy, dt)

        is_ball = entity == "ball"
        if is_ball and ball_filter_type:
            if ball_filter_type == "savitzky_golay":
                w = ball_window_length + 1 if ball_window_length % 2 == 0 else ball_window_length
                ax = savgol_filter(ax, w, ball_polyorder)
                ay = savgol_filter(ay, w, ball_polyorder)
            elif ball_filter_type == "moving_average":
                ax = pd.Series(ax).rolling(ball_window_length, center=True, min_periods=1).mean().values
                ay = pd.Series(ay).rolling(ball_window_length, center=True, min_periods=1).mean().values
        elif filter_type:
            if filter_type == "savitzky_golay":
                w = window_length + 1 if window_length % 2 == 0 else window_length
                ax = savgol_filter(ax, w, polyorder)
                ay = savgol_filter(ay, w, polyorder)

        acc = np.linalg.norm([ax, ay], axis=0)

        amax = ball_max_acceleration if is_ball else max_acceleration
        if np.isfinite(amax):
            mask = acc > amax
            if np.any(mask):
                scale = amax / acc[mask]
                ax[mask] *= scale
                ay[mask] *= scale
                acc[mask] = amax

        df[f"{entity}_ax"] = ax
        df[f"{entity}_ay"] = ay
        df[f"{entity}_acceleration"] = acc

    return df

def calc_decelerations(
    df: pd.DataFrame,
    frame_rate: float = 25.0,
    inplace: bool = False,
    column_ids: list = None,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Requires columns: <entity>_vx, <entity>_vy, <entity>_ax, <entity>_ay
    Adds:
      <entity>_a_t           : signed tangential acceleration (m/s^2)
      <entity>_decel_signed  : negative part of a_t (≤ 0)
    """
    if not inplace:
        df = df.copy()

    vx_cols = {c[:-3] for c in df.columns if c.endswith("_vx")}
    vy_cols = {c[:-3] for c in df.columns if c.endswith("_vy")}
    ax_cols = {c[:-3] for c in df.columns if c.endswith("_ax")}
    ay_cols = {c[:-3] for c in df.columns if c.endswith("_ay")}
    entity_ids = sorted(vx_cols & vy_cols & ax_cols & ay_cols)

    if column_ids is not None:
        entity_ids = [eid for eid in entity_ids if eid in column_ids]

    for entity in entity_ids:
        vx = df[f"{entity}_vx"].to_numpy(dtype=float)
        vy = df[f"{entity}_vy"].to_numpy(dtype=float)
        ax = df[f"{entity}_ax"].to_numpy(dtype=float)
        ay = df[f"{entity}_ay"].to_numpy(dtype=float)

        speed = np.hypot(vx, vy)
        denom = np.where(speed > eps, speed, np.nan)  # avoid divide-by-zero

        # Tangential (signed): change in speed
        a_t = (vx * ax + vy * ay) / denom
        a_t = np.nan_to_num(a_t, nan=0.0)

        # Decelerations
        decel_signed = np.where(a_t < 0, a_t, 0.0)   # ≤ 0

        new_cols = pd.DataFrame({
            f"{entity}_a_t": a_t,
            f"{entity}_decel_signed": decel_signed,
        }, index=df.index)
        
        df = pd.concat([df, new_cols], axis=1)

    return df