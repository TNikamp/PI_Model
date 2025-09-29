import numpy as np

def is_moving_towards_ball(pos, vel, ball_pos, max_angle_deg=30):
    """
    Determine whether a player is moving roughly towards the ball.

    Args:
        pos (np.ndarray): Player's current position as a 2D array [x, y].
        vel (np.ndarray): Player's current velocity vector as a 2D array [vx, vy].
        ball_pos (np.ndarray): Ball's current position as a 2D array [x, y].
        max_angle_deg (float, optional): Maximum angle (in degrees) allowed 
            between the player's velocity vector and the vector to the ball
            to be considered "moving towards". Defaults to 30 degrees.

    Returns:
        bool: True if the player is moving towards the ball within the allowed angle,
              False otherwise or if position/velocity cannot define a direction.
    """
    # Vector pointing from player to ball
    to_ball_vec = ball_pos - pos

    # Magnitude of vectors
    to_ball_mag = np.linalg.norm(to_ball_vec)  # Distance to ball
    vel_mag = np.linalg.norm(vel)             # Speed of player

    # If player is stationary or at the same position as ball, direction is undefined
    if to_ball_mag == 0 or vel_mag == 0:
        return False

    # Cosine of angle between velocity vector and vector to ball
    cos_theta = np.dot(to_ball_vec, vel) / (to_ball_mag * vel_mag)

    # Convert to angle in radians, then degrees
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid numerical errors
    angle_deg = np.degrees(angle_rad)

    # Return True if angle is within allowed threshold
    return angle_deg <= max_angle_deg
