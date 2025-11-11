import numpy as np
import matplotlib.pyplot as plt

def dubinEHF3d(east1, north1, alt1, psi1, east2, north2, r, step, gamma, data_count):
    """ Dubin's path calculation for 3D, with end heading free.

    INPUTS:
    - east1, north1, alt1: Start position coordinates
    - psi1: Initial heading in radians
    - east2, north2: Goal position coordinates
    - r: Turn radius
    - step: Step size for trajectory discretization
    - gamma: Flight path angle in radians

    OUTPUTS:
    - path: Array of (east, north, altitude) points forming the trajectory
    - psi_end: Final heading at the end of the path
    - num_path_points: Number of points in the path
    """

    MAX_NUM_PATH_POINTS = 1000
    path = np.zeros((MAX_NUM_PATH_POINTS, 3))
    r_sq = r**2
    psi1 %= 2*np.pi

    theta_l = psi1 + np.pi/2
    eastc_l = east1 + r*np.cos(theta_l)
    northc_l = north1 + r*np.sin(theta_l)

    theta_r = psi1 - np.pi/2
    eastc_r = east1 + r*np.cos(theta_r)
    northc_r = north1 + r*np.sin(theta_r)

    d2c_l_sq = (east2-eastc_l)**2 + (north2-northc_l)**2
    d2c_r_sq = (east2-eastc_r)**2 + (north2-northc_r)**2
    d2c_l = np.sqrt(d2c_l_sq)
    d2c_r = np.sqrt(d2c_r_sq)

    if d2c_l < r or d2c_r < r:
        # print(str(data_count) + ": No solution: Distance to goal is less than turn radius.")
       
        return np.zeros((0, 3)), 0, 0

    theta_c_l = np.arctan2(north2-northc_l, east2-eastc_l)
    if theta_c_l < 0: theta_c_l += 2*np.pi
    theta_c_r = np.arctan2(north2-northc_r, east2-eastc_r)
    if theta_c_r < 0: theta_c_r += 2*np.pi

    lt_l = np.sqrt(d2c_l_sq-r_sq)
    lt_r = np.sqrt(d2c_r_sq-r_sq)

    theta_start_l = theta_r
    theta_start_r = theta_l
    theta_d_l = np.arccos(r/d2c_l)
    theta_end_l = theta_c_l - theta_d_l

    if theta_end_l < theta_start_l:
        theta_end_l += 2*np.pi
    elif theta_end_l > theta_start_l + 2*np.pi:
        theta_end_l -= 2*np.pi

    theta_d_r = np.arccos(r/d2c_r)
    theta_end_r = theta_c_r + theta_d_r

    if theta_end_r < theta_start_r - 2*np.pi:
        theta_end_r += 2*np.pi
    elif theta_end_r > theta_start_r:
        theta_end_r -= 2*np.pi

    arc_l = abs(theta_end_l - theta_start_l)
    arc_r = abs(theta_end_r - theta_start_r)
    arc_length_l = arc_l * r
    arc_length_r = arc_r * r

    total_length_l = arc_length_l + lt_l
    total_length_r = arc_length_r + lt_r

    if total_length_l < total_length_r:
        if arc_length_l > 0.1:
            theta_step = step/r
            num_arc_seg = max(2, int(np.ceil(arc_l/theta_step)))
            angles = np.linspace(theta_start_l, theta_end_l, num_arc_seg)
            alt_end = alt1 + arc_length_l * np.tan(gamma)
            altitude = np.linspace(alt1, alt_end, num_arc_seg)
            arc_traj = np.column_stack([
                eastc_l + r*np.cos(angles),
                northc_l + r*np.sin(angles),
                altitude
            ])
        else:
            arc_traj = np.array([[east1, north1, alt1]])
            num_arc_seg = 1

        if lt_l > 0.1 or arc_length_l < 0.1:
            num_line_seg = max(2, int(np.ceil(lt_l/step)))
            alt_begin = arc_traj[-1, 2]
            alt_end = alt_begin + lt_l * np.tan(gamma)
            line_traj = np.column_stack([
                np.linspace(arc_traj[-1, 0], east2, num_line_seg),
                np.linspace(arc_traj[-1, 1], north2, num_line_seg),
                np.linspace(alt_begin, alt_end, num_line_seg)
            ])
        else:
            line_traj = np.zeros((1, 3))
            num_line_seg = 0
    else:
        if arc_length_r > 0.1:
            theta_step = step/r
            num_arc_seg = max(2, int(np.ceil(arc_r/theta_step)))
            angles = np.linspace(theta_start_r, theta_end_r, num_arc_seg)
            alt_end = alt1 + arc_length_r * np.tan(gamma)
            altitude = np.linspace(alt1, alt_end, num_arc_seg)
            arc_traj = np.column_stack([
                eastc_r + r*np.cos(angles),
                northc_r + r*np.sin(angles),
                altitude
            ])
        else:
            arc_traj = np.array([[east1, north1, alt1]])
            num_arc_seg = 1

        if lt_r > 0.1 or arc_length_r < 0.1:
            num_line_seg = max(2, int(np.ceil(lt_r/step)))
            alt_begin = arc_traj[-1, 2]
            alt_end = alt_begin + lt_r * np.tan(gamma)
            line_traj = np.column_stack([
                np.linspace(arc_traj[-1, 0], east2, num_line_seg),
                np.linspace(arc_traj[-1, 1], north2, num_line_seg),
                np.linspace(alt_begin, alt_end, num_line_seg)
            ])
        else:
            line_traj = np.zeros((1, 3))
            num_line_seg = 0

    if num_line_seg > 1:
        num_path_points = num_arc_seg + num_line_seg - 1
        path[:num_arc_seg, :] = arc_traj
        path[num_arc_seg:num_path_points, :] = line_traj[1:, :]
    else:
        num_path_points = num_arc_seg
        path[:num_path_points, :] = arc_traj

    psi_end = np.arctan2(north2-arc_traj[-1, 1], east2-arc_traj[-1, 0])
    # print(str(data_count) + ": Solution Found!")
    return path, psi_end, num_path_points

def test_dubinEHF3d():
    # Initial conditions
    x1, y1, alt1 = 0, 0, 0
    psi1 = 20 * np.pi / 180  # Initial heading

    # Goal conditions
    gamma = 30 * np.pi / 180  # Climb angle
    x2, y2 = -500, -500

    # Path parameters
    step_length = 10  # Trajectory discretization step size
    r_min = 100  # Minimum turn radius

    # Generate path
    path, psi_end, num_path_points = dubinEHF3d(
        x1, y1, alt1, psi1, x2, y2, r_min, step_length, gamma
    )

    # Plot the path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(path[:num_path_points, 0], path[:num_path_points, 1], path[:num_path_points, 2], 'b.-')
    ax.plot([x1], [y1], [alt1], 'r*', markersize=10)
    ax.plot([x2], [y2], [0], 'm*', markersize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Altitude')
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    test_dubinEHF3d()