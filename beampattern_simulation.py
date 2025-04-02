from scipy.constants import pi, c
import numpy as np
import matplotlib.pyplot as plt


def steering_vector(k, xv, yv, theta_deg, phi_deg):
    """
    Params:
        k (): wave vector
        xv (): x vector of element positions
        yv (): y vector of element positions
        theta_deg: steering angle in elevation direction
        phi_deg: steering angle in azimuth direction
    Returns:
        vector of phase weights for each element
    """
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    kx = k * np.sin(theta) * np.cos(phi)
    ky = k * np.sin(theta) * np.sin(phi)

    phase_weights = np.exp(1j * (kx * xv + ky * yv))

    return phase_weights

def azel_to_thetaphi(az, el):
    """ Az-El to Theta-Phi conversion.

    Args:
        az (float or np.array): Azimuth angle, in radians
        el (float or np.array): Elevation angle, in radians

    Returns:
      (theta, phi): Tuple of corresponding (theta, phi) angles, in radians
    """

    cos_theta = np.cos(el) * np.cos(az)
    # tan_phi = np.where(np.abs(np.sin(az)) < 1e-6, 0, np.tan(el) / np.sin(az)) # Avoid the divide by zero

    theta     = np.arccos(cos_theta)
    phi       = np.arctan2(np.tan(el), np.sin(az))
    phi = (phi + 2 * np.pi) % (2 * np.pi)

    return theta, phi

def thetaphi_to_azel(theta, phi):
    """ Az-El to Theta-Phi conversion.

    Args:
        theta (float or np.array): Theta angle, in radians
        phi (float or np.array): Phi angle, in radians

    Returns:
      (az, el): Tuple of corresponding (azimuth, elevation) angles, in radians
    """
    sin_el = np.sin(phi) * np.sin(theta)
    tan_az = np.cos(phi) * np.tan(theta)
    el = np.arcsin(sin_el)
    az = np.arctan(tan_az)

    return az, el

def dBi_to_linear(dBi):
  """Converts dBi to linear scale."""
  return 10**(dBi / 10)

def AF(theta, phi, x, y, w, k):
  """
  Calculates the array factor for a given set of angles, coordinates, weights, and wave number.

  Args:
    theta: Elevation angle in radians.
    phi: Azimuth angle in radians.
    x: X-coordinates of the antenna elements.
    y: Y-coordinates of the antenna elements.
    w: Complex weights of the antenna elements.
    k: Wave number.

  Returns:
    The array factor as a complex number.
  """

  N = len(x)  # Number of antenna elements

  # Calculate the phase shift for each antenna element
  phase_shift = -1j * k * (x * np.sin(theta) * np.cos(phi) + y * np.sin(theta) * np.sin(phi))

  # Reshape the complex weights into a 1-D vector
  w_vec = w.reshape(-1)

  # Multiply the weights by the phase shift and sum them up
  AF = np.sum(w_vec * np.exp(phase_shift))

  return AF

def antenna_element_pattern(theta: np.ndarray, phi: np.ndarray,
                             cos_factor_theta: float = 1.0, cos_factor_phi: float = 1.0,
                             max_gain_dBi: float = 0.0) -> np.ndarray:
  """
  Calculates the radiation pattern of a single antenna element using a raised cosine model.

  Args:
    theta: Elevation angles in radians (numpy.ndarray).
    phi: Azimuth angles in radians (numpy.ndarray).
    cos_factor_theta: Cosine power factor for theta (float, default=1.0).
    cos_factor_phi: Cosine power factor for phi (float, default=1.0).
    max_gain_dBi: Maximum gain of the element pattern in dBi (float, default=0.0).

  Returns:
    A numpy array containing the element pattern values in linear scale.
  """

  # Convert max gain from dBi to linear scale
  max_gain = dBi_to_linear(max_gain_dBi)

  # Calculate the radiation pattern
  pattern = max_gain * np.cos(theta) ** cos_factor_theta * np.cos(phi) ** cos_factor_phi

  return pattern


def find_angle_for_magnitude_np(thetas, magnitudes, target_magnitude, threshold):
    """
    Efficiently find the angle corresponding to the magnitude closest to a target value,
    within a specified threshold.

    Parameters:
        thetas : np.ndarray of angles
        magnitudes : np.ndarray of magnitudes (same shape as thetas)
        target_magnitude : float
        threshold : float

    Returns:
        angle (float) if found, else None
    """
    diffs = np.abs(magnitudes - target_magnitude)
    valid_indices = np.where(diffs <= threshold)[0]

    if len(valid_indices) == 0:
        return None  # No value within the threshold

    # Among valid ones, choose the one with minimum difference
    best_index = valid_indices[np.argmin(diffs[valid_indices])]
    return thetas[best_index]


def compute_gains(steering_theta, steering_phi):
    Nx = 40 # number of elements in the x-direction
    Ny = 30 # number of elements in the y-direction
    dx = 0.5  # spacing between elements in the x-direction (in wavelengths)
    dy = dx  # spacing between elements in the y-direction (in wavelengths)
    freq_GHz = 10 # frequency (GHz)

    ep_max_gain_dBi = 0 # Max gain of the element pattern (EP)
    cos_factor_theta = 1.2 # Raised cosine factor of the EP in theta
    cos_factor_phi = 1.2 # Raised cosine factor of the EP in phi

    # Derived antenna parameters
    f = 1e9 * freq_GHz  # convert frequency to Hz
    lambda_ = c / f  # wavelength (meters)
    k = 2 * pi / lambda_ # wave vector

    # Express grid spacing in meters
    dx_m = dx * lambda_
    dy_m = dy * lambda_

    # Compute approximate aperture directivity
    aperture_area = Nx * Ny * dx_m * dy_m
    D = 4 * pi * aperture_area / lambda_**2
    D_dBi = 10 * np.log10(D)

    # Estimate 3 dB beamwidth (BW) at broadside for the array aperature
    beamwidth_broadside_x = 0.886 * lambda_ / (Nx * dx_m)
    beamwidth_broadside_y = 0.886 * lambda_ / (Ny * dy_m)

    # Number of Array Elements
    num_elements = Nx * Ny

    # Print derived antenna parameters
    print('Estimated aperture directivity (dBi):', np.round(D_dBi, 1))
    print('Estimated 3 dB beamwidth at broadside (x):', np.round(beamwidth_broadside_x * 180 / pi, 1), 'degrees')
    print('Estimated 3 dB beamwidth at broadside (y):', np.round(beamwidth_broadside_y * 180 / pi, 1), 'degrees')
    print('Aperture dimensions (x):', np.round(Nx*dx_m, 2), 'meters')
    print('Aperture dimensions (y):', np.round(Ny*dy_m, 2), 'meters')
    print('Apreture area:', np.round(aperture_area, 2), 'm^2')
    print('Total number of antenna elements:', num_elements)

    # Steering angle
    theta0 = steering_theta # Beam steering angle in theta (degrees)
    phi0 = steering_phi # Beam steering angle in phi (degrees)

    x = np.arange(Nx) - (Nx-1) / 2
    y = np.arange(Ny) - (Ny-1) / 2

    x = x * dx_m
    y = y * dy_m

    X, Y = np.meshgrid(x, y)

    X_vec = X.reshape(-1)
    Y_vec = Y.reshape(-1)

    phase_weights = steering_vector(k=k,
                                xv=X,
                                yv=Y,
                                theta_deg=theta0,
                                phi_deg=phi0)
    
    # Define observation angles
    # theta_deg = np.linspace(-90, 90, 181)
    # phi_deg = np.linspace(-90, 90, 181)

    theta_deg = np.linspace(theta0-20, theta0+20, 41*20)
    phi_deg = np.linspace(phi0-20, phi0+20, 41*20)

    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    # Make a meshgrid of theta and phi
    THETA, PHI = np.meshgrid(theta, phi)

    # TODO: we need to speed this up and not compute the entire array

    # Compute element pattern over all THETA, PHI angles
    element_pattern = antenna_element_pattern(THETA, PHI,
                                            cos_factor_theta,
                                            cos_factor_phi,
                                            max_gain_dBi=ep_max_gain_dBi)

    # Calculate the array factor for each angle
    array_factor = np.zeros((len(theta), len(phi)), dtype=complex)

    for i, thi in enumerate(theta):
        for j, phj in enumerate(phi):
            array_factor[i, j] = element_pattern[i, j] * AF(thi, phj, x=X_vec, y=Y_vec, w=phase_weights, k=k)

        array_factor_dB = 10 * np.log10(abs(array_factor))
        array_gain_dBi = array_factor_dB

        # Normalize array_factor_dB
        # array_factor_dB_norm = array_factor_dB - np.max(array_factor_dB)
        # array_gain_dBi  = array_factor_dB_norm

        # Normalize array_factor_dB
        # array_gain_dBi = D_dBi - system_losses_dB + array_factor_dB_norm


    min_thres = np.max(array_factor_dB) - 50

    min_thres = np.max(array_gain_dBi) - 50
    array_gain_plot_dBi = array_gain_dBi.clip(min=min_thres)

    phi_zero_index = np.argmin(np.abs(phi_deg - phi0))
    theta_zero_index = np.argmin(np.abs(theta_deg - theta0))

    return {"thetas":theta_deg, "gains":array_gain_plot_dBi[:, phi_zero_index], 
            "phi_zero_index":phi_zero_index, "theta_zero_index":theta_zero_index}

    



if __name__ == "__main__": 
    # Nx = 40 # number of elements in the x-direction
    # Ny = 30 # number of elements in the y-direction
    # dx = 0.5  # spacing between elements in the x-direction (in wavelengths)
    # dy = dx  # spacing between elements in the y-direction (in wavelengths)
    # freq_GHz = 10 # frequency (GHz)


    # system_losses_dB = 0
    # ep_max_gain_dBi = 0 # Max gain of the element pattern (EP)
    # cos_factor_theta = 1.2 # Raised cosine factor of the EP in theta
    # cos_factor_phi = 1.2 # Raised cosine factor of the EP in phi

    # # Derived antenna parameters
    # f = 1e9 * freq_GHz  # convert frequency to Hz
    # lambda_ = c / f  # wavelength (meters)
    # k = 2 * pi / lambda_ # wave vector

    # # Express grid spacing in meters
    # dx_m = dx * lambda_
    # dy_m = dy * lambda_

    # # Compute approximate aperture directivity
    # aperture_area = Nx * Ny * dx_m * dy_m
    # D = 4 * pi * aperture_area / lambda_**2
    # D_dBi = 10 * np.log10(D)

    # # Estimate 3 dB beamwidth (BW) at broadside for the array aperature
    # beamwidth_broadside_x = 0.886 * lambda_ / (Nx * dx_m)
    # beamwidth_broadside_y = 0.886 * lambda_ / (Ny * dy_m)

    # # Number of Array Elements
    # num_elements = Nx * Ny

    # # Steering angle
    # theta0 = 0 # Beam steering angle in theta (degrees)
    # phi0 = 0 # Beam steering angle in phi (degrees)

    # # Print derived antenna parameters
    # print('Estimated aperture directivity (dBi):', np.round(D_dBi, 1))
    # print('Estimated 3 dB beamwidth at broadside (x):', np.round(beamwidth_broadside_x * 180 / pi, 1), 'degrees')
    # print('Estimated 3 dB beamwidth at broadside (y):', np.round(beamwidth_broadside_y * 180 / pi, 1), 'degrees')
    # print('Aperture dimensions (x):', np.round(Nx*dx_m, 2), 'meters')
    # print('Aperture dimensions (y):', np.round(Ny*dy_m, 2), 'meters')
    # print('Apreture area:', np.round(aperture_area, 2), 'm^2')
    # print('Total number of antenna elements:', num_elements)


    # x = np.arange(Nx) - (Nx-1) / 2
    # y = np.arange(Ny) - (Ny-1) / 2

    # x = x * dx_m
    # y = y * dy_m

    # X, Y = np.meshgrid(x, y)

    # X_vec = X.reshape(-1)
    # Y_vec = Y.reshape(-1)

    # phase_weights = steering_vector(k=k,
    #                             xv=X,
    #                             yv=Y,
    #                             theta_deg=theta0,
    #                             phi_deg=phi0)

    # phase_shift_rad = np.angle(phase_weights)
    # phase_shift_deg = np.degrees(phase_shift_rad)

    # # print("Phase shifts (deg)", phase_shift_deg)


    # # Define observation angles
    # # theta_deg = np.linspace(-90, 90, 181)
    # # phi_deg = np.linspace(-90, 90, 181)

    # theta_deg = np.linspace(theta0-20, theta0+20, 41*10)
    # phi_deg = np.linspace(phi0-20, phi0+20, 41*10)

    # theta = np.deg2rad(theta_deg)
    # phi = np.deg2rad(phi_deg)

    # # Make a meshgrid of theta and phi
    # THETA, PHI = np.meshgrid(theta, phi)

    # # Convert to azimuth and elevation
    # AZ, EL = thetaphi_to_azel(THETA, PHI)



    # # Compute element pattern over all THETA, PHI angles
    # element_pattern = antenna_element_pattern(THETA, PHI,
    #                                         cos_factor_theta,
    #                                         cos_factor_phi,
    #                                         max_gain_dBi=ep_max_gain_dBi)

    # # Calculate the array factor for each angle
    # array_factor = np.zeros((len(theta), len(phi)), dtype=complex)

    # for i, thi in enumerate(theta):
    #     for j, phj in enumerate(phi):
    #         array_factor[i, j] = element_pattern[i, j] * AF(thi, phj, x=X_vec, y=Y_vec, w=phase_weights, k=k)

    #     array_factor_dB = 10 * np.log10(abs(array_factor))
    #     array_gain_dBi = array_factor_dB

    #     # Normalize array_factor_dB
    #     # array_factor_dB_norm = array_factor_dB - np.max(array_factor_dB)
    #     # array_gain_dBi  = array_factor_dB_norm

    #     # Normalize array_factor_dB
    #     # array_gain_dBi = D_dBi - system_losses_dB + array_factor_dB_norm


    # min_thres = np.max(array_factor_dB) - 50
    # array_factor_plot_dB = array_factor_dB.clip(min=min_thres)

    # min_thres = np.max(array_gain_dBi) - 50
    # array_gain_plot_dBi = array_gain_dBi.clip(min=min_thres)


    # # Plot cross section of beam pattern
    # # For array_factor_plot_dB, plot the theta and phi radiation pattern cuts where the other variable are the steer angles, respectively
    # thres_dB = 30

    # # Find index where phi_deg and theta_deg are the steer angles, respectively
    # # phi_zero_index = np.where(phi_deg == phi0)[0][0]
    # # theta_zero_index = np.where(theta_deg == theta0)[0][0]

    # phi_zero_index = np.argmin(np.abs(phi_deg - phi0))
    # theta_zero_index = np.argmin(np.abs(theta_deg - theta0))


    # # Find the peak value in the data
    # peak_value_theta = np.max(array_gain_plot_dBi[:, phi_zero_index])
    # peak_value_phi = np.max(array_gain_plot_dBi[theta_zero_index, :])

    # # Set the lower y-axis limit to 30 dB below the peak
    # if np.max(array_gain_plot_dBi[:, phi_zero_index]) - np.min(array_gain_plot_dBi[:, phi_zero_index]) > thres_dB:
    #     lower_limit_theta = peak_value_theta - thres_dB

    # if np.max(array_gain_plot_dBi[theta_zero_index, :]) - np.min(array_gain_plot_dBi[theta_zero_index, :]) > thres_dB:
    #     lower_limit_phi = peak_value_phi - thres_dB

    # max_gain = np.max(array_gain_plot_dBi[:, phi_zero_index])
    # three_db_down = max_gain - 3

    # print("max_gain", max_gain)
    # left_half_thetas = theta_deg[:theta_zero_index]
    # right_half_thetas = theta_deg[theta_zero_index:]

    # left_half_gains = array_gain_plot_dBi[:, phi_zero_index][:theta_zero_index]
    # right_half_gains = array_gain_plot_dBi[:, phi_zero_index][theta_zero_index:]

    # left_theta  = find_angle_for_magnitude_np(left_half_thetas, left_half_gains, three_db_down, 0.2)
    # right_theta = find_angle_for_magnitude_np(right_half_thetas, right_half_gains, three_db_down, 0.2)
    # print("3dB down (left):", left_theta)
    # print("3dB down (right):", right_theta)
    # print("HPBW", right_theta-left_theta)

    # print("")
    # # Plot the theta cut (phi = 0)
    # plt.figure()
    # plt.plot(theta_deg, array_gain_plot_dBi[:, phi_zero_index], label='Array Gain (User 1)')
    # plt.plot(theta_deg+2.0, array_gain_plot_dBi[:, phi_zero_index], label='Array Gain (User 2)')
    # plt.xlabel('Theta (deg)')
    # plt.ylabel('Array Gain (dBi)')
    # plt.title(f'Steering Angle: theta={theta0}°, phi={phi0}; Array Gain: Theta Cut (Phi = ' + str(phi0) + '°)')
    # plt.grid(True)
    # # plt.ylim(lower_limit_theta, peak_value_theta+5)
    # plt.xlim([np.min(theta_deg), np.max(theta_deg)])
    # plt.legend()
    # # plt.xlim([theta0-10, theta0+10])
    # # plt.xticks(np.arange(theta0-10, theta0+10, 1))
    # # plt.yticks(np.arange(lower_limit_theta, peak_value_theta+5,2))
    # plt.axvline(x=theta0, color='black', linestyle='--', label='Steer Angle (Theta)')
    # plt.show()

    s_i_ratio = 30.79/30.69 # CHANGE ME
    print("s_i_ratio", s_i_ratio)
    beamwidth = 3.4
    thetas = [70]
    # thetas = [0, 20, 40, 60, 80] # CHANGE ME
    min_separation = []

    for theta in thetas:
        # keep shifting the user 2 beam pattern over incrementally until they intersect at the target
        # save this shift in angle and then normalize to one beamwidth

        result = compute_gains(theta, 0)
        thetas = result["thetas"]
        gains = result["gains"]
        phi_zero_index = result["phi_zero_index"]
        theta_zero_index = result["theta_zero_index"]

        # find the peak gain for current steering angle
        max_gain = np.max(gains)
        three_db_down = max_gain - 3

        left_half_thetas = thetas[:theta_zero_index]
        right_half_thetas = thetas[theta_zero_index:]

        left_half_gains = gains[:theta_zero_index]
        right_half_gains = gains[theta_zero_index:]

        # left_theta  = find_angle_for_magnitude_np(left_half_thetas, left_half_gains, three_db_down, 0.5)
        # right_theta = find_angle_for_magnitude_np(right_half_thetas, right_half_gains, three_db_down, 0.5)
        # print("3dB down (left):", left_theta)
        # print("3dB down (right):", right_theta)
        # print("HPBW", right_theta-left_theta)

        # # compute the next target
        # target_gain = max_gain / s_i_ratio
        # target_theta = find_angle_for_magnitude_np(right_half_thetas, right_half_gains, target_gain, 0.2)
        # print("target_gain", target_gain)
        # print("target_theta", target_theta)

        # shifts = np.linspace(0, beamwidth, 100) # CHANGE ME
        # diffs = []
        # for shift in shifts:
        #     shifted_thetas = thetas + shift
            
        #     shifted_left_half_thetas = shifted_thetas[:theta_zero_index]
        #     shifted_right_half_thetas = shifted_thetas[theta_zero_index:]

        #     left_half_gains = gains[:theta_zero_index]
        #     right_half_gains = gains[theta_zero_index:]

        #     user_2_theta = find_angle_for_magnitude_np(shifted_left_half_thetas, left_half_gains, target_gain, 0.2)

        #     diffs.append(abs(user_2_theta-target_theta))
        
        # diffs = np.array(diffs)

        # print("diffs", diffs)
        # best_shift = shifts[np.argmin(diffs)]
        # print("shifts", shifts)
        # print("best_shift", best_shift)
        # print("min distance btw users (normalized)", best_shift/beamwidth)
        # min_separation.append(best_shift/beamwidth)


        plt.figure()
        plt.plot(thetas, gains, label='Array Gain (User 1)')
        # plt.plot(thetas+best_shift, gains, label='Array Gain (User 2)')
        plt.xlabel('Theta (deg)')
        plt.ylabel('Array Gain (dBi)')
        plt.title(f'Steering Angle: theta={theta}°, phi={0.0}; Array Gain: Theta Cut (Phi = ' + str(0.0) + '°)')
        plt.grid(True)
        # plt.ylim(lower_limit_theta, peak_value_theta+5)
        plt.xlim([np.min(thetas), np.max(thetas)])
        plt.legend()
        # plt.xlim([theta0-10, theta0+10])
        # plt.xticks(np.arange(theta0-10, theta0+10, 1))
        # plt.yticks(np.arange(lower_limit_theta, peak_value_theta+5,2))
        plt.axvline(x=theta, color='black', linestyle='--', label='Steer Angle (Theta)')
        plt.show()

    plt.plot(thetas, min_separation)
    plt.show()
        







