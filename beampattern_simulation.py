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
    Nx = 30 # number of elements in the x-direction
    Ny = 20 # number of elements in the y-direction
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

    theta_deg = np.linspace(theta0-10, theta0+10, 41*40) # CHANGE ME
    phi_deg = np.linspace(phi0-10, phi0+10, 41*40) # CHANGE ME

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
    # 40x30 array
    # s_i_ratio = 30.79/30.69 # CHANGE ME, S/I=1.003258390355
    # beamwidth = 3.4 # CHANGE ME

    # 20x30 array
    s_i_ratio = 27.781/27.677 # S/I=1.003757632691
    beamwidth = 4.602

    steering_thetas = [70]
    # steering_thetas = [0, 10, 20, 30, 40, 50, 60, 70] # CHANGE ME
    min_separation = []

    for theta in steering_thetas:
        print(f"Processing theta = {theta}")
        # keep shifting the user 2 beam pattern over incrementally until they intersect at the target
        # save this shift in angle and then normalize to one beamwidth

        result = compute_gains(theta, 0)
        thetas = result["thetas"]
        gains = result["gains"]
        phi_zero_index = result["phi_zero_index"]
        theta_zero_index = result["theta_zero_index"]

        # # find the peak gain for current steering angle
        # max_gain = np.max(gains)

        # # split the pattern into left/right half for easier processing
        # left_half_thetas = thetas[:theta_zero_index]
        # right_half_thetas = thetas[theta_zero_index:]

        # left_half_gains = gains[:theta_zero_index]
        # right_half_gains = gains[theta_zero_index:]

        # # compute the next target
        # target_gain = max_gain / s_i_ratio
        # target_theta = find_angle_for_magnitude_np(right_half_thetas, right_half_gains, target_gain, 0.2) # CHANGE ME
        # print("target_gain", target_gain)
        # print("target_theta", target_theta)

        # shifts = np.linspace(0, beamwidth, 1000) # CHANGE ME
        # diffs = []
        # for shift in shifts:
        #     shifted_thetas = thetas + shift
            
        #     shifted_left_half_thetas = shifted_thetas[:theta_zero_index]
        #     shifted_right_half_thetas = shifted_thetas[theta_zero_index:]

        #     left_half_gains = gains[:theta_zero_index]
        #     right_half_gains = gains[theta_zero_index:]

        #     user_2_theta = find_angle_for_magnitude_np(shifted_left_half_thetas, left_half_gains, target_gain, 0.2) # CHANGE ME

        #     diffs.append(abs(user_2_theta-target_theta))
        
        # diffs = np.array(diffs)
        # best_shift = shifts[np.argmin(diffs)]
        # min_separation.append(best_shift/beamwidth)

        # print("diffs", diffs)
        # print("shifts", shifts)
        # print("best_shift", best_shift)
        # print("min distance btw users (normalized)", best_shift/beamwidth)
        
        best_shift = 2.388438
        plt.figure()
        plt.plot(thetas, gains, label='Array Gain (User 1)')
        plt.plot(thetas+best_shift, gains, label='Array Gain (User 2)')
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

    print("steering_thetas", steering_thetas)
    print("min_separation", min_separation)
    plt.plot(steering_thetas, min_separation)
    plt.xlabel("steering angle (theta)")
    plt.ylabel("min user separation (normalized by HPBW @ boresight)")
    plt.show()
        







