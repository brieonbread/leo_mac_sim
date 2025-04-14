import numpy as np
import matplotlib.pyplot as plt

# Array parameters
N = 10                      # Number of antennas
wavelength = 1.0            # Wavelength (arbitrary units)
d = wavelength / 2          # Element spacing (half-wavelength)
k = 2 * np.pi / wavelength  # Wave number

# Function to generate a steering vector for an angle (in degrees)
def steering_vector(theta_deg):
    theta_rad = np.radians(theta_deg)
    # Generate the steering vector (not normalized; each element has unit magnitude)
    return np.exp(1j * k * d * np.arange(N) * np.sin(theta_rad))

# Define the scanning angles (the grid over which we'll compute the beam pattern)
angles_deg = np.linspace(-90, 90, 181)  # from -90 to 90 degrees

# Define a list of beam steering angles (the desired main lobe directions)
steering_angles = [0, 30, 60]  # Example: steer the beam at 0°, 30° and 60°

plt.figure(figsize=(10, 6))

# Loop through each steering angle to compute its beam pattern
for theta_0 in steering_angles:
    # The weight vector is the steering vector that steers the beam to theta_0.
    w = steering_vector(theta_0).reshape(N, 1)
    
    # Compute the steering vectors for all scanning angles.
    # "n" represents the element indices.
    n = np.arange(N).reshape(-1, 1)
    angles_rad = np.deg2rad(angles_deg)
    # Each column in "a" is the steering vector for a scanning angle.
    a = np.exp(1j * k * d * n * np.sin(angles_rad))
    
    # Compute the array response (beam pattern) for each scanning angle.
    # We calculate the inner product of the beamforming weight and the scanning vector.
    response = w.conj().T @ a  # Response is a 1 x (number of angles) array.
    
    # Compute the magnitude of the response
    beam_pattern = np.abs(response).flatten()
    # Normalize the beam pattern so that its maximum value is 1.
    beam_pattern_normalized = beam_pattern / np.max(beam_pattern)
    # Convert to decibels (dB)
    beam_pattern_dB = 20 * np.log10(beam_pattern_normalized + 1e-12)  # adding epsilon avoids log(0)
    
    # Plot the beam pattern for the current steering angle.
    plt.plot(angles_deg, beam_pattern_dB, label=f"Steered to {theta_0}°")

# Set plot labels and grid
plt.xlabel("Angle (degrees)")
plt.ylabel("Beam Pattern (dB)")
plt.title("Beam Patterns for Different Steering Angles")
plt.legend()
plt.grid(True)
plt.ylim([-60, 0])   # Display down to -60 dB for sidelobes
plt.show()
