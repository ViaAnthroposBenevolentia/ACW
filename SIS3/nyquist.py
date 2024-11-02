import control as ct
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Define the transfer function: G_k(s) = 2200000 / (2.4s^2 + 1400s + 2300000)
numerator = [2200000]
denominator = [2.4, 1400, 2300000]
G_k = ct.TransferFunction(numerator, denominator)

# Generate a frequency range for the Nyquist plot
omega = np.logspace(-1, 5, 5000)  # From 0.1 to 100,000 rad/s

# Compute the frequency response
mag, phase, omega = ct.frequency_response(G_k, omega)

# Extract real and imaginary parts
real = mag.flatten() * np.cos(phase.flatten())
imag = mag.flatten() * np.sin(phase.flatten())

# Initialize the plot
fig, ax = plt.subplots(figsize=(15, 9))

# Plot the Nyquist diagram
ax.plot(real, imag, 'b', label='Nyquist Plot')
ax.plot(real, -imag, 'b')  # Mirror image for negative frequencies

# Add critical point (-1, 0)
ax.plot(-1, 0, 'rx', label='Critical Point (-1, 0)')

# Add arrows to indicate the direction of increasing frequency
arrow_freqs = np.logspace(-1, 5, 20)
arrow_indices = [np.argmin(np.abs(omega - f)) for f in arrow_freqs]

for idx in arrow_indices[:-1]:  # Skip the last index to avoid out-of-bounds error
    # Calculate the difference in x and y for the arrow direction
    dx = real[idx + 1] - real[idx]
    dy = imag[idx + 1] - imag[idx]

    # Plot the arrow to indicate frequency progression
    ax.arrow(real[idx], imag[idx], dx, dy,
             head_width=0.05 * mag[idx], head_length=0.05 * mag[idx],
             fc='k', ec='k', linewidth=0.5)

# Set labels and title
ax.set_title(r'Nyquist Plot of $G_k(s) = \frac{2200000}{2.4s^2 + 1400s + 2300000}$', fontsize=16)
ax.set_xlabel('Real Part', fontsize=14)
ax.set_ylabel('Imaginary Part', fontsize=14)
ax.grid(True)
ax.legend()

# Ensure aspect ratio is equal
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)
ax.set_aspect('equal', 'box')

# Calculate gain and phase margins
gm, pm, wg, wp = ct.margin(G_k)
gm_db = 20 * np.log10(gm) if gm != np.inf else np.inf

# Display margins on the plot
textstr = '\n'.join((
    r'Gain Margin: %.2f dB' % gm_db,
    r'Gain Crossover Freq: %.2f rad/s' % wg,
    r'Phase Margin: %.2f°' % pm,
    r'Phase Crossover Freq: %.2f rad/s' % wp))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 1.2, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

# Add a unit circle to the plot
unit_circle = plt.Circle((0, 0), 1, color='gray', linestyle='--', fill=False)
ax.add_artist(unit_circle)

# Highlight the phase margin visually from the left side of the Nyquist plot
if pm > 0 and wp is not None:
    # Find the point on the Nyquist plot where the magnitude is closest to 1 (|G(jω)| = 1) and is to the left of the critical point
    left_idx = np.argmin(np.abs(real + 1))  # Find the point closest to (-1, 0)
    crossover_real = real[left_idx]
    crossover_imag = imag[left_idx]

    # Draw a line from origin point to the crossover point on the left side
    ax.plot([crossover_real, 0], [crossover_imag, 0], 'g--', label=f'Phase Margin Line ({pm:.2f}°)')
    ax.plot(crossover_real, crossover_imag, 'go', label='Phase Crossover Point')
    ax.text(crossover_real+0.1, crossover_imag-0.2, 'Phase Crossover Point', color='green', fontsize=10, fontweight='bold', ha='center', va='bottom')

    # Convert phase margin to radians if it's in degrees
    pm_rad = np.deg2rad(pm)  # Ensure phase margin angle is in radians

    # Set the radius for the arc (adjust if necessary for clarity)
    arc_radius = 0.5

    # Create an arc patch at the origin, going downwards
    # `theta1` starts from -180° (left side) and moves counterclockwise to -180° + phase margin angle
    arc = patches.Arc((0, 0), 2*arc_radius, 2*arc_radius,
                      theta1=-180, theta2=-180 + np.rad2deg(pm_rad),
                      color='green', linestyle='-', linewidth=2)
    ax.add_patch(arc)

    # Position the angle label near the arc
    # Compute the midpoint coordinates for the label at half of the phase margin angle
    label_angle = -180 + pm_rad / 2  # Midpoint angle in radians
    angle_x = arc_radius * np.cos(label_angle)  # X-coordinate at midpoint
    angle_y = arc_radius * np.sin(label_angle)  # Y-coordinate at midpoint
    ax.text(angle_x-0.3, angle_y-0.4, f"PM: {pm:.2f}°", color='green', fontsize=10, fontweight='bold', ha='center')


# Save the figure
plt.savefig("figures/Nyquist_Plot.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
