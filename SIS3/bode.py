import control as ct
import matplotlib.pyplot as plt
import numpy as np

# Define the transfer function: G_k(s) = 2200000 / (2.4s^2 + 1400s + 2300000)
numerator = [2200000]
denominator = [2.4, 1400, 2300000]
G_k = ct.TransferFunction(numerator, denominator)

# Generate a frequency range for the Bode plot
omega = np.logspace(-1, 5, 5000)  # Frequency range from 0.1 to 100,000 rad/s

# Compute Bode magnitude and phase
mag, phase, omega = ct.bode(G_k, omega, plot=False)  # Plot=False to avoid automatic plotting

# Initialize the figure for the Bode plot
fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot magnitude in dB
ax_mag.plot(omega, 20 * np.log10(mag), 'b', label='Magnitude (dB)')
ax_mag.set_ylabel('Magnitude (dB)', fontsize=14)
ax_mag.grid(True, which='both', linestyle='--', linewidth=0.5)
ax_mag.set_xscale('log')
ax_mag.set_title(r'Bode Plot of $G_k(s) = \frac{2200000}{2.4s^2 + 1400s + 2300000}$', fontsize=16)

# Plot phase in degrees
ax_phase.plot(omega, np.degrees(phase), 'g', label='Phase (°)')
ax_phase.set_ylabel('Phase (°)', fontsize=14)
ax_phase.set_xlabel('Frequency (rad/s)', fontsize=14)
ax_phase.grid(True, which='both', linestyle='--', linewidth=0.5)
ax_phase.set_xscale('log')

# Calculate gain and phase margins
gm, pm, wg, wp = ct.margin(G_k)
gm_db = 20 * np.log10(gm) if gm != np.inf else np.inf

# Annotate gain margin on the magnitude plot
if gm != np.inf:
    ax_mag.axhline(-gm_db, color='r', linestyle='--', label=f'Gain Margin: {gm_db:.2f} dB')
    ax_mag.axvline(wg, color='r', linestyle='--', label=f'Gain Crossover: {wg:.2f} rad/s')
    ax_mag.plot(wg, -gm_db, 'ro')  # Mark the gain crossover point

# Annotate phase margin on the phase plot
if pm != np.inf:
    ax_phase.axhline(-180 + pm, color='m', linestyle='--', label=f'Phase Margin: {pm:.2f}°')
    ax_phase.axvline(wp, color='m', linestyle='--', label=f'Phase Crossover: {wp:.2f} rad/s')
    ax_phase.plot(wp, -180, 'mo')  # Mark the phase crossover point

# Add legends
ax_mag.legend(loc='best')
ax_phase.legend(loc='best')

# Save the figure
plt.savefig("figures/Bode_Plot.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
