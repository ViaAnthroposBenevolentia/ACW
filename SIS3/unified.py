import os
import control as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.patches as patches
import warnings

def ensure_directories(directories):
    """
    Ensure that the specified directories exist. Create them if they don't.
    """
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def define_transfer_function():
    """
    Define and return the transfer function G_k(s).
    """
    numerator = [2200000]
    denominator = [2.4, 1400, 2300000]
    G_k = ct.TransferFunction(numerator, denominator)
    return G_k

def step_response_analysis(G_k):
    """
    Perform step response analysis, plot the response, and save metrics.
    """
    # Generate a custom time vector with more steps (e.g., 5000 points over 0.025 seconds)
    time = np.linspace(0, 0.025, 5000)  # 5000 points between 0 and 0.025 seconds

    # Generate the step response
    time, response = ct.step_response(G_k, T=time)

    # Plot the step response
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.plot(time, response, linewidth=3, label='Step Response')
    ax.set_title(r"Step Response of $G_k(s) = \frac{2200000}{2.4s^2 + 1400s + 2300000}$", fontsize=16)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel(r"Flow Rate ($m^3/h$)", fontsize=14)
    ax.grid(True)
    plt.grid(True)

    ax.set_xlim(left=0)  # Ensure x-axis starts at 0
    ax.set_ylim(bottom=0)  # Ensure y-axis starts at 0

    # Get step response information
    info = ct.step_info(G_k)

    # Steady-state value
    ss_value = info.get('SteadyStateValue', np.nan)
    ax.axhline(y=ss_value, color='r', linestyle='--', label='Steady-State Value')

    # Annotate SS error
    if not np.isnan(ss_value):
        ss_error = abs(1 - ss_value) * 100
        ax.axhline(y=1, color='black', linestyle='--')
        x_pos = time[-1] * 0.98
        ax.annotate(
            '', xy=(x_pos, 1), xytext=(x_pos, ss_value),
            arrowprops=dict(arrowstyle='<->', color='black')
        )
        ax.annotate(
            f"SS Error:\n{ss_error:.2f}%", xy=(x_pos, 1),
            xytext=(x_pos - time[-1]*0.2, 1 + 0.05),
            arrowprops=dict(arrowstyle='->', color='black')
        )

    # Rise time
    rise_time = info.get('RiseTime', np.nan)

    # Calculate 10% and 90% thresholds
    threshold_10 = 0.1 * ss_value
    threshold_90 = 0.9 * ss_value

    # Find the indices where the response reaches 10% and 90% of the SS value
    try:
        index_10 = next(i for i, r in enumerate(response) if r >= threshold_10)
        index_90 = next(i for i, r in enumerate(response) if r >= threshold_90)
    except StopIteration:
        index_10, index_90 = -1, -1

    if index_10 != -1 and index_90 != -1:
        # Get the corresponding times
        time_10 = time[index_10]
        time_90 = time[index_90]

        # Plot rise time markers and lines
        ax.vlines(x=time_10, ymin=0, ymax=threshold_10, color='g', linestyle='--', label='Rise Time')
        ax.hlines(y=threshold_10, xmin=0, xmax=time_10, color='g', linestyle='--')
        ax.plot(time_10, threshold_10, 'go')

        ax.vlines(x=time_90, ymin=0, ymax=threshold_90, color='g', linestyle='--')
        ax.hlines(y=threshold_90, xmin=0, xmax=time_90, color='g', linestyle='--')
        ax.plot(time_90, threshold_90, 'go')

        # Annotate Rise Time
        ax.annotate(
            '', xy=(time_90, threshold_10), xytext=(time_10, threshold_10),
            arrowprops=dict(arrowstyle='<->', color='black')
        )
        ax.text(
            (time_10 + time_90) / 2, threshold_10 + 0.02,
            f"Rise Time:\n{rise_time:.4f}s", ha='center', fontsize=12
        )

    # Peak time and value
    peak_time = info.get('PeakTime', np.nan)
    peak_value = info.get('Peak', np.nan)
    overshoot = info.get('Overshoot', np.nan)

    if not np.isnan(peak_time) and not np.isnan(peak_value):
        ax.vlines(x=peak_time, ymin=0, ymax=peak_value, color='m', linestyle='--', label='Peak Time')
        ax.hlines(y=peak_value, xmin=0, xmax=peak_time, color='m', linestyle='--')
        ax.plot(peak_time, peak_value, 'mo')

        # Annotate Overshoot
        if peak_value > 1:
            x_os = time[len(time)//20]
            ax.annotate(
                '', xy=(x_os, peak_value), xytext=(x_os, 1),
                arrowprops=dict(arrowstyle='<->', color='black')
            )
            ax.text(
                x_os - time[-1]*0.02, (peak_value + 1) / 2,
                "Overshoot", rotation=90, va='center', fontsize=12
            )
            ax.text(
                peak_time + time[-1]*0.02, peak_value - 0.02,
                f"Overshoot: {overshoot:.2f}%", fontsize=12
            )

    # Settling time
    settling_time = info.get('SettlingTime', np.nan)
    if not np.isnan(settling_time):
        settling_index = next((i for i, t in enumerate(time) if t >= settling_time), -1)
        if settling_index != -1:
            settling_value = response[settling_index]
            ax.vlines(x=settling_time, ymin=0, ymax=settling_value, color='b', linestyle='--', label='Settling Time')
            ax.plot(settling_time, settling_value, 'bo')

            # Annotate Settling Time
            y_st = ss_value * 0.7
            ax.annotate(
                '', xy=(settling_time, y_st), xytext=(0, y_st),
                arrowprops=dict(arrowstyle='<->', color='black')
            )
            ax.text(
                settling_time / 3, y_st + 0.02,
                f"Settling Time: {settling_time:.3f}s", fontsize=12
            )

    # Add ticks with custom labels
    yticks = [0.1*ss_value, 0.9*ss_value, ss_value, 1]
    yticklabels = [
        f"10% Rise: {0.1*ss_value:.3f}",
        f"90% Rise: {0.9*ss_value:.3f}",
        f"SS value: {ss_value:.3f}",
        "Input: 1"
    ]
    if not np.isnan(peak_value):
        yticks.append(peak_value)
        yticklabels.append(f"Peak Value: {peak_value:.3f}")

    xticks = []
    xticklabels = []
    if index_10 != -1 and index_90 != -1:
        xticks.extend([time_10, time_90])
        xticklabels.extend([f"{time_10:.4f}s", f"{time_90:.4f}s"])
    if not np.isnan(peak_time):
        xticks.append(peak_time)
        xticklabels.append(f"{peak_time:.4f}s")
    if not np.isnan(settling_time):
        xticks.append(settling_time)
        xticklabels.append(f"{settling_time:.3f}s")
    xticks.append(time[-1])
    xticklabels.append(f"{time[-1]:.3f}s")

    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xticklabels(xticklabels)

    ax.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig("figures/Step_Response.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Display the plot
    plt.figure(figsize=(15, 9))
    img = plt.imread("figures/Step_Response.png")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Table and csv file
    # Create a DataFrame to hold the metrics
    metrics = {
        'Metric': ['Rise Time', 'Settling Time', 'Settling Min', 'Settling Max',
                   'Overshoot', 'Undershoot', 'Peak', 'Peak Time', 'Steady-State Value', 'Steady-State Error'],
        'Value': [
            info.get('RiseTime', np.nan),
            info.get('SettlingTime', np.nan),
            info.get('SettlingMin', np.nan),
            info.get('SettlingMax', np.nan),
            info.get('Overshoot', np.nan),
            info.get('Undershoot', np.nan),
            info.get('Peak', np.nan),
            info.get('PeakTime', np.nan),
            info.get('SteadyStateValue', np.nan),
            abs(1 - info.get('SteadyStateValue', np.nan))
        ]
    }

    df_metrics = pd.DataFrame(metrics)

    # Print the table using tabulate for better formatting
    print("\nStep Response Metrics:")
    print(tabulate(df_metrics, headers='keys', tablefmt='pretty', floatfmt=".6f"))

    # Save the metrics to a CSV file
    df_metrics.to_csv('csv/step.csv', index=False)

def impulse_response_analysis(G_k):
    """
    Perform impulse response analysis, plot the response, and save metrics.
    """
    # Generate a custom time vector with more steps (e.g., 5000 points over 0.025 seconds)
    time = np.linspace(0, 0.025, 5000)  # 5000 points between 0 and 0.025 seconds

    # Generate the impulse response
    time, response = ct.impulse_response(G_k, T=time)

    # Plot the impulse response
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.plot(time, response, linewidth=3, label='Impulse Response')
    ax.set_title(r"Impulse Response of $G_k(s) = \frac{2200000}{2.4s^2 + 1400s + 2300000}$", fontsize=16)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Amplitude", fontsize=14)
    ax.grid(True)

    # Ensure axes start at zero
    ax.set_xlim(left=0)

    # Find the peak value and time
    peak_index = np.argmax(response)
    peak_time = time[peak_index]
    peak_value = response[peak_index]

    # Mark the peak point
    ax.plot(peak_time, peak_value, 'ro', label='Peak')

    # Draw vertical and horizontal lines to the peak point
    ax.vlines(x=peak_time, ymin=min(response), ymax=peak_value, color='r', linestyle='--')
    ax.hlines(y=peak_value, xmin=0, xmax=peak_time, color='r', linestyle='--')

    # Annotate the peak value and time
    ax.text(peak_time, peak_value + peak_value * 0.1,
            f"Peak Value: {peak_value:.3f}\nPeak Time: {peak_time:.5f}s",
            ha='center', fontsize=12)

    # Show legend
    ax.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig("figures/Impulse_Response.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Display the plot
    plt.figure(figsize=(15, 9))
    img = plt.imread("figures/Impulse_Response.png")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Compute the area under the impulse response (should be equal to the gain for stable systems)
    area = np.trapz(response, time)

    # Create a DataFrame for the metrics
    metrics = {
        'Metric': ['Peak Time', 'Peak Value', 'Area Under Curve'],
        'Value': [peak_time, peak_value, area]
    }

    df_metrics = pd.DataFrame(metrics)

    # Print the table using tabulate
    print("\nImpulse Response Metrics:")
    print(tabulate(df_metrics, headers='keys', tablefmt='pretty', floatfmt=".6f"))

    # Save the metrics to a CSV file
    df_metrics.to_csv('csv/impulse.csv', index=False)

def pole_zero_map(G_k):
    """
    Plot the pole-zero map of the transfer function and save the figure.
    """
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Compute the poles and zeros
    try:
        zeros = ct.zeros(G_k)
        poles = ct.poles(G_k)
    except Exception as e:
        print(f"Error computing poles and zeros: {e}")
        zeros = np.array([])
        poles = np.array([])

    # Create a figure for the pole-zero map
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot zeros as 'o' and poles as 'x'
    if zeros.size > 0:
        ax.plot(np.real(zeros), np.imag(zeros), 'go', markersize=10, label='Zeros')
    else:
        print("No zeros to plot.")

    if poles.size > 0:
        ax.plot(np.real(poles), np.imag(poles), 'rx', markersize=10, label='Poles')
    else:
        print("No poles to plot.")

    # Annotate the plot
    ax.set_title('Pole-Zero Map of the Transfer Function', fontsize=16)
    ax.set_xlabel('Real Axis', fontsize=14)
    ax.set_ylabel('Imaginary Axis', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.legend()

    # Set axis limits for better visibility
    if poles.size > 0 or zeros.size > 0:
        all_real = np.concatenate((np.real(poles), np.real(zeros))) if zeros.size > 0 else np.real(poles)
        all_imag = np.concatenate((np.imag(poles), np.imag(zeros))) if zeros.size > 0 else np.imag(poles)
        real_min = np.min(all_real) - abs(np.min(all_real)) * 0.2 - 1 if np.min(all_real) != 0 else -10
        real_max = np.max(all_real) + abs(np.max(all_real)) * 0.2 + 1 if np.max(all_real) != 0 else 10
        imag_min = np.min(all_imag) - abs(np.min(all_imag)) * 0.2 - 1 if np.min(all_imag) != 0 else -10
        imag_max = np.max(all_imag) + abs(np.max(all_imag)) * 0.2 + 1 if np.max(all_imag) != 0 else 10
    else:
        real_min, real_max = -10, 10
        imag_min, imag_max = -10, 10

    # Adjust limits to include origin if necessary
    real_min = min(real_min, -1)
    real_max = max(real_max, 1)
    imag_min = min(imag_min, -1)
    imag_max = max(imag_max, 1)

    ax.set_xlim([real_min, real_max])
    ax.set_ylim([imag_min, imag_max])

    # Annotate pole and zero values
    offset = (real_max - real_min) * 0.03  # Offset for annotations
    if zeros.size > 0:
        for z in zeros:
            ax.annotate(f'Zero at {z:.2f}', xy=(np.real(z), np.imag(z)),
                        xytext=(np.real(z) + offset, np.imag(z) + offset),
                        arrowprops=dict(arrowstyle='->', color='green'))
    if poles.size > 0:
        for p in poles:
            ax.annotate(f'Pole at {p:.2f}', xy=(np.real(p), np.imag(p)),
                        xytext=(np.real(p) + offset, np.imag(p) + offset),
                        arrowprops=dict(arrowstyle='->', color='red'))

    # Add stability regions for continuous-time systems
    # Shade the right-half plane (RHP) to indicate instability region
    ax.fill_between([0, real_max], imag_min, imag_max, color='red', alpha=0.1, label='Unstable Region')
    # Shade the left-half plane (LHP) to indicate stability region
    ax.fill_between([real_min, 0], imag_min, imag_max, color='green', alpha=0.1, label='Stable Region')

    # Bring legend to the front
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Display poles and zeros in a table
    data = []
    for p in poles:
        data.append(['Pole', f'{p:.6f}', f'{np.real(p):.6f}', f'{np.imag(p):.6f}'])
    for z in zeros:
        data.append(['Zero', f'{z:.6f}', f'{np.real(z):.6f}', f'{np.imag(z):.6f}'])

    table_headers = ['Type', 'Location', 'Real Part', 'Imaginary Part']
    print("\nPoles and Zeros of the Transfer Function:")
    print(tabulate(data, headers=table_headers, tablefmt='pretty'))

    # Save the figure
    plt.tight_layout()
    plt.savefig("figures/Pole_Zero_Map.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Display the plot
    plt.figure(figsize=(10, 8))
    img = plt.imread("figures/Pole_Zero_Map.png")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def nyquist_plot_analysis(G_k):
    """
    Generate the Nyquist plot and save the figure.
    """
    # Generate a frequency range for the Nyquist plot
    omega = np.logspace(-1, 5, 5000)  # From 0.1 to 100,000 rad/s

    # Compute the frequency response
    mag, phase, omega = ct.freqresp(G_k, omega)

    # Extract real and imaginary parts
    real = mag.flatten() * np.cos(phase.flatten())
    imag = mag.flatten() * np.sin(phase.flatten())

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(16, 12))

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
    ax.set_aspect('equal', 'box')

    # Calculate gain and phase margins
    gm, pm, wg, wp = ct.margin(G_k)
    gm_db = 20 * np.log10(gm) if gm != np.inf else np.inf

    # Display margins on the plot
    textstr = '\n'.join((
        f'Gain Margin: {gm_db:.2f} dB',
        f'Gain Crossover Freq: {wg:.2f} rad/s',
        f'Phase Margin: {pm:.2f}°',
        f'Phase Crossover Freq: {wp:.2f} rad/s'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0, 1.05, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props)

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
        ax.text(crossover_real + 0.1, crossover_imag - 0.2,
                'Phase Crossover Point', color='green', fontsize=10, fontweight='bold', ha='center', va='bottom')

        # Convert phase margin to radians if it's in degrees
        pm_rad = np.deg2rad(pm)  # Ensure phase margin angle is in radians

        # Set the radius for the arc (adjust if necessary for clarity)
        arc_radius = 0.5

        # Create an arc patch at the origin, going downwards
        # `theta1` starts from -180° (left side) and moves counterclockwise to -180° + phase margin angle
        arc = patches.Arc((0, 0), 2*arc_radius, 2*arc_radius,
                          theta1=-180, theta2=-180 + np.rad2deg(pm_rad),
                          color='black', linestyle='-', linewidth=2)
        ax.add_patch(arc)

        # Position the angle label near the arc
        # Compute the midpoint coordinates for the label at half of the phase margin angle
        label_angle = -180 + np.rad2deg(pm_rad)/2  # Midpoint angle in degrees
        angle_x = arc_radius * np.cos(np.deg2rad(label_angle))  # X-coordinate at midpoint
        angle_y = arc_radius * np.sin(np.deg2rad(label_angle))  # Y-coordinate at midpoint
        ax.text(angle_x - 0.3, angle_y, f"PM: {pm:.2f}°",
                color='green', fontsize=10, fontweight='bold', ha='center')

    # Save the figure
    plt.savefig("figures/Nyquist_Plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Display the plot
    plt.figure(figsize=(16, 12))
    img = plt.imread("figures/Nyquist_Plot.png")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def bode_plot_analysis(G_k):
    """
    Generate the Bode plot and save the figure.
    """
    # Generate a frequency range for the Bode plot
    omega = np.logspace(-1, 5, 5000)  # Frequency range from 0.1 to 100,000 rad/s

    # Compute Bode magnitude and phase
    mag, phase, omega = ct.bode(G_k, omega, plot=False)  # Plot=False to avoid automatic plotting

    # Initialize the figure for the Bode plot
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot magnitude in dB
    ax_mag.semilogx(omega, 20 * np.log10(mag), 'b', label='Magnitude (dB)')
    ax_mag.set_ylabel('Magnitude (dB)', fontsize=14)
    ax_mag.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_mag.set_title(r'Bode Plot of $G_k(s) = \frac{2200000}{2.4s^2 + 1400s + 2300000}$', fontsize=16)

    # Plot phase in degrees
    ax_phase.semilogx(omega, np.degrees(phase), 'g', label='Phase (°)')
    ax_phase.set_ylabel('Phase (°)', fontsize=14)
    ax_phase.set_xlabel('Frequency (rad/s)', fontsize=14)
    ax_phase.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Calculate gain and phase margins
    gm, pm, wg, wp = ct.margin(G_k)
    gm_db = 20 * np.log10(gm) if gm != np.inf else np.inf

    # Annotate gain margin on the magnitude plot
    if gm != np.inf and not np.isnan(gm_db):
        ax_mag.axhline(-gm_db, color='r', linestyle='--', label=f'Gain Margin: {gm_db:.2f} dB')
        ax_mag.axvline(wg, color='r', linestyle='--', label=f'Gain Crossover: {wg:.2f} rad/s')
        ax_mag.plot(wg, -gm_db, 'ro')  # Mark the gain crossover point

    # Annotate phase margin on the phase plot
    if pm != np.inf and not np.isnan(pm):
        ax_phase.axhline(-180 + pm, color='m', linestyle='--', label=f'Phase Margin: {pm:.2f}°')
        ax_phase.axvline(wp, color='m', linestyle='--', label=f'Phase Crossover: {wp:.2f} rad/s')
        ax_phase.plot(wp, -180 + pm, 'mo')  # Mark the phase crossover point

    # Add legends
    ax_mag.legend(loc='best')
    ax_phase.legend(loc='best')

    # Save the figure
    plt.tight_layout()
    plt.savefig("figures/Bode_Plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Display the plot
    plt.figure(figsize=(12, 10))
    img = plt.imread("figures/Bode_Plot.png")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def main():
    """
    Main function to perform all analyses.
    """
    # Ensure necessary directories exist
    ensure_directories(['figures', 'csv'])

    # Define the transfer function
    G_k = define_transfer_function()

    # Perform Step Response Analysis
    step_response_analysis(G_k)

    # Perform Impulse Response Analysis
    impulse_response_analysis(G_k)

    # Plot Pole-Zero Map
    pole_zero_map(G_k)

    # Generate Nyquist Plot
    nyquist_plot_analysis(G_k)

    # Generate Bode Plot
    bode_plot_analysis(G_k)

if __name__ == "__main__":
    main()
