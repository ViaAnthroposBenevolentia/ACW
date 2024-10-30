import control as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

# 1. Define the transfer function: G_k(s) = 2200000 / (2.4s^2 + 1400s + 2300000)
numerator = [2200000]
denominator = [2.4, 1400, 2300000]
G_k = ct.TransferFunction(numerator, denominator)

# 2. Generate a custom time vector with more steps (e.g., 5000 points over 0.025 seconds)
time = np.linspace(0, 0.025, 5000)  # 5000 points between 0 and 0.025 seconds

# 3. Generate the impulse response
time, response = ct.impulse_response(G_k, T=time)

# 4. Plot the impulse response
fig = plt.figure(figsize=(15, 9), num='Impulse Response')
ax = plt.gca()

plt.plot(time, response, linewidth=3)
plt.title(r"Impulse Response of $G_k(s) = \frac{2200000}{2.4s^2 + 1400s + 2300000}$", fontsize=14)
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel(r"Amplitude", fontsize=14)
plt.grid(True)

# Ensure axes start at zero
ax.set_xlim(left=0)
# ax.set_ylim(bottom=0)

# 5. Annotate key points in the impulse response

# Find the peak value and time
peak_index = np.argmax(response)
peak_time = time[peak_index]
peak_value = response[peak_index]

# Mark the peak point
ax.plot(peak_time, peak_value, 'ro', label='Peak')

# Draw vertical and horizontal lines to the peak point
ax.vlines(x=peak_time, ymin=-300, ymax=peak_value, color='r', linestyle='--')
ax.hlines(y=peak_value, xmin=0, xmax=peak_time, color='r', linestyle='--')

# Annotate the peak value and time
plt.text(peak_time, peak_value + peak_value * 0.1, f"Peak Value: {peak_value:.3f}\nPeak Time: {peak_time:.5f}s", ha='center', fontsize=12)

# 6. Calculate and display relevant metrics

# Since impulse responses don't have settling time or overshoot in the same way as step responses, we'll focus on other metrics

# Compute the area under the impulse response (should be equal to the gain for stable systems)
area = np.trapz(response, time)

# Create a DataFrame for the metrics
metrics = {
    'Metric': ['Peak Time', 'Peak Value', 'Area Under Curve'],
    'Value': [peak_time, peak_value, area]
}

df_metrics = pd.DataFrame(metrics)

# Print the table using tabulate
print(tabulate(df_metrics, headers='keys', tablefmt='pretty', floatfmt=".6f"))

# Save the metrics to a CSV file
df_metrics.to_csv('csv/impulse.csv', index=False)

# Show legend
plt.legend()

# Save the figure
plt.figure('Impulse Response')
plt.savefig("figures/Impulse_Response.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()