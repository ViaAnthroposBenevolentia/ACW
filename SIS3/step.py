import control as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

# Define the transfer function: G_k(s) = 2200000 / (2.4s^2 + 1400s + 2300000)
numerator = [2200000]
denominator = [2.4, 1400, 2300000]
G_k = ct.TransferFunction(numerator, denominator)

# Generate a custom time vector with more steps (e.g., 5000 points over 0.025 seconds)
time = np.linspace(0, 0.025, 5000)  # 5000 points between 0 and 0.025 seconds

# Generate the step response
time, response = ct.step_response(G_k, T=time)

# Plot the step response
fig = plt.figure(figsize=(15, 9), num='Step Response')
ax = plt.gca()

plt.plot(time, response, linewidth=3)
plt.title(r"Step Response of $G_k(s) = \frac{2200000}{2.4s^2 + 1400s + 2300000}$", fontsize=14)
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel(r"Flow Rate ($m^3/h$)", fontsize=14)
plt.grid(True)

ax.set_xlim(left=0)  # Ensure x-axis starts at 0
ax.set_ylim(bottom=0)  # Ensure y-axis starts at 0

# Get step response information
info = ct.step_info(G_k)

# Steady-state value
ss_value = info['SteadyStateValue']
ax.axhline(y=ss_value, color='r', linestyle='--', label='Steady-State Value')

# Annotate SS error
x_pos = time[-1] * 0.98
ax.axhline(y=1, color='black', linestyle='--')
ss_error = abs(1 - ss_value) * 100
ax.annotate(
    '', xy=(x_pos, 1), xytext=(x_pos, ss_value),
    arrowprops=dict(arrowstyle='<->', color='black')
)
ax.annotate(
    f"SS Error:\n {ss_error:.2f}%", xy=(x_pos, 1),
    xytext=(x_pos - time[-1]*0.2, 1 + 0.05),
    arrowprops=dict(arrowstyle='->', color='black')
)

# Rise time
rise_time = info['RiseTime']

# Calculate 10% and 90% thresholds
threshold_10 = 0.1 * ss_value
threshold_90 = 0.9 * ss_value

# Find the indices where the response reaches 10% and 90% of the SS value
index_10 = next(i for i, r in enumerate(response) if r >= threshold_10)
index_90 = next(i for i, r in enumerate(response) if r >= threshold_90)

# Get the corresponding times
time_10 = time[index_10]
time_90 = time[index_90]

# Plot rise time markers and lines
ax.vlines(x=time_10, ymin=0, ymax=ss_value*0.1, color='g', linestyle='--', label='Rise Time')
ax.hlines(y=ss_value*0.1, xmin=0, xmax=time_10, color='g', linestyle='--')
plt.plot(time_10, ss_value*0.1, 'go')

ax.vlines(x=time_90, ymin=0, ymax=ss_value*0.9, color='g', linestyle='--')
ax.hlines(y=ss_value*0.9, xmin=0, xmax=time_90, color='g', linestyle='--')
plt.plot(time_90, ss_value*0.9, 'go')

# Annotate Rise Time
y_arrow = ss_value * 0.1
ax.annotate(
    '', xy=(time_90, y_arrow), xytext=(time_10, y_arrow),
    arrowprops=dict(arrowstyle='<->', color='black')
)
plt.text(
    (time_10 + time_90) / 2, y_arrow + 0.02,
    f"Rise Time:\n{rise_time:.4f}s", ha='center'
)

# Peak time and value
peak_time = info['PeakTime']
peak_value = info['Peak']
overshoot = info['Overshoot']

ax.vlines(x=peak_time, ymin=0, ymax=peak_value, color='m', linestyle='--', label='Peak Time')
ax.hlines(y=peak_value, xmin=0, xmax=peak_time, color='m', linestyle='--')
plt.plot(peak_time, peak_value, 'mo')

# Annotate Overshoot
if peak_value > 1:
    x_os = time[len(time)//20]
    ax.annotate(
        '', xy=(x_os, peak_value), xytext=(x_os, 1),
        arrowprops=dict(arrowstyle='<->', color='black')
    )
    plt.text(
        x_os - time[-1]*0.02, (peak_value + 1) / 2,
        "Overshoot", rotation=90, va='center'
    )
    plt.text(
        peak_time + time[-1]*0.02, peak_value - 0.02,
        f"Overshoot: {overshoot:.2f}%"
    )

# Settling time
settling_time = info['SettlingTime']
settling_index = next(i for i, t in enumerate(time) if t >= settling_time)
settling_value = response[settling_index]

ax.vlines(x=settling_time, ymin=0, ymax=settling_value, color='b', linestyle='--', label='Settling Time')
plt.plot(settling_time, settling_value, 'bo')

# Annotate Settling Time
y_st = ss_value * 0.7
ax.annotate(
    '', xy=(settling_time, y_st), xytext=(0, y_st),
    arrowprops=dict(arrowstyle='<->', color='black')
)
plt.text(
    settling_time / 3, y_st + 0.02,
    f"Settling Time: {settling_time:.3f}s"
)

# Add ticks with custom labels
ax.set_yticks([ss_value*0.1, ss_value*0.9, ss_value, 1, peak_value])
ax.set_xticks([time_10, time_90, peak_time, settling_time, time[-1]])
ax.set_yticklabels([
    f"10% Rise: {ss_value*0.1:.3f}",
    f"90% Rise: {ss_value*0.9:.3f}",
    f"SS value: {ss_value:.3f}",
    "Input: 1",
    f"Peak Value: {peak_value:.3f}"
])
ax.set_xticklabels([
    f"{time_10:.4f}s",
    f"{time_90:.4f}s",
    f"{peak_time:.4f}s",
    f"{settling_time:.3f}s",
    f"{time[-1]:.3f}s"
])

plt.legend()

# Save the figure
plt.savefig("figures/Step_Response.png", dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

# Table and csv file
# Create a DataFrame to hold the metrics
metrics = {
    'Metric': ['Rise Time', 'Settling Time', 'Settling Min', 'Settling Max',
               'Overshoot', 'Undershoot', 'Peak', 'Peak Time', 'Steady-State Value', 'Steady-State Error'],
    'Value': [info['RiseTime'], info['SettlingTime'], info['SettlingMin'], info['SettlingMax'],
              info['Overshoot'], info['Undershoot'], info['Peak'], info['PeakTime'],
              info['SteadyStateValue'], abs(1 - info['SteadyStateValue'])]
}

df_metrics = pd.DataFrame(metrics)

# Print the table using tabulate for better formatting
print(tabulate(df_metrics, headers='keys', tablefmt='pretty', floatfmt=".6f"))

# Optionally, save the metrics to a CSV file
df_metrics.to_csv('csv/step.csv', index=False)
