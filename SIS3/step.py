import os
import control as ct
import matplotlib.pyplot as plt

# Define the transfer function: G_k(s) = 2200000 / (2.4s^2 + 1400s + 2300000)
numerator = [2200000]
denominator = [2.4, 1400, 2300000]
G_k = ct.TransferFunction(numerator, denominator)

# Generate the step response
time, response = ct.step_response(G_k)

# Plot the step response
fig = plt.figure(figsize=(10, 6), num='Step Response')

def on_key(event):
    if event.key == 'escape':
        plt.close()
fig.canvas.mpl_connect('key_press_event', on_key)

plt.plot(time, response, linewidth=3)
plt.title(r"Step Response of $G_k(s) = \frac{2200000}{2.4s^2 + 1400s + 2300000}$", fontsize=14)
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel(r"Flow Rate ($m^3/h$)", fontsize=14)
plt.grid(True)

# Get the current axes
ax = plt.gca()
ax.set_xlim(left=0)  # Ensure x-axis starts at 0
ax.set_ylim(bottom=0)  # Ensure y-axis starts at 0

# Get step response information
info = ct.step_info(G_k)

# Steady-state value
ss_value = info['SteadyStateValue']
ax.axhline(y=ss_value, color='r', linestyle='--', label='Steady-State Value')
plt.plot(time[-1], ss_value, 'rx')
plt.text(time[-1] - 0.0025, ss_value - 0.05, f"SS Value: {ss_value:.2f}")

# Annotate SS error
ax.axhline(y=1, color='black', linestyle='--')
ss_error = abs(1 - ss_value) * 100
ax.annotate('', xy=(0.0245, 1), xytext=(0.0245, ss_value), arrowprops=dict(arrowstyle='<->', color='black'))
ax.annotate(f"SS Error:\n {ss_error:.2f}%", xy=(0.0245, 1), xytext=(0.02275, 1.18), arrowprops=dict(arrowstyle='->', color='black'))
# Rise time
rise_time = info['RiseTime']

# Calculate 10% and 90% thresholds
threshold_10 = 0.1 * ss_value
threshold_90 = 0.9 * ss_value

# Find the indices where the response is closest to 10% and 90% of the SS value
index_10 = next(i for i, r in enumerate(response) if r >= threshold_10)
index_90 = next(i for i, r in enumerate(response) if r >= threshold_90)

# Get the corresponding times
time_10 = time[index_10]
time_90 = time[index_90]

ax.vlines(x=time_10, ymin=0, ymax=ss_value*0.15, color='g', linestyle='--', label='Rise Time')
ax.hlines(y=ss_value*0.15, xmin=0, xmax=time_10, color='g', linestyle='--')
plt.plot(time_10, ss_value*0.15, 'go')

ax.vlines(x=time_90, ymin=0, ymax=ss_value*0.9, color='g', linestyle='--', label='Rise Time')
ax.hlines(y=ss_value*0.9, xmin=0, xmax=time_90, color='g', linestyle='--')
plt.plot(time_90, ss_value*0.9, 'go')

# Draw double headed arrow for rise time between 10 and 90 % of the final value
ax.annotate('', xy=(time_90, 0.1), xytext=(time_10, 0.1), arrowprops=dict(arrowstyle='<->', color='black'))
plt.text(time_10, 0.15, f" Rise  Time:\n  {rise_time:.4f}  s")


# Peak time and value
peak_time = info['PeakTime']
peak_value = info['Peak']
overshoot = info['Overshoot']
ax.vlines(x=peak_time, ymin=0, ymax=peak_value, color='m', linestyle='--', label='Peak Time')
ax.hlines(y=peak_value, xmin=0, xmax=peak_time, color='m', linestyle='--')
plt.plot(peak_time, peak_value, 'mo')
plt.text(peak_time + 0.001, peak_value - 0.06, f"Peak Value: {peak_value:.2f}")
plt.text(peak_time + 0.001, peak_value - 0.02, f"Overshoot: {overshoot:.2f}%")

# Draw double headed arrow between input and 1
ax.annotate('', xy=(0.0015, peak_value), xytext=(0.0015, 1), arrowprops=dict(arrowstyle='<->', color='black'))
plt.text(0.001, 1.04, f"Overshoot", rotation=90)


# Settling time
settling_time = info['SettlingTime']
settling_index = next(i for i, t in enumerate(time) if t >= settling_time)
settling_value = response[settling_index]
ax.vlines(x=settling_time, ymin=0, ymax=settling_value, color='b', linestyle='--', label='Settling Time')
ax.hlines(y=settling_value, xmin=0, xmax=settling_time, color='b', linestyle='--')
plt.plot(settling_time, settling_value, 'bo')
# plt.text(settling_time + 0.00025, settling_value + 0.05, f"Settling Time: {settling_time:.2f}s")

# Draw arrow for settling time
ax.annotate('', xy=(settling_time, 0.7), xytext=(0, 0.7), arrowprops=dict(arrowstyle='->', color='black'))
plt.text(settling_time/2, 0.72, f"Settling Time: {settling_time:.2f}s")

plt.legend()
plt.show()
