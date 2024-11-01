import control as ct
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Define the transfer function: G_k(s) = 2200000 / (2.4s^2 + 1400s + 2300000)
numerator = [2200000]
denominator = [2.4, 1400, 2300000]

# Ensure coefficients are floats to prevent potential errors in calculations
numerator = [float(coef) for coef in numerator]
denominator = [float(coef) for coef in denominator]

G_k = ct.TransferFunction(numerator, denominator)

# 2. Compute the poles and zeros
try:
    zeros = ct.zeros(G_k)
    poles = ct.poles(G_k)
except Exception as e:
    print(f"Error computing poles and zeros: {e}")
    zeros = np.array([])
    poles = np.array([])

# Ensure poles and zeros are numpy arrays for consistency
zeros = np.array(zeros)
poles = np.array(poles)

# 3. Create a figure for the pole-zero map
fig, ax = plt.subplots(figsize=(10, 8))

# 4. Plot the poles and zeros
# Plot zeros as 'o' and poles as 'x'
if zeros.size > 0:
    ax.plot(np.real(zeros), np.imag(zeros), 'go', markersize=10, label='Zeros')
else:
    print("No zeros to plot.")

if poles.size > 0:
    ax.plot(np.real(poles), np.imag(poles), 'rx', markersize=10, label='Poles')
else:
    print("No poles to plot.")

# 5. Annotate the plot
ax.set_title('Pole-Zero Map of the Transfer Function', fontsize=14)
ax.set_xlabel('Real Axis', fontsize=12)
ax.set_ylabel('Imaginary Axis', fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.legend()

# 6. Set axis limits for better visibility
# Determine limits based on poles and zeros
if poles.size > 0 or zeros.size > 0:
    all_real = np.concatenate((np.real(poles), np.real(zeros))) if zeros.size > 0 else np.real(poles)
    all_imag = np.concatenate((np.imag(poles), np.imag(zeros))) if zeros.size > 0 else np.imag(poles)
    real_min = np.min(all_real) - abs(np.min(all_real)) * 0.2 - 1
    real_max = np.max(all_real) + abs(np.max(all_real)) * 0.2 + 1
    imag_min = np.min(all_imag) - abs(np.min(all_imag)) * 0.2 - 1
    imag_max = np.max(all_imag) + abs(np.max(all_imag)) * 0.2 + 1
else:
    # Default limits if no poles or zeros
    real_min, real_max = -10, 10
    imag_min, imag_max = -10, 10

# Adjust limits to include origin if necessary
real_min = min(real_min, -1)
real_max = max(real_max, 1)
imag_min = min(imag_min, -1)
imag_max = max(imag_max, 1)

ax.set_xlim([real_min, real_max])
ax.set_ylim([imag_min, imag_max])

# 7. Annotate pole and zero values
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

# 8. Add stability regions for continuous-time systems
# Shade the right-half plane (RHP) to indicate instability region
ax.fill_between([0, real_max], imag_min, imag_max, color='red', alpha=0.1, label='Unstable Region')

# Shade the left-half plane (LHP) to indicate stability region
ax.fill_between([real_min, 0], imag_min, imag_max, color='green', alpha=0.1, label='Stable Region')

# Bring legend to the front
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

# 9. Handle potential errors
# Ensure that poles and zeros are not empty arrays
if zeros.size == 0:
    print("No zeros found for the system.")
    
if poles.size == 0:
    print("No poles found for the system.")

# 10. Display poles and zeros in a table
data = []
for i, p in enumerate(poles):
    data.append(['Pole', f'{p:.6f}', f'{np.real(p):.6f}', f'{np.imag(p):.6f}'])
for i, z in enumerate(zeros):
    data.append(['Zero', f'{z:.6f}', f'{np.real(z):.6f}', f'{np.imag(z):.6f}'])

table_headers = ['Type', 'Location', 'Real Part', 'Imaginary Part']
print("\nPoles and Zeros of the Transfer Function:")
print(tabulate(data, headers=table_headers, tablefmt='pretty'))

# 11. Save the figure
plt.savefig("figures/Pole_Zero_Map.png", dpi=300, bbox_inches='tight')

# 12. Show the plot
plt.show()
