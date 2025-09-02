import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('wave_data.csv')

# Create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Amplitude comparison
ax1.plot(data['time'], data['true_A'], label='True Amplitude', linewidth=2)
ax1.plot(data['time'], data['est_A'], label='Estimated Amplitude', linestyle='--')
ax1.set_ylabel('Amplitude')
ax1.legend()
ax1.grid(True)

# Plot 2: Frequency comparison
ax2.plot(data['time'], data['true_f'], label='True Frequency', linewidth=2)
ax2.plot(data['time'], data['est_f'], label='Estimated Frequency', linestyle='--')
ax2.set_ylabel('Frequency (Hz)')
ax2.legend()
ax2.grid(True)

# Plot 3: Waveform comparison
ax3.plot(data['time'], data['true_wave'], label='True Wave', alpha=0.7)
ax3.plot(data['time'], data['measured'], label='Measured', linestyle='', marker='.', markersize=2)
ax3.plot(data['time'], data['est_wave'], label='Estimated Wave', linestyle='--')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Wave Value')
ax3.legend()
ax3.grid(True)

# Plot 4: Phase comparison
ax4.plot(data['time'], data['true_phi'], label='True Phase', linewidth=2)
ax4.plot(data['time'], data['est_phi'], label='Estimated Phase', linestyle='--')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Phase (rad)')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.savefig('wave_comparison.png', dpi=300)
plt.show()