import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Setting up the figure and axis for animation
fig, ax = plt.subplots()
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1.5, 1.5)

# Create an empty line object for the animation
xdata, ydata = [], []
ln, = plt.plot([], [], 'r-', animated=True)

# Initialize other plot elements for a more complex animation
line2, = ax.plot([], [], 'b-', label='Cosine Wave')
line3, = ax.plot([], [], 'g-', label='Tangent Wave')

# Adding a title and labels
ax.set_title('Animated Trigonometric Functions')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.legend(loc='upper right')

# Adding a grid
ax.grid(True)

# Function to initialize the plot
def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1.5, 1.5)
    return ln, line2, line3

# Function to update the plot for animation
def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))  # Sine wave
    ln.set_data(xdata, ydata)
    
    # Update the second line for cosine wave
    ydata2 = np.cos(xdata)
    line2.set_data(xdata, ydata2)
    
    # Update the third line for tangent wave
    ydata3 = np.tan(xdata)
    line3.set_data(xdata, ydata3)
    
    return ln, line2, line3

# Set up the FuncAnimation
ani = FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 500), init_func=init, blit=True)

# Display the animation
plt.show()

# Saving the animation as a .gif file
ani.save('trig_animation.gif', writer='imagemagick', fps=30)

# Adding interactive elements
from matplotlib.widgets import Slider

# Create sliders for controlling the frequency and amplitude of the waves
axfreq = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_freq = Slider(axfreq, 'Frequency', 0.1, 10.0, valinit=1.0)

axamp = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_amp = Slider(axamp, 'Amplitude', 0.1, 2.0, valinit=1.0)

# Update function for sliders
def update_slider(val):
    frequency = slider_freq.val
    amplitude = slider_amp.val
    
    # Modify the sine wave based on the sliders
    line1.set_ydata(amplitude * np.sin(frequency * xdata))
    line2.set_ydata(amplitude * np.cos(frequency * xdata))
    line3.set_ydata(amplitude * np.tan(frequency * xdata))
    fig.canvas.draw_idle()

# Attach the sliders to the update function
slider_freq.on_changed(update_slider)
slider_amp.on_changed(update_slider)

plt.show()
