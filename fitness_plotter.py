import numpy as np
import matplotlib.pyplot as plt

# Defining the bezier_curve method
def bezier_curve(y0, y1, y2, y3, t):
    y = (1 - t) * ((1 - t) * ((1 - t) * y0 + t * y1) + t * ((1 - t) * y1 + t * y2)) + t * ((1 - t) * ((1 - t) * y1 + t * y2) + t * ((1 - t) * y2 + t * y3))
    return y

# Bezier curve control points for each category
beziercurve_dict = {
    "sugarcontent": [0, 2.165, 0.03, 0],
    "fatcontent": [0, 1.953, 0.468, 0],
    "watercontent": [0, 0.103, 2.1, 0],
    "flour": [0, 2.165, 0.03, 0]
}

# Exponential decay constants for each category
exponential_decay_dict = {
    "price": [2],
    "saltcontent": [5],
    "vanilla extract": [25]
}

# Defining the exponential decay function
def exponential_decay(c, t):
    return np.exp(-c * t)

# Creating an array of t values from 0 to 1
t_values = np.linspace(0, 1, 100)

# Updated plot_curves function to handle both Bezier and Exponential Decay curves
def plot_curves(curve_dict: dict, fig_name: str, curve_type: str = "bezier"):
    plt.figure(figsize=(10, 6))
    for label, points in curve_dict.items():
        if curve_type == "bezier":
            y_values = [bezier_curve(*points, t) for t in t_values]
        elif curve_type == "exponential":
            y_values = [exponential_decay(points[0], t) for t in t_values]
        else:
            raise ValueError("Unsupported curve type. Use 'bezier' or 'exponential'.")
        plt.plot(t_values, y_values, label=label)

    # Adding labels, legend, and title
    plt.xlim(0, 1)
    plt.xlabel('t (Normalized Time)')
    plt.ylabel(f'{fig_name} Curve Output')
    plt.title(f'{fig_name} Curves for Different Content Attributes')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting Bezier and Exponential Decay curves
plot_curves(beziercurve_dict, "Bezier", curve_type="bezier")
plot_curves(exponential_decay_dict, "Exponential Decay", curve_type="exponential")
