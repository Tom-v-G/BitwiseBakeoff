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

exponential_decay_dict = {
        "price": [1, 0.982, 0.538, 0],
        "saltcontent": [0, 2.204, 0, 0],
}

# Creating an array of t values from 0 to 1
t_values = np.linspace(0, 1, 100)

def plot_curves(curve_dict: dict, fig_name: str):
    plt.figure(figsize=(10, 6))
    for label, points in curve_dict.items():
        y_values = [bezier_curve(*points, t) for t in t_values]
        plt.plot(t_values, y_values, label=label)

    # Extending the x-axis range and adding labels and legend
    plt.xlim(0, 1)
    plt.xlabel('t (Normalized Time)')
    plt.ylabel(f'{fig_name} Curve Output')
    plt.title(f'{fig_name} Curves for Different Content Attributes')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_curves(beziercurve_dict, "Bezier")
plot_curves(exponential_decay_dict, "Exponential Decay")