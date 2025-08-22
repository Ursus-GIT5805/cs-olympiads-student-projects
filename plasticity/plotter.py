"""
Utiltity classes to help plotting lines interactively.
"""

import matplotlib.pyplot as plt

# Plot helper class
class Plot:
    def __init__(
        self,
        title="",
        ylabel="",
        xlabel="",
    ):
        self.fig, self.ax = plt.subplots()
        self.lines = {}

        if 0 < len(ylabel):
            self.ax.set_ylabel(ylabel)
        if 0 < len(xlabel):
            self.ax.set_xlabel(xlabel)
        if 0 < len(title):
            self.ax.set_title(title)

        self.ax.grid()

    def create_line(self, name):
        line_plt = self.ax.plot([], label=name)

        self.lines[name] = {
            "line": line_plt[0],
            "x": [],
            "y": [],
        }
        self.ax.legend()

    # Append a new data point to the line with NAME
    def append(self, name, y, x=None):
        if name not in self.lines:
            self.create_line(name)
        line = self.lines[name]

        if x is None:
            x = 0.0
            if 0 < len(line["x"]):
                x = line["x"][-1] + 1.0

        line["x"].append(x)
        line["y"].append(y)

    def draw(self):
        for lineobj in self.lines.values():
            line, x, y = lineobj["line"], lineobj["x"], lineobj["y"]
            line.set_xdata(x)
            line.set_ydata(y)

        self.ax.relim()
        self.ax.autoscale_view()


# Plothandler helper class
class Plothandler:
    def __init__(self):
        self.plots = {}

    def __getitem__(self, key):
        return self.plots[key]

    def __setitem__(self, key, value):
        self.plots[key] = value

    def draw(self, wait=0.001):
        for plot in self.plots.values():
            plot.draw()
        plt.draw()
        plt.pause(wait)
