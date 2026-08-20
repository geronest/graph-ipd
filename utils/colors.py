"""
author: Seongho Son (seong.son.22@ucl.ac.uk)
functions for assigning colors when using matplotlib.
"""


class ColorRevolver:
    def __init__(self):
        self.colors = [
            "tab:red",
            "tab:orange",
            "tab:green",
            "tab:blue",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
        self.idx = 0

    def get_color(self):
        """
        return one of the colors in sequence,
        going back to the beginning when it reaches the end of the list.

        return:
            one of the colors
        """
        color = self.colors[self.idx]
        self.idx = (self.idx + 1) % len(self.colors)

        return color
