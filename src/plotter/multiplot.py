"""Abstracted functionality to plot figures with a suptitle"""
import matplotlib.pyplot as plt


class MultiPlot:
    """
    MultiPlot receives images and creates a Matplotlib figure for them
    """
    def __init__(self):
        """Initialize variables images, names, cmaps and figure"""
        self.images = []
        self.names = []
        self.cmaps = []
        self.figure = None

    def add_figure(self, image, name, cmap=None):
        """Adds a figure with a name and cmap"""
        self.images.append(image)
        self.names.append(name)
        self.cmaps.append(cmap)

    def draw(self, suptitle: str) -> plt.Figure:
        """Draws the figures in the order they were added and adds a suptitle"""
        total = len(self.images)
        fig = plt.figure()
        for index in range(total):
            fig.add_subplot(1, total, index + 1)
            cmap = self.cmaps[index]
            plt.imshow(self.images[index], cmap)
            plt.title(self.names[index])

        plt.suptitle(suptitle)
        self.figure = fig

        return fig
