import matplotlib.pyplot as plt

from datetime import timedelta
from matplotlib.patches import Rectangle


class ImageInterface:
    def __init__(self, image):
        self.image = image
        self.bounding_box = None
        (self.width, self.height, _) = self.image.shape

    class TwoClickSelector:
        def __init__(self, ax, callback):
            self.ax = ax
            self.callback = callback
            self.clicks = []
            self.cid = ax.figure.canvas.mpl_connect('button_press_event', self)

        def __call__(self, event):
            if event.inaxes != self.ax:
                return
            self.clicks.append((int(event.xdata), int(event.ydata)))
            if len(self.clicks) == 1:
                print("First corner selected. Click to select the second corner.")
            if len(self.clicks) == 2:
                self.callback(self.clicks)
                self.ax.figure.canvas.mpl_disconnect(self.cid)
                plt.close()

    def select_rectangle(self, clicks):
        self.bounding_box = clicks

    def get_bounding_box(self, title="Select bounding box", required=False, defaults=None, color="green"):
        while True:
            # Create the main figure and axis for selection
            fig, ax = plt.subplots()
            ax.imshow(self.image)
            ax.set_title("Click to select two corners of the rectangle")

            # Create the TwoClickSelector
            selector = self.TwoClickSelector(ax, self.select_rectangle)
            plt.title(title)
            plt.show()

            if self.bounding_box is None:
                if required:
                    continue
                elif defaults is None:
                    confirmation = input("You have not chosen a bounding box. Is this correct? Y/n").lower().strip() or "y"
                else:
                    self.bounding_box = defaults
                    confirmation = input("You have not chosen a bounding box. Use default? Y/n").lower().strip() or "y"
                    
            else:
                # Create a new figure to display the result
                fig, ax = plt.subplots()
                ax.imshow(self.image)

                # Draw the rectangle
                (x1, y1), (x2, y2) = self.bounding_box
                width = x2 - x1
                height = y2 - y1
                rect = Rectangle((x1, y1), width, height, fill=False, edgecolor=color, linewidth=2)
                ax.add_patch(rect)

                plt.title('Image with Selected Rectangle')
                plt.show(block=False)

                # Ask for confirmation
                confirmation = input("Does this bounding box look correct? Y/n").lower().strip() or "y"
                plt.close()

            if confirmation == 'y':
                break

        bounding_box = self.bounding_box
        self.bounding_box = None
        return bounding_box

    def display_bounding_box(self):
        fig, ax = plt.subplots()
        ax.imshow(self.image)

        (x1, y1), (x2, y2) = self.bounding_box
        width = x2 - x1
        height = y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        plt.title('Final Image with Selected Rectangle')
        plt.show()
