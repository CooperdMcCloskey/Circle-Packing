import matplotlib.pyplot as plt

from matplotlib.markers import MarkerStyle
from matplotlib import patches 
from matplotlib.transforms import Affine2D

# Create a figure and axis
def display(positions, radii):

  fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})

  for i in range(len(positions)):
    ellipse = patches.Circle(
      xy=positions[i],
      radius=radii[i],
    )
    ax.add_patch(ellipse)

  plt.show()