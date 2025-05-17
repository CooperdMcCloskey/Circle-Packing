import sys
import math

import matplotlib as plt
import numpy as np

import plot

sys.setrecursionlimit(10000)

#constants
n = 3
expansion_precision = 8 #number of digits


circle_positions = np.random.random((n,2))
circle_radii = np.zeros(n)

def check_overlap(radii, positions=circle_positions):
  circle_position_differences = np.tile(positions[np.newaxis, :, :], (n,1,1)) - np.tile(positions[:, np.newaxis, :], (1,n,1))
  circle_position_distances = np.linalg.norm(circle_position_differences, axis=2)
  circle_distances = circle_position_distances - np.tile(radii[np.newaxis, :], (n,1)) - np.tile(radii[:, np.newaxis], (1,n))
  np.fill_diagonal(circle_distances, np.inf)

  wall_position_differences = np.hstack((positions, 1-positions))
  wall_distances = wall_position_differences - np.tile(radii[:, np.newaxis], (1, 4))

  distances = np.hstack((circle_distances, wall_distances))
  min_distances = np.min(distances, axis=1)

  return min_distances < 0

def expand_circles(radii, positions = circle_positions, delta=0.1):
  if delta < math.pow(10, -expansion_precision): 
    return radii
  elif(np.all(check_overlap(radii + delta))):
    return expand_circles(radii, delta=delta/10)
  else:
    return expand_circles(radii + delta * np.where(check_overlap(radii + delta), 0, 1))


def get_area(radii):
  return np.sum(math.pi * radii ** 2)

def get_gradients(delta):
  point_area = get_area(circle_radii)
  partial_derivatives = np.zeros((n,3))
  for i in range(n):
    circle_positions[i][0] += delta
    partial_derivatives[i][0] = get_partial_derivative(circle_radii, circle_positions, point_area, delta)
    circle_positions[i][0] -= delta

    circle_positions[i][1] += delta
    partial_derivatives[i][1] = get_partial_derivative(circle_radii, circle_positions, point_area, delta)
    circle_positions[i][1] -= delta

    circle_radii[i] += delta
    partial_derivatives[i][1] = get_partial_derivative(circle_radii, circle_positions, point_area, delta)
    circle_radii[i] -= delta
  
  return partial_derivatives


def get_partial_derivative(radii, positions, point_area, delta):
  area = get_area(expand_circles(radii, positions=positions))
  delta_area = area-point_area
  return delta_area/delta


circle_radii = expand_circles(circle_radii)
print(get_gradients(0.1))
plot.display(circle_positions, circle_radii)



      
      






