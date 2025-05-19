import sys
import math

import matplotlib as plt
import numpy as np

import plot

sys.setrecursionlimit(10000)

learning_rate = 1e-5
delta = 1e-3

#constants
n = 3
expansion_precision = 8 #number of digits


circle_positions = np.random.random((n,2))
circle_radii = np.zeros(n)

def check_overlap(radii, positions):
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
  elif(np.all(check_overlap(radii + delta, positions))):
    return expand_circles(radii, positions, delta=delta/10)
  else:
    return expand_circles(radii + delta * np.where(check_overlap(radii + delta, positions), 0, 1), positions)


def get_area(radii):
  return np.sum(math.pi * radii ** 2)

def get_gradients(delta_range):
  radius_deltas = np.random.uniform(-delta_range, delta_range, (n))
  position_deltas = np.random.uniform(-delta_range, delta_range, (n,2))

  test_radii = np.copy(circle_radii)
  test_positions = np.copy(circle_positions)

  radius_gradient = np.zeros(n)
  position_gradient = np.zeros((n,2))

  point_area = get_area(circle_radii)

  for i in range(n):
    test_radii[i] += radius_deltas[i]
    radius_gradient[i] = (get_area(test_radii) - point_area) / radius_deltas[i]
    test_radii[i] -= radius_deltas[i]

    test_positions[i][0] += position_deltas[i][0]
    position_gradient[i][0] = (get_area(expand_circles(test_radii, test_positions)) - point_area) / position_deltas[i][0]
    test_positions[i][0] -= position_deltas[i][0]

    test_positions[i][1] += position_deltas[i][1]
    position_gradient[i][1] = (get_area(expand_circles(test_radii, test_positions)) - point_area) / position_deltas[i][1]
    test_positions[i][1] -= position_deltas[i][1]
  # print(position_gradient)
  # print(radius_gradient)
  if(np.any(check_overlap(circle_radii + radius_gradient * learning_rate, circle_positions + position_gradient * learning_rate))): return get_gradients(delta_range)
  return radius_gradient, position_gradient

circle_radii = expand_circles(circle_radii)

for i in range(10000):
  radius_gradient, position_gradient = get_gradients(delta) 
  circle_radii += radius_gradient * learning_rate
  circle_positions +=  position_gradient * learning_rate
  circle_radii = expand_circles(circle_radii)
  print(get_area(circle_radii))






      
      






