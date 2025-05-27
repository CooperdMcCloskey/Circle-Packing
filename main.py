import sys
import math
import time

import matplotlib as plt
import numpy as np

import plot

sys.setrecursionlimit(100000)
startTime = time.time()

initial_learning_rate = 1e-5
learning_rate = 1e-5
active_learning_rate = 0;
delta = 1e-3

#constants
c= 300000
n = 10
expansion_precision = 9 #number of digits


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

def expand_circles(radii, positions = circle_positions, expansion_delta=0.1):
  while expansion_delta > math.pow(10, -expansion_precision):
    overlaps = check_overlap(radii + expansion_delta, positions)
    if(np.all(overlaps)):
      expansion_delta/=10
    else:
      radii += expansion_delta * np.where(overlaps, 0, 1)
  return radii


def get_sum(radii):
  return np.sum(radii)

def get_gradients(delta_range, counter=0):
  global learning_rate, active_learning_rate, circle_radii
  while True:
    active_learning_rate = learning_rate * math.pow(0.999, max(0, counter-20))
    circle_radii -= 0 if counter < 100 else 1e-5

    radius_deltas = np.random.uniform(-delta_range, delta_range*0.1, (n))
    position_deltas = np.random.uniform(-delta_range, delta_range, (n,2))

    test_radii = np.copy(circle_radii)
    test_positions = np.copy(circle_positions)

    radius_gradient = np.zeros(n)
    position_gradient = np.zeros((n,2))

    point_value = get_sum(expand_circles(circle_radii, circle_positions))

    for i in range(n):
      test_radii[i] += radius_deltas[i]
      radius_gradient[i] = (get_sum(test_radii) - point_value) / radius_deltas[i]
      test_radii[i] -= radius_deltas[i]

      test_positions[i][0] += position_deltas[i][0]/3
      position_gradient[i][0] += (get_sum(expand_circles(test_radii, test_positions)) - point_value) / (3*position_deltas[i][0])
      test_positions[i][0] -= position_deltas[i][0]

      test_positions[i][1] += position_deltas[i][1]
      position_gradient[i][1] = (get_sum(expand_circles(test_radii, test_positions)) - point_value) / position_deltas[i][1]
      test_positions[i][1] -= position_deltas[i][1]
    
    clip_factor = 100*initial_learning_rate/learning_rate
    radius_gradient = np.clip(radius_gradient, -clip_factor, clip_factor)
    position_gradient = np.clip(position_gradient, -clip_factor, clip_factor)

    if(np.any(check_overlap(circle_radii + radius_gradient * active_learning_rate, circle_positions + position_gradient * active_learning_rate))):
      delta_range*0.99
      counter+1
    elif(get_sum(expand_circles(circle_radii + radius_gradient * active_learning_rate, circle_positions + position_gradient * active_learning_rate)) - point_value < 0):
      delta_range*0.99
    else:
      return radius_gradient, position_gradient, counter

total_count = 0

while get_sum(circle_radii) < 1.0d:
  circle_positions = np.random.random((n,2))
  circle_radii = expand_circles(np.zeros(n), circle_positions)

last_sum = 0
for i in range(c):
  grad_time = time.time()
  radius_gradient, position_gradient, counter = get_gradients(delta) 
  if(time.time()-grad_time > 10): print(f"grad time: {time.time()-grad_time}")
  

  sum_before = get_sum(circle_radii)
  circle_radii_before = circle_radii.copy()
  circle_positions_before = circle_positions.copy()

  circle_radii += radius_gradient * active_learning_rate
  circle_positions +=  position_gradient * active_learning_rate
  circle_radii = expand_circles(circle_radii, circle_positions)

  if(i % 20==0):
    sum = get_sum(circle_radii)
    print(f'i: {i}')
    print(f'sum: {sum}')
    print(f'delta sum: {sum-last_sum}')
    last_sum=sum
    print(f'{time.time() - startTime} seconds elapsed')


  total_count += counter
  # if(counter > 10): learning_rate = learning_rate*0.95 + active_learning_rate * 0.05
  if i%500 == 0: plot.display(circle_positions, circle_radii)

print(f'average get gradient iterations: {total_count/c}')
print(f'{time.time() - startTime} seconds elapsed')
print(get_sum(circle_radii))
# plot.display(circle_positions, circle_radii)
