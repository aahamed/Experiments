"""
Author: Aadil Ahamed
Date: 2/24/16
nn_brute.py: Sequential brute force implementation of nearest neighbor search
"""
import math


def distance(p, q):
    return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)


def min_distance(distances):
  min_dist = distances[0]
  min_ind = 0
  for i in range(len(distances)):
    if distances[i] < min_dist:
      min_dist = distances[i]
      min_ind = i
  return (min_dist, min_ind)


def nn(data, queries):
  res = []
  for qpoint in queries:
    dist = []
    for dpoint in data:
      dist.append(distance(qpoint, dpoint))
    res.append(data[min_distance(dist)[1]])
  return res


def test_nn():
  data = [(1, 1), (2, 2), (3, 3), (4, 4)]
  queries = [(1, 2), (2, 3), (3, 1), (0, 6)]
  res = nn(data, queries)
  print(res)

def main():
  test_nn()

if __name__ == "__main__":
  main()
