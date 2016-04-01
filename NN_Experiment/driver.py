"""
Author: Aadil Ahamed
Date: 3/13/16
driver.py: Driver for experiment
"""

import gen_data
import subprocess

# Global Variables
brute_out = "./output/brute_out.txt"
kd_out = "./output/kd_out.txt"


def create_data(N):
  data_file = "./data/data.txt"
  query_file = "./data/query.txt"
  gen_data.gen_data(data_file, N)
  gen_data.gen_data(query_file, N)


def run_brute():
  subprocess.call(["./nn_aadil"])


def run_kd():
  subprocess.call(["./CUDA_KDtree"])


def write_header(filename):
  with open(filename, 'w') as out_file:
    out_file.write("N Time\n");


def setup():
  write_header(brute_out)
  write_header(kd_out)

def read_output(filename):
  N = []
  time = []
  with open(filename) as out_file:
    header = out_file.readline()
    for line in out_file:
     line = line.split(' ')
     N.append(int(line[0]))
     time.append(float(line[1]))
  return (N, time)

def display_results():
  pass 

def main():
  setup()
  for i in range(10, 21):
    create_data(2**i)
    run_brute()
    run_kd()

if __name__ == "__main__":
  main()

