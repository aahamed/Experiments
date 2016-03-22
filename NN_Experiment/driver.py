"""
Author: Aadil Ahamed
Date: 3/13/16
driver.py: Driver for experiment
"""

import gen_data
import subprocess

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
    out_file.write("N\tTime\n");


def setup():
  brute_out = "./output/brute_out.txt"
  kd_out = "./output/kd_out.txt"
  write_header(brute_out)
  write_header(kd_out)


def main():
  setup()
  for i in range(10, 21):
    create_data(2**i)
    run_brute()
    run_kd()

if __name__ == "__main__":
  main()

