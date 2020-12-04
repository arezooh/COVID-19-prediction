from sys import argv
import sys
import os
import subprocess

def Run(r):
    subprocess.call("python ./" + str(r) + "/sc.py", shell=True)

base_r = 1
    
def main():
    for r in range(base_r, 11):
        print(100*'*')
        print('r =', base_r, 'r" =', r)
        Run(r)



if __name__ == "__main__":
    main()
