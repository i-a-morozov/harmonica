# Input arguments flag
import sys
_, *flag = sys.argv

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_soft', description='Perform SOFT kick for selected bunch {E1,E2,P1,P2}.')
parser.add_argument('-b', '--bunch', choices=('E1', 'E2', 'P1', 'P2'), help='bunch id', default='E1')
parser.add_argument('-p', '--pause', type=float, help='pause value (default 1.0sec)', default=1.0)
args = parser.parse_args(args=None if flag else ['--help'])

# Import
from os import system
from time import sleep
from epics import caput

# Kick (soft)
def main():
  """ Perform SOFT kick. """
  caput('VEPP4:ALL:turns_kick-SP', 'SOFT', wait=True)
  sleep(args.pause)
  caput('VEPP4:ALL:turns_bunch-SP', args.bunch, wait=True)
  sleep(args.pause)
  caput('VEPP4:ALL:turns_do-SP', 1, wait=True)

if __name__ == '__main__':
	main()