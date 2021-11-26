"""
usage: hs_soft [-h] -b {E1,E2,P1,P2} [-p PAUSE]

Perform SOFT for given bunch {E1,E2,P1,P2}.

optional arguments:
  -h, --help            show this help message and exit
  -b {E1,E2,P1,P2}, --bunch {E1,E2,P1,P2}
                        bunch id
  -p PAUSE, --pause PAUSE
                        pause value (default 1.0sec)
"""

# Input arguments flag
import sys
_, *flag = sys.argv

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_soft', description='Perform SOFT for given bunch {E1,E2,P1,P2}.')
parser.add_argument('-b', '--bunch', choices=('E1', 'E2', 'P1', 'P2'), help='bunch id', default='E1')
parser.add_argument('-p', '--pause', type=float, help='pause value (default 1.0sec)', default=1.0)
args = parser.parse_args(args=None if flag else ['--help'])

# Import
from os import system
from time import sleep
from epics import caput

# Kick (soft)
def main():
    """ Perform measurement. """
    caput('VEPP4:ALL:turns_kick-SP', 'SOFT')
    sleep(args.pause)
    caput('VEPP4:ALL:turns_bunch-SP', args.bunch)
    sleep(args.pause)
    caput('VEPP4:ALL:turns_do-SP', 1)

if __name__ == '__main__':
	main()
