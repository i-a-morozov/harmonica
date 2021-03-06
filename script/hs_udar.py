# Input arguments flag
import sys
_, *flag = sys.argv

# Parse arguments
import argparse
parser = argparse.ArgumentParser(prog='hs_soft', description='Perform UDAR kick for selected bunch {E1,E2,P1,P2}.')
parser.add_argument('-b', '--bunch', choices=('E1', 'E2', 'P1', 'P2'), help='bunch id', default='E1')
parser.add_argument('-p', '--pause', type=float, help='pause value (default 1.0sec)', default=1.0)
args = parser.parse_args(args=None if flag else ['--help'])

# Check host
import socket
if socket.gethostname() != 'vepp4-pult6.inp.nsk.su':
	exit('hs_udar: error: not on vepp4-pult6.inp.nsk.su host')

# Import
from os import system
from time import sleep
from epics import caput

# Set CHAN
def tuki_on():
	""" TUKI on. """
	system('/home/vepp4/kadrs/converter/dsend/bin/dsend-bin -o CHAN -d "LG TUKI 21" > /dev/null')
def tuki_off():
	""" TUKI off. """
	system('/home/vepp4/kadrs/converter/dsend/bin/dsend-bin -o CHAN -d "LG TUKI 22" > /dev/null')
def tuki_kick():
	""" TUKI kick. """
	system('/home/vepp4/kadrs/converter/dsend/bin/dsend-bin -o CHAN -d "LG TUKI 24" > /dev/null')

# Set BPMs
def prepare():
	""" Prepare BPMs. Set to UDAR mode, set given bunch and set trigger. """
	caput('VEPP4:ALL:turns_kick-SP', 'UDAR', wait=True)
	sleep(0.1)
	caput('VEPP4:ALL:turns_bunch-SP', args.bunch, wait=True)
	sleep(0.1)
	caput('VEPP4:ALL:turns_do-SP', 1, wait=True)
	sleep(0.1)

# Kick
def main():
	""" Perform UDAR kick. """
	tuki_on()
	sleep(args.pause)
	prepare()
	tuki_kick()
	sleep(args.pause)
	tuki_off()

if __name__ == '__main__':
	main()