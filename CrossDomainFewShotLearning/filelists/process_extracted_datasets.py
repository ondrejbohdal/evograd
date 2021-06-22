import sys
import os
from subprocess import call

# run this from filelists directory

# process file
datasets = ['cars', 'cub', 'places', 'miniImagenet', 'plantae']

for dataset in datasets:
    os.chdir(dataset)
    print(os.getcwd())
    call('python3 write_' + dataset + '_filelist.py', shell=True)
    os.chdir('..')