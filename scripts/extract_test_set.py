PATH = './data/probe_set.dat'
TARGET = './data/test_set.dat'

with open(TARGET, 'w') as out:
    with open(PATH, 'r') as src:
        for ln in [ln.split('::') for ln in src.readlines()]:
            if ln[2] == '5':
                out.write('{}::{}::{}\n'.format(*ln[:3]))
                
