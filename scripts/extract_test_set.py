PATH = './data/filmtrust_probe_set.dat'
TARGET = './data/filmtrust_test_set.dat'

with open(TARGET, 'w') as out:
    with open(PATH, 'r') as src:
        for ln in [ln.rstrip().split(' ') for ln in src.readlines()]:
            print(ln)
            if ln[2] == '4':
                out.write('{} {} {}\n'.format(*ln[:3]))
                
