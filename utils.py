import sys

from scipy.sparse import dok_matrix


def log(msg, tab=0, bpad=0):
    global silent
    
    if not False:
        msgs = msg.split('\n') if '\n' in msg else [msg]

        for msg in msgs:
            for t in range(tab):
                sys.stdout.write('  ')
            
            sys.stdout.write(msg + '\n')

        if bpad:
            sys.stdout.write('\n')


def calc_averages(dictionary):
    averages = {}

    for key in dictionary:
        values = dictionary[key].values()
        averages[key] = float(sum(values)) / len(values)

    return averages
