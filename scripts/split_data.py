import sys
import random
import fileinput


SAMPLE_SIZE = 14000
PATH = './data/movielens_ratings.dat'  

count = 0
samples = []

with fileinput.FileInput(PATH, inplace=True) as file:
    while count < SAMPLE_SIZE:
        for line in file:
            if random.random() < 0.0143 and random.random() < 0.98 and count < SAMPLE_SIZE:
                samples.append(line)
                count += 1
            else:
                sys.stdout.write(line)


with open('./mock_result.dat', 'w') as out:
    for sample in samples:
        out.write(sample)
