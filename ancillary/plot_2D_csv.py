# Assistance via:
# https://pythonprogramming.net/loading-file-data-matplotlib-tutorial/

import matplotlib.pyplot as plt
import csv
import sys

def run_conversion(i_file, o_file):
    """This is intended to take CSV files with headers and turn them into SVG files. These 
           files are intended to be from Tensorboard."""

    # Handle extensions
    if not i_file.endswith('.csv'):
        i_file = i_file + '.csv'

    if not o_file.endswith('.svg'):
        o_file = o_file + '.svg'
        
    fig = plt.figure()

    X = []
    Y = []

    with open(i_file,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        next(plots, None) # skip headers
        for row in plots:
            # row[0] is wall time, don't care about it
            X.append(float(row[1]))
            Y.append(float(row[2]))

    plt.subplot(111, adjustable='box', aspect='auto')
    plt.plot(X,Y, label='Loss/Advantage')
    plt.xlabel('Step')
    plt.ylabel('Advantage')
    plt.title(i_file[:-4] + ' over steps')
    plt.legend(loc=0) # 0: top right .... 4: bottom right

    plt.show()
    fig.savefig(o_file)


def main():
    if len(sys.argv) != 3:
        print("Please specify input and output files as arguments: \"input.csv output\"")
        exit()

    i_file = sys.argv[1]
    o_file = sys.argv[2]    


    run_conversion(i_file,o_file)
    

if __name__ == '__main__':
    main()