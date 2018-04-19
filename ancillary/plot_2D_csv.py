# https://pythonprogramming.net/loading-file-data-matplotlib-tutorial/

# Base implementation, works
# import matplotlib.pyplot as plt
# import csv

# fig = plt.figure()

# X = []
# Y = []

# with open('HTRU_2_inverse.csv','r') as csvfile:
#     plots = csv.reader(csvfile, delimiter=',')
#     for row in plots:
#         X.append(float(row[1]))
#         Y.append(float(row[2]))

# plt.plot(X,Y, label='Loaded from file!')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Interesting Graph\nCheck it out')
# plt.legend()
# plt.show()
# fig.savefig('test_2D.svg')


import matplotlib.pyplot as plt
import csv
import sys

def main():
    """This is intended to take CSV files with headers and turn them into SVG files. These 
       files are intended to be from Tensorboard."""

    if len(sys.argv) != 3:
        print("Please specify input and output files as arguments: \"input.csv output\"")
        exit()

    i_file = sys.argv[1]
    o_file = sys.argv[2]

    if not sys.argv[1].endswith('.csv'):
        i_file = i_file + '.csv'

    if not sys.argv[2].endswith('.svg'):
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
    plt.plot(X,Y, label='Loss')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(i_file[:-4] + ' over steps')
    plt.legend(loc=0) # 0: top right .... 4: bottom right

    plt.show()
    fig.savefig(o_file)

if __name__ == '__main__':
    main()