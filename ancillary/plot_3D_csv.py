from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import csv

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Grab some test data.
X = []
Y = []
Z = []

with open('HTRU_2_inverse.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        X.append(float(row[2]))
        Y.append(float(row[3]))
        Z.append(float(row[4]))

# Plot a basic wireframe.
# ax.plot_wireframe(X, Y, Z, rstride=20, cstride=30)

# Plot a basic scatter plot.
ax.scatter(X, Y, Z)

plt.show()
fig.savefig('test_3D_scatter.svg')
