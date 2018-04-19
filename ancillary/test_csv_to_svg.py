import matplotlib.pyplot as plt

fig = plt.figure()

plt.subplot(221) # 2 rows, 2 columns, plot 1
plt.plot([1,2,3])

plt.subplot(222) # 2 rows, 2 columns, plot 2
plt.plot([3,1,3])

plt.subplot(223) # 2 rows, 2 columns, plot 3
plt.plot([3,2,1])

plt.subplot(224) # 2 rows, 2 columns, plot 4
plt.plot([1,3,1])

plt.show()

fig.savefig('test.svg')