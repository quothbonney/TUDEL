import numpy as np
import matplotlib.pyplot as plt;
import csv

FILENAME = "ETHAN.csv";

fields, rows = [], [];

with open(FILENAME, 'r') as csvfile:
    csvreader = csv.reader(csvfile);

    for row in csvreader:
        rows.append(row[1:]);



# Defining rows from matrix
saturation = np.array(rows[1]).astype(float)
value = np.array(rows[2]).astype(float)
x = np.array(rows[3:]).astype(float)  # Don't count first ros

added = saturation
fig = plt.figure()
ax = fig.add_subplot(111)


ax.scatter(x[430], added, c='b', label='730')

# Getting line of best fit
theta = np.polyfit(x[430], added, 1);
y_line = theta[1] + theta[0] * x[430]
ax.plot(x[430], y_line)
# of the line using the numpy.polyfit() functiond, c='b');
print(theta);

ax.scatter(x[430], value, c='g', label='550')

y = x[0] ** 2;
# ax.plot(x[0], added, c='y', label='750')
# ax.plot(x[600], added, c='r', label='950')
ax.set(xlabel='Absorbance', ylabel='Value + Saturation')

# plt.legend(loc='upper right');

plt.show()

