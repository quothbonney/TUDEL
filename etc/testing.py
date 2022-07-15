import numpy as np
import matplotlib.pyplot as plt;
import csv

FILENAME = "ETHAN.csv";

fields, rows = [], [];

with open(FILENAME, 'r') as csvfile:
    csvreader = csv.reader(csvfile);

    for row in csvreader:
        rows.append(row[1:]);


def save(w):
    # Defining rows from matrix
    saturation = np.array(rows[1]).astype(float)
    value = np.array(rows[2]).astype(float)
    r = np.array(rows[3:]).astype(float)  # Don't count header rows 

    p_sat = np.poly1d(np.polyfit(r[w], saturation, 3))
    p_val = np.poly1d(np.polyfit(r[w], value, 3))


    t = np.linspace(0, r[w].max(), 200)



    plt.xlabel("Absorbance")
    plt.ylabel("Saturation")
    # plt.plot(r[w], saturation, 'o', t_sat, p_sat(t_sat), '-')

    plt.scatter(r[w], saturation, c='C1', label="Saturation")
    plt.plot(t, p_sat(t), 'C1')

    plt.scatter(r[w], value, c='C2', label="Value")
    plt.plot(t, p_val(t), 'C2')

    plt.legend(loc='lower right', title=f"{w}nm");

    plt.savefig(f'figures/satvval/{w}.jpg')
    print(f"Saved figure {w}")
    plt.clf()

if __name__ == '__main__':
    for i in range(0, 500, 2):
        save(i)
