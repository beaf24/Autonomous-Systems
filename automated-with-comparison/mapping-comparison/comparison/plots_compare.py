import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parent_dir = os.getcwd()
    file = open(parent_dir + "/automated-with-comparison/mapping-comparison/data/data_compare.txt", "r")

    logs = []
    adnns = []
    errors = []

    for line in file.readlines():
        frag = line.split("\t")
        logs.append(float(frag[2]))
        adnns.append(float(frag[3]))
        errors.append(float(frag[7]))

    plt.figure()

    plt.subplot(1,2,1)
    plt.scatter(np.array(logs), np.array(adnns))
    plt.xlabel("Free probability")
    plt.ylabel("ADNN")
    plt.title("ADNN variation")
    plt.subplot(1,2,2)
    plt.scatter(np.array(logs), np.array(errors))
    plt.xlabel("Free probability")
    plt.ylabel("Error")
    plt.title("Error variation")
    plt.show()