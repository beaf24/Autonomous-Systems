import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parent_dir = os.getcwd()
    file = open(parent_dir + "/automated-with-comparison/mapping-comparison/data/data_compare.txt", "r")
    file.readline()
    logs = []
    adnns = []
    errors = []
    error_unk = []
    error_free = []
    error_occ = []

    for line in file.readlines():
        print(line)
        frag = line.split("\t")
        logs.append(float(frag[2]))
        adnns.append(float(frag[3]))
        errors.append(float(frag[7]))
        error_unk.append(float(frag[8]))
        error_free.append(float(frag[9]))
        error_occ.append(float(frag[10]))

    print(adnns, logs)

    plt.figure(figsize=(10,2))

    plt.subplot(1,2,1)
    plt.plot(np.array(logs), np.array(adnns), 'k')
    plt.xlabel("P_free")
    plt.ylabel("ADNN")
    plt.title("ADNN variation")

    plt.subplot(1,2,2)
    plt.plot(np.array(logs), np.array(errors), 'k')
    plt.xlabel("P_free")
    plt.ylabel("Error")
    plt.title("Error variation")
    plt.show()

    plt.figure(figsize=(10,2))

    plt.subplot(1,3,1)
    plt.plot(np.array(logs), np.array(error_unk), 'k')
    plt.xlabel("P_free")
    plt.ylabel("Error")
    plt.title("Error unknown")

    plt.subplot(1,3,2)
    plt.plot(np.array(logs), np.array(error_free), 'k')
    plt.xlabel("P_free")
    plt.ylabel("Error")
    plt.title("Error free")

    plt.subplot(1,3,3)
    plt.plot(np.array(logs), np.array(error_occ), 'k')
    plt.xlabel("P_free")
    plt.ylabel("Error")
    plt.title("Error occupied")
    plt.show()