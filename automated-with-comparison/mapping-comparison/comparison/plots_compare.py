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
        error_unk.append(float(frag[-3]))
        error_free.append(float(frag[-2]))
        error_occ.append(float(frag[-1]))

    logs = np.array(logs)
    adnns = np.array(adnns)
    errors = np.array(errors)
    error_unk = np.array(error_unk)
    error_free = np.array(error_free)
    error_occ = np.array(error_occ)

    print(adnns, logs)

    # plt.figure()
    # plt.plot(np.array(logs), np.array(adnns), 'k')
    # plt.xlabel("P_free")
    # plt.ylabel("ADNN (m)")
    # plt.title("ADNN in function of P_free", fontsize = "12")
    # # plt.savefig("adnn", dpi = 1024)
    # plt.show()

    # plt.figure()
    # plt.plot(np.array(logs), np.array(errors), 'k')
    # plt.xlabel("P_free")
    # plt.ylabel("Error (%)")
    # plt.title("Global error percentage in function of P_free", fontsize = "12")
    # # plt.savefig("global error", dpi = 1024)
    # plt.show()

    # plt.figure()
    # plt.plot(np.array(logs), np.array(error_unk), 'k')
    # # i = min(error_unk)
    # # j = logs[error_unk == i][-1]
    # # print(i, j)
    # # plt.scatter(j, i, marker='x', color = 'red', label = 'minimum')
    # plt.xlabel("P_free")
    # plt.ylabel("Error (%)")
    # plt.title("$\mathit{Unknown}$ partial error", fontsize = "12")
    # # plt.legend(prop = { "size": 8 })
    # plt.savefig("unknown partial", dpi = 1024)
    # plt.show()

    # plt.figure()
    # # i = min(error_free)
    # # j = logs[error_free == i]
    # plt.plot(np.array(logs), np.array(error_free), 'k')
    # # plt.scatter(j, i, marker='x', color = 'red', label = 'minimum')
    # plt.xlabel("P_free")
    # plt.ylabel("Error (%)")
    # plt.title("$\mathit{Free}$ partial error", fontsize = "12")
    # # plt.legend(prop = { "size": 8 })
    # plt.savefig("free partial", dpi = 1024)
    # plt.show()

    # plt.figure()
    # plt.plot(np.array(logs), np.array(error_occ), 'k')
    # # i = min(error_occ)
    # # j = logs[error_occ == i]
    # # plt.scatter(j, i, marker='x', color = 'red',label = 'minimum')
    # plt.xlabel("P_free")
    # plt.ylabel("Error (%)")
    # plt.title("$\mathit{Occupied}$ partial error", fontsize = "12")
    # # plt.legend(prop = { "size": 8 })
    # plt.savefig("occupied partial", dpi = 1024)
    # plt.show()

    plt.figure()
    plt.plot(np.array(logs), np.array(error_unk), '.k', label = 'class $\mathit{unknown}$')
    plt.plot(np.array(logs), np.array(error_free), '--k', label = 'class $\mathit{free}$')
    plt.plot(np.array(logs), np.array(error_occ), '*k', label = 'class $\mathit{occupied}$')
    plt.xlabel("P_free")
    plt.ylabel("Error (%)")
    plt.legend(prop = { "size": 8 })
    plt.ylim((0,1))
    plt.title("Classes' partial errors")
    plt.savefig("partials", dpi = 1024)
    plt.show()