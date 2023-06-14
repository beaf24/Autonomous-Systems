import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# a certo ponto substituir esta inicialização por função que vai buscar valores destas variáveis
# a um txt criado por código main.c
p_occ_start=0.65
p_occ_end=0.81
p_occ_increment=0.01

start = int(p_occ_start *100)
end = int(p_occ_end *100)   
increment = int(p_occ_increment *100)

# iterações para o nome de cada txt e png conter o valor do p_occ que lhe está associado
for i in range(start, end, increment):
    p_occ = i/100
    p_free = 1 - p_occ
    filename = f"export_data/export_{p_occ:.2f}.txt"
    output_image = f"algo_maps/map_algo_{p_occ:.2f}.png"

    mapp = np.loadtxt(filename, delimiter=",")

    prob_map = 1 - 1/(1+np.exp(mapp))

    # adicionado para thresholding
    prob_map[prob_map>=p_occ]=1
    prob_map[prob_map<=p_free]=0
    prob_map[(prob_map > p_free) & (prob_map < p_occ)] = 0.5

    fig, ax = plt.subplots()
    # plt.figure(frameon=False)
    ax.set_axis_off()
    # fig.add_axes(ax)
    im = ax.matshow(prob_map, cmap="Greys")
    IM = Image.fromarray(255-np.uint8(prob_map*255))
    IM.save(output_image)
    plt.close()





