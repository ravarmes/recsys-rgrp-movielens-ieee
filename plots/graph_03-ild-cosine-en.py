import matplotlib.pyplot as plt
import numpy as np

# Dados fornecidos
data = {
    "ALS | Activity": {
        "Active": 0.328033,
        "Inactive": 0.301388
    },
    "ALS | Age": {
        "00": 0.315705,
        "18": 0.299067,
        "25": 0.290623,
        "35": 0.312620,
        "45": 0.325341,
        "50": 0.341853,
        "56": 0.314255
    },
    "ALS | Gender": {
        "Male": 0.408281,
        "Female": 0.437081
    },
    "ALS | Agglomerative": {
        "G1": 0.437081,
        "G2": 0.429517,
        "G3": 0.415467,
        "G4": 0.400216,
        "G5": 0.406956
    },
    "NCF | Activity": {
        "Active": 0.362572,
        "Inactive": 0.339928
    },
    "NCF | Age": {
        "00": 0.388948,
        "18": 0.382073,
        "25": 0.366133,
        "35": 0.349264,
        "45": 0.345146,
        "50": 0.357089,
        "56": 0.331451
    },
    "NCF | Gender": {
        "Male": 0.455505,
        "Female": 0.452497
    },
    "NCF | Agglomerative": {
        "G1": 0.432640,
        "G2": 0.438289,
        "G3": 0.429463,
        "G4": 0.434551,
        "G5": 0.436447
    },
    "CBF | Activity": {
        "Active": 0.383118,
        "Inactive": 0.439678
    },
    "CBF | Age": {
        "00": 0.388485,
        "18": 0.414707,
        "25": 0.425062,
        "35": 0.431325,
        "45": 0.445817,
        "50": 0.441043,
        "56": 0.408332
    },
    "CBF | Gender": {
        "Male": 0.55910,
        "Female": 0.57126
    },
    "CBF | Agglomerative": {
        "G1": 0.573169,
        "G2": 0.556505,
        "G3": 0.563013,
        "G4": 0.567777,
        "G5": 0.562873
    }
}

# Definindo os títulos dos subplots
titles = [
    "ALS - Activity",
    "ALS - Age",
    "ALS - Gender",
    "ALS - Agglomerative",
    "NCF - Activity",
    "NCF - Age",
    "NCF - Gender",
    "NCF - Agglomerative",
    "CBF - Activity",
    "CBF - Age",
    "CBF - Gender",
    "CBF - Agglomerative"
]

# Paletas de cores
cmap_1 = plt.cm.get_cmap('Blues')
cmap_2 = plt.cm.get_cmap('Purples')
cmap_3 = plt.cm.get_cmap('Oranges')
cmap_4 = plt.cm.get_cmap('Greens')

# Mapeando cores para os subplots
colors_1 = cmap_1(np.linspace(0.3, 0.7, 2))  # Para 2 grupos
colors_2 = cmap_2(np.linspace(0.3, 0.7, 7))  # Para 7 grupos
colors_3 = cmap_3(np.linspace(0.3, 0.7, 2))  # Para 2 grupos
colors_4 = cmap_4(np.linspace(0.3, 0.7, 5))  # Para 5 grupos

# Criando os subplots com altura ajustada
fig, axs = plt.subplots(3, 4, figsize=(20, 15))
fig.subplots_adjust(left=0.314, bottom=0.23, right=0.993, top=0.945, wspace=0.463, hspace=0.451)

# Iterando sobre os dados e os subplots
for i, (alg, groups) in enumerate(data.items()):
    ax = axs[i // 4, i % 4]
    if i % 4 == 0:
        colors = colors_1
        hatches = ['//', '//']  # Diferentes padrões para 2 grupos
    elif i % 4 == 1:
        colors = colors_2
        hatches = ['//', '//', '//', '//', '//', '//', '//']  # Vários padrões para 7 grupos
    elif i % 4 == 2:
        colors = colors_3
        hatches = ['//', '//']  # Diferentes padrões para 2 grupos
    else:
        colors = colors_4
        hatches = ['//', '//', '//', '//', '//']  # Vários padrões para 5 grupos

    for j, (group, loss) in enumerate(groups.items()):
        bar = ax.bar(group, loss, color=colors[j % len(colors)], hatch=hatches[j % len(hatches)])  # Adicionando hachuras
    if i in [0, 4, 8]:  # Apenas para os subplots 1, 5 e 9
        ax.set_ylabel('Diversity Loss')
    else:
        ax.set_yticklabels([])  # Remove os rótulos do eixo y para os outros subplots
    ax.set_ylim(0, 0.58)  # Definindo a escala do eixo y
    ax.set_title(titles[i])

# Ajustando layout
plt.tight_layout()
plt.show()