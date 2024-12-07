import matplotlib.pyplot as plt
import numpy as np

# Dados fornecidos
data = {
    "ALS | Activity": {
        "Active": 0.8135170,
        "Inactive": 0.8122950
    },
    "ALS | Age": {
        "00": 0.815145,
        "18": 0.818140,
        "25": 0.815589,
        "35": 0.807983,
        "45": 0.800129,
        "50": 0.799131,
        "56": 0.801511
    },
    "ALS | Gender": {
        "Male": 0.8126140,
        "Female": 0.8113810
    },
    "ALS | Agglomerative": {
        "G1": 0.811381,
        "G2": 0.805027,
        "G3": 0.805451,
        "G4": 0.817345,
        "G5": 0.814994
    },
    "NCF | Activity": {
        "Active": 0.8062890,
        "Inactive": 0.8025060
    },
    "NCF | Age": {
        "00": 0.820044,
        "18": 0.811279,
        "25": 0.811751,
        "35": 0.811507,
        "45": 0.807663,
        "50": 0.818541,
        "56": 0.814544
    },
    "NCF | Gender": {
        "Male": 0.8079780,
        "Female": 0.8074640
    },
    "NCF | Agglomerative": {
        "G1": 0.810505,
        "G2": 0.809504,
        "G3": 0.810180,
        "G4": 0.808791,
        "G5": 0.807768
    },
    "CBF | Activity": {
        "Active": 0.826594,
        "Inactive": 0.8270930
    },
    "CBF | Age": {
        "00": 0.829836,
        "18": 0.833242,
        "25": 0.831559,
        "35": 0.828452,
        "45": 0.826305,
        "50": 0.824418,
        "56": 0.828255
    },
    "CBF | Gender": {
        "Male": 0.8278510,
        "Female": 0.8276770
    },
    "CBF | Agglomerative": {
        "G1": 0.828584,
        "G2": 0.826750,
        "G3": 0.825567,
        "G4": 0.830658,
        "G5": 0.832387
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

# Determinando o limite global do eixo Y
all_values = []
for groups in data.values():
    all_values.extend(groups.values())

ymin = min(all_values) - 0.0005  # Pequena margem abaixo
ymax = max(all_values) + 0.0010  # Pequena margem acima

# Iterando sobre os dados e os subplots
for i, (alg, groups) in enumerate(data.items()):
    ax = axs[i // 4, i % 4]

    # Selecionar as cores e hachuras com base no índice
    if i % 4 == 0:
        colors = colors_1
        hatches = ['//', '//']
    elif i % 4 == 1:
        colors = colors_2
        hatches = ['//', '//', '//', '//', '//', '//', '//']
    elif i % 4 == 2:
        colors = colors_3
        hatches = ['//', '//']
    else:
        colors = colors_4
        hatches = ['//', '//', '//', '//', '//']

    # Configurando os limites do eixo Y para todos os subplots
    ax.set_ylim(ymin, ymax)

    # Criando as barras
    for j, (group, loss) in enumerate(groups.items()):
        bar = ax.bar(group, loss, color=colors[j % len(colors)], hatch=hatches[j % len(hatches)])
    
    # Configurando rótulos e títulos
    if i in [0, 4, 8]:
        ax.set_ylabel('Group Diversity')
    else:
        ax.set_yticklabels([])  # Remove os rótulos do eixo y para os outros subplots
    
    ax.set_title(titles[i])

# Ajustando layout
plt.tight_layout()
plt.show()
