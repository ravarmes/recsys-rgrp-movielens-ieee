import matplotlib.pyplot as plt
import numpy as np

# Dados fornecidos
data = {
    "ALS | Activity | 0.0012966": {
        "Active": 0.9301405907229064,
        "Inactive": 0.7924189726144895
    },
    "ALS | Age | 0.0017027": {
        "00": 0.8247613125576507,
        "18": 0.8224719337414289,
        "25": 0.8087658957727243,
        "35": 0.775033909604687,
        "45": 0.770911139322351,
        "50": 0.7636150396897847,
        "56": 0.7724930257069191
    },
    "ALS | Gender | 0.0042653": {
        "Male": 0.8109695826256147,
        "Female": 0.7551584385791826
    },
    "ALS | Agglomerative | 0.0061508": {
        "G1": 0.7551584385791826,
        "G2": 0.886610967050285,
        "G3": 0.7526660815902215,
        "G4": 0.8000541254351657,
        "G5": 0.8711348140385946
    },
    "NCF | Activity | 0.0040016": {
        "Active": 0.8263883606895774,
        "Inactive": 0.6610165934108952
    },
    "NCF | Age | 0.0016120": {
        "00": 0.7193617753340573,
        "18": 0.6929365077639352,
        "25": 0.673799300668831,
        "35": 0.6558222176317067,
        "45": 0.6569813513922259,
        "50": 0.6373916902404612,
        "56": 0.5866545875520105
    },
    "NCF | Gender | 0.0030178": {
        "Male": 0.6705508090774811,
        "Female": 0.6208998266563986
    },
    "NCF | Agglomerative | 0.0048747": {
        "G1": 0.6541057872626246,
        "G2": 0.8144608716178134,
        "G3": 0.6568048672068941,
        "G4": 0.6638845390258113,
        "G5": 0.754320093062109
    },
    "CBF | Activity | 0.0019527": {
        "Active": 0.9726483051026178,
        "Inactive": 0.8695664296742877
    },
    "CBF | Age | 0.0076711": {
        "00": 0.8871245344521566,
        "18": 0.864824843283687,
        "25": 0.8743060143329462,
        "35": 0.8789458313411441,
        "45": 0.9173282004957014,
        "50": 0.8966286946880302,
        "56": 0.8398329529252321
    },
    "CBF | Gender | 0.0005350": {
        "Male": 0.8660332272226908,
        "Female": 0.8523170216501792
    },
    "CBF | Agglomerative | 0.0030414": {
        "G1": 0.8540654976955769,
        "G2": 0.9551758622803523,
        "G3": 0.8725942365766931,
        "G4": 0.8462606080860541,
        "G5": 0.9158294848399023
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
        ax.set_ylabel('NDCG@10')
    else:
        ax.set_yticklabels([])  # Remove os rótulos do eixo y para os outros subplots
    ax.set_ylim(0, 1)  # Definindo a escala do eixo y
    ax.set_title(titles[i])

# Ajustando layout
plt.tight_layout()
plt.show()