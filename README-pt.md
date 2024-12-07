<h1 align="center">
    <img alt="RVM" src="https://github.com/ravarmes/recsys-rgrp-movielens-ieee/blob/main/assets/logo.jpg" />
</h1>

<h3 align="center">
  Análise de Justiça de Grupo no Dataset MovieLens
</h3>

<p align="center">Exemplo de agrupamentos utilizando medidas de justiça social </p>

<p align="center">
  <img alt="GitHub language count" src="https://img.shields.io/github/languages/count/ravarmes/recsys-rgrp-movielens-ieee?color=%2304D361">

  <a href="http://www.linkedin.com/in/rafael-vargas-mesquita">
    <img alt="Made by Rafael Vargas Mesquita" src="https://img.shields.io/badge/made%20by-Rafael%20Vargas%20Mesquita-%2304D361">
  </a>

  <img alt="License" src="https://img.shields.io/badge/license-MIT-%2304D361">

  <a href="https://github.com/ravarmes/recsys-rgrp-movielens-ieee/stargazers">
    <img alt="Stargazers" src="https://img.shields.io/github/stars/ravarmes/recsys-rgrp-movielens-ieee?style=social">
  </a>
</p>

<p align="center">
  <a href="#-sobre">Sobre o projeto</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="#-licenca">Licença</a>
</p>

## :page_with_curl: Sobre o projeto <a name="-sobre"/></a>

Este estudo investiga a equidade em sistemas de recomendação utilizando o dataset MovieLens, aplicando estratégias de filtragem colaborativa: ALS, KNN e NMF. Avaliamos a injustiça em diferentes configurações de agrupamento: Gênero, Idade, Avaliações e Aglomerativo. Os resultados indicam variações significativas de injustiça entre as estratégias, com o método Aglomerativo destacando-se por apresentar os maiores níveis de injustiça do grupo na maioria das abordagens. Esta análise sugere a necessidade de uma seleção cuidadosa da estratégia de filtragem e do método de agrupamento para promover sistemas de recomendação mais justos e inclusivos, destacando a importância de considerar múltiplas dimensões de injustiça na concepção destes sistemas.

### Funções de Objetivo Social (Social Objective Functions)

* Individual fairness (Justiça Individual): a perda do usuário i é a estimativa do erro quadrático médio sobre as classificações conhecidas do usuário i;
* Group Fairness (Justiça de Grupo): a perda do grupo Li como a estimativa do erro quadrático médio sobre todas as avaliações conhecidas no grupo i.

### Arquivos

| Arquivo                               | Descrição                                                                                                                                                                                                                                   |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AlgorithmImpartiality                | Classe para promover justiça nas recomendações de algoritmos de sistemas de recomendação.                                                                                                                                                   |
| AlgorithmUserFairness                | Classes para medir a justiça (polarização, justiça individual e justiça do grupo) das recomendações de algoritmos de sistemas de recomendação.                                                                                               |
| RecSys                               | Classe no padrão fábrica para instanciar um sistema de recomendação com base em parâmetros string.                                                                                                                                           |
| RecSysALS                            | Alternating Least Squares (ALS) para Filtragem Colaborativa é um algoritmo que otimiza iterativamente duas matrizes para melhor prever avaliações de usuários em itens, baseando-se na ideia de fatoração de matrizes.                       |
| RecSysCBF | A Filtragem Baseada em Conteúdo para Sistemas de Recomendação é um método que recomenda itens com base na similaridade entre os gêneros dos filmes, utilizando atributos de conteúdo para corresponder às preferências dos usuários. |
| RecSysNCF | A Filtragem Colaborativa Neural para Sistemas de Recomendação utiliza redes neurais para modelar interações complexas entre usuários e itens, prevendo avaliações ausentes por meio de técnicas de aprendizado profundo. |
| Test_FairnessGroup_MovieLens_Activity         | Script de teste do algoritmo de medidas de justiça (AlgorithmUserFairness) considerando o agrupamento dos usuários por número de avaliações                                                                                                |
| Test_FairnessGroup_MovieLens_Age         | Script de teste do algoritmo de medidas de justiça (AlgorithmUserFairness) considerando o agrupamento dos usuários por idade                                                                                                |
| Test_FairnessGroup_MovieLens_Agglomerative         | Script de teste do algoritmo de medidas de justiça (AlgorithmUserFairness) considerando o agrupamento aglomerativo                                                                                                |
| Test_FairnessGroup_MovieLens_Gender         | Script de teste do algoritmo de medidas de justiça (AlgorithmUserFairness) considerando o agrupamento dos usuários por gênero                                                                                                |



## :memo: Licença <a name="-licenca"/></a>

Esse projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE.md) para mais detalhes.

## :email: Contato

Rafael Vargas Mesquita - [GitHub](https://github.com/ravarmes) - [LinkedIn](https://www.linkedin.com/in/rafael-vargas-mesquita) - [Lattes](http://lattes.cnpq.br/6616283627544820) - **ravarmes@hotmail.com**

---

Feito com ♥ by Rafael Vargas Mesquita :wink: