# ME731, Projeto 1: Análise de Componentes Principais

João Eduardo Pastori Garcia, RA 277172 (Unicamp, ME731, Análise Multivariada).

Opção A do projeto: aspectos metodológicos de uma técnica. A técnica é a ACP, aplicada a
oito indicadores de 275 hospitais do SUS no estado de São Paulo em 2025.

## Arquivos

| Arquivo | O que é |
|---|---|
| `codigo_P1.py` | o programa: reproduz todos os números, tabelas e figuras do relatório |
| `painel_definitivo.csv` | painel hospital × ano (275 hospitais, 2015 a 2025, 3.025 linhas, 54 colunas) |
| `figuras/` | as 7 figuras do relatório, em PDF vetorial |
| `resultados/valores.tex` | um comando LaTeX `\v...` para cada número citado no texto |
| `resultados/tab_*.tex` | as tabelas do relatório |
| `resultados/saida.txt` | registro legível de tudo o que o programa calculou |
| `resultados/*.csv` | escores, cargas, distâncias de Mahalanobis e grupos, para conferência |

Tudo em `figuras/` e `resultados/` é gerado pelo programa.

## Como rodar

```
python codigo_P1.py
```

Leva cerca de 5 segundos. O programa baixa o painel deste repositório por um link preso ao
commit `85ac694`, então lê sempre o mesmo arquivo, mesmo que o repositório mude. As saídas vão
para `figuras/` e `resultados/`, ao lado do programa. A semente é fixa (731) e duas execuções
produzem resultados idênticos.

Bibliotecas: numpy, scipy, pandas, matplotlib e scikit-learn (este só para a silhueta e o
índice de Rand ajustado). Testado com Python 3.12 e com Python 3.10 (numpy 1.26, pandas 2.2,
matplotlib 3.8), com os mesmos números. As figuras usam a fonte Lato quando ela está
instalada com o MiKTeX ou o TeX Live; sem ela, saem em DejaVu Sans e os números não mudam.

O programa também confere o texto: verifica numericamente as proposições da Seção 3 do
relatório e tem um `assert` para cada frase interpretativa. Se alguma deixar de valer, ele
para com `AssertionError`.

O relatório (`relatorio_P1.tex`, compilado com pdfLaTeX depois de rodar o programa) lê os
números de `resultados/valores.tex` e as tabelas de `resultados/tab_*.tex`.

## Sobre os dados

O painel vem de um projeto de pesquisa sobre gestão hospitalar no SUS e foi montado a partir
do SIH/SUS e do CNES, bases públicas do DataSUS. O relatório usa só as linhas de 2025; o
programa lê o painel inteiro porque tira dele os nomes dos hospitais (que só existem em 2020 e
2021) e as contagens de hospitais e linhas. A coluna `custo_saida` é faturamento por saída
(valor da AIH), e não custo.
