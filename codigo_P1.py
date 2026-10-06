# ME731, Projeto 1 (opção A): ACP de indicadores de hospitais do SUS-SP em 2025
# João Eduardo Pastori Garcia, RA 277172
#
# python codigo_P1.py
# Baixa o painel (URL abaixo) e grava as figuras em figuras/ e, em resultados/, as tabelas,
# os números citados no texto (valores.tex) e um log (saida.txt).
# O relatório não tem número digitado: o .tex lê resultados/valores.tex.

import itertools
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, silhouette_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

# link preso ao commit, para o arquivo não mudar por baixo do relatório
URL = ("https://raw.githubusercontent.com/JoaoEPGarcia/me731-p1/"
       "85ac694b2f650ebcedbbaeec53d09dd21571d964/painel_definitivo.csv")
PASTA = Path(__file__).resolve().parent
DIR_FIG = PASTA / "figuras"
DIR_RES = PASTA / "resultados"

SEMENTE = 731
ANO = 2025
B_BOOT = 2000
B_PARALELA = 1000
N_INICIOS = 100      # inícios do k-médias
Q_CORTE = 0.975      # quantil da qui-quadrado que marca atípico

# chave: (coluna no painel, rótulo, unidade, casas decimais); uti e giro não existem
# no painel. custo_saida é faturamento (valor da AIH), não custo.
VARIAVEIS = {
    "leitos": ("total_leitos_sus", "Leitos SUS", "leitos", 0),
    "uti": (None, "Fração de UTI", "\\%", 1),
    "alta": ("pct_alta_complex", "Alta complexidade", "\\%", 1),
    "fat": ("custo_saida", "Faturamento por saída", "R\\$", 0),
    "tmp": ("tmp", "TMP", "dias", 2),
    "mort": ("mort_sem_excl", "Mortalidade", "\\%", 2),
    "ocup": ("ocupacao_internacao", "Ocupação", "\\%", 1),
    "giro": (None, "Saídas por leito", "saídas/leito", 1),
}
CHAVES = list(VARIAVEIS)
P = len(CHAVES)
idx = {k: i for i, k in enumerate(CHAVES)}
ROTULO = {k: v[1] for k, v in VARIAVEIS.items()}
# no biplot os nomes inteiros ficam encavalados
ROTULO_BIPLOT = {"leitos": "Leitos", "uti": "UTI", "alta": "Alta compl.", "fat": "Faturamento",
                 "tmp": "TMP", "mort": "Mortalidade", "ocup": "Ocupação", "giro": "Saídas/leito"}

REDES = ["OSS", "Filantrópico", "Direta", "Universitário"]
SIGLA_REDE = {"OSS": "OSS", "Filantrópico": "Filant", "Direta": "Direta", "Universitário": "Univ"}

# nome de comando no LaTeX não aceita algarismo: CP1 vira Um
EXTENSO = ["Zero", "Um", "Dois", "Tres", "Quatro", "Cinco", "Seis", "Sete", "Oito", "Nove"]
POR_EXTENSO = "zero um dois três quatro cinco seis sete oito".split()


# ---- estilo dos gráficos (paleta do documento do projeto) ----
AZUL, VERDE, VERMELHO, ROXO = "#377eb8", "#4daf4a", "#e41a1c", "#984ea3"
LARANJA, MARROM, ROSA = "#ff7f00", "#a65628", "#f781bf"
LARANJA_TEXTO = "#ba5c04"    # o laranja puro some quando é texto
AZUL_CLARO = "#c7dbeb"
PRETO, BRANCO = "#0A0310", "#ffffff"
CINZA_20, CINZA_40, CINZA_60, CINZA_70, CINZA_90 = "#e6e6e7", "#aca9ae", "#5d5961", "#403a45", "#110b17"
NOTA = 9.0                   # notas e números dentro das células

CORES_REDE = {"Direta": AZUL, "OSS": VERDE, "Filantrópico": VERMELHO, "Universitário": ROXO}
MARCADOR_REDE = {"Direta": "o", "OSS": "s", "Filantrópico": "D", "Universitário": "^"}
DIVERGENTE = LinearSegmentedColormap.from_list("div", [
    "#a50026", "#d73027", "#f46d43", "#fdae61", "#fee090", "#ffffbf",
    "#e0f3f8", "#abd9e9", "#74add1", "#4575b4", "#313695"])

# a Lato vem com o MiKTeX; sem ela fica a fonte padrão do matplotlib
fonte = "DejaVu Sans"
for raiz in ["~/AppData/Local/Programs/MiKTeX", "C:/Program Files/MiKTeX", "/usr/share/texlive/texmf-dist"]:
    lato = Path(os.path.expanduser(raiz)) / "fonts/truetype/typoland/lato"
    if (lato / "Lato-Regular.ttf").exists():
        for peso in ["Regular", "Bold", "Italic", "BoldItalic"]:
            font_manager.fontManager.addfont(str(lato / f"Lato-{peso}.ttf"))
        fonte = "Lato"
        break
else:
    print("Aviso: não achei a fonte Lato, as figuras vão sair em DejaVu Sans")

plt.rcParams.update({
    "font.family": fonte, "font.size": 12,
    "axes.titlesize": 11.4, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": CINZA_90, "axes.labelsize": 10.8, "axes.labelcolor": CINZA_70,
    "xtick.labelsize": 10.2, "ytick.labelsize": 10.2,
    "xtick.color": CINZA_70, "ytick.color": CINZA_70, "text.color": CINZA_70,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": CINZA_20,
    "grid.linewidth": 0.7, "grid.linestyle": (0, (4, 4)), "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": CINZA_20, "axes.linewidth": 0.8,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 10.2,
    "mathtext.fontset": "custom", "mathtext.rm": fonte, "mathtext.it": fonte + ":italic",
    "mathtext.default": "regular",
    "figure.facecolor": BRANCO, "axes.facecolor": BRANCO, "savefig.facecolor": BRANCO,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def virgula(v, pos):
    # marca de eixo no formato brasileiro (1.500 e 0,5)
    s = "%.10g" % v
    inteiro, _, dec = s.lstrip("-").partition(".")
    if "e" not in s and abs(v) >= 1000:
        inteiro = f"{int(inteiro):,}".replace(",", ".")
    return ("\u2212" if v < 0 else "") + inteiro + ("," + dec if dec else "")


def eixo_virgula(ax, eixos="xy"):
    if "x" in eixos:
        ax.xaxis.set_major_formatter(FuncFormatter(virgula))
    if "y" in eixos:
        ax.yaxis.set_major_formatter(FuncFormatter(virgula))


def cor_texto(fundo):
    # texto preto ou branco, o que contrastar mais com o fundo (critério da WCAG,
    # que com este preto vira um corte em 0,183 de luminância)
    c = np.array(to_rgb(fundo))
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return PRETO if c @ [0.2126, 0.7152, 0.0722] > 0.183 else BRANCO


def separar_rotulos(ax, textos):
    # afasta na vertical os rótulos que se encostam
    ax.figure.canvas.draw()
    r = ax.figure.canvas.get_renderer()
    for _ in range(600):
        caixas = [t.get_window_extent(r).expanded(1.04, 1.25) for t in textos]
        pares = [(i, j) for i, j in itertools.combinations(range(len(textos)), 2)
                 if caixas[i].overlaps(caixas[j])]
        if not pares:
            return
        for i, j in pares:
            if caixas[i].y0 + caixas[i].y1 < caixas[j].y0 + caixas[j].y1:
                i, j = j, i
            for k, sentido in ((i, 1), (j, -1)):     # o de cima sobe, o de baixo desce
                px, py = ax.transData.transform(textos[k].get_position())
                textos[k].set_position(ax.transData.inverted().transform((px, py + 1.5 * sentido)))


# ---- saída para o LaTeX ----
MACROS, LOG = {}, []


def br(x, casas=2, menos="\\ensuremath{-}"):
    # 1.234,56; nas figuras o menos é o caractere −, não o comando do LaTeX
    s = f"{abs(x):,.{casas}f}".translate(str.maketrans(",.", ".,"))
    return menos + s if x < 0 and s.strip("0.,") else s


def macro(nome, valor, nota=""):
    assert nome not in MACROS, nome
    MACROS[nome] = valor
    LOG.append(f"  \\{nome:<28s} = {valor:<22s} {nota}")


def log(*linhas):
    for linha in linhas:
        print(linha)
        LOG.append(str(linha))


def tabela_latex(nome, cabecalho, linhas, colunas):
    texto = ("\\begin{tabular}{" + colunas + "}\n\\toprule\n" + cabecalho + " \\\\\n\\midrule\n"
             + "\n".join(linhas) + "\n\\bottomrule\n\\end{tabular}\n")
    (DIR_RES / nome).write_text(texto, encoding="utf-8")


# ---- métodos ----
def autos(M):
    # autovalores em ordem decrescente; o sinal do autovetor é arbitrário, então
    # deixo positivo o maior coeficiente de cada um
    lam, E = np.linalg.eigh(M)
    ordem = np.argsort(lam)[::-1]
    lam, E = lam[ordem], E[:, ordem]
    return lam, E * np.sign(E[np.abs(E).argmax(axis=0), range(len(lam))])


def autovalores(X):
    return np.sort(np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)))[::-1]


def paralela(X, rng, permutar):
    # percentil 95 dos autovalores de dados sem correlação do mesmo tamanho: normais
    # (Horn, 1965) ou com cada coluna embaralhada (Buja e Eyuboglu, 1992)
    n, p = X.shape
    sim = []
    for _ in range(B_PARALELA):
        if permutar:
            sim.append(autovalores(np.column_stack([rng.permutation(X[:, j]) for j in range(p)])))
        else:
            sim.append(autovalores(rng.standard_normal((n, p))))
    return np.percentile(sim, 95, axis=0)


def n_paralela(lam, ref):
    return next((k for k in range(len(lam)) if lam[k] <= ref[k]), len(lam))


def mardia(X):
    # assimetria e curtose multivariadas (Mardia, 1970), com S de divisor n
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    G = Xc @ np.linalg.inv(Xc.T @ Xc / n) @ Xc.T
    b1 = (G ** 3).sum() / n ** 2
    b2 = (np.diag(G) ** 2).sum() / n
    gl = p * (p + 1) * (p + 2) / 6
    A = n * b1 / 6
    z = (b2 - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    return b1, b2, A, gl, stats.chi2.sf(A, gl), z, 2 * stats.norm.sf(abs(z))


def kmo(R):
    # KMO global, MSA de cada variável e SMC (R² de cada uma contra as outras)
    Ri = np.linalg.inv(R)
    d = np.sqrt(np.diag(Ri))
    fora = ~np.eye(len(R), dtype=bool)
    r2 = R ** 2 * fora
    a2 = (Ri / np.outer(d, d)) ** 2 * fora
    msa = r2.sum(axis=0) / (r2.sum(axis=0) + a2.sum(axis=0))
    return r2.sum() / (r2.sum() + a2.sum()), msa, 1 - 1 / np.diag(Ri)


def eixos_principais(R, m, tol=1e-6):
    # AF por eixos principais iterados, partindo das SMC; para quando alguma
    # comunalidade passa de 1 (caso de Heywood)
    h2 = 1 - 1 / np.diag(np.linalg.inv(R))
    for it in range(1, 1001):
        Rh = R.copy()
        np.fill_diagonal(Rh, h2)
        lam, E = np.linalg.eigh(Rh)
        maiores = np.argsort(lam)[::-1][:m]
        L = E[:, maiores] * np.sqrt(np.clip(lam[maiores], 0, None))
        novo = (L ** 2).sum(axis=1)
        if np.any(novo > 1):
            return True, it, novo
        if np.max(np.abs(novo - h2)) < tol:
            return False, it, novo
        h2 = novo
    return False, 1000, h2


def congruencia(a, b):
    # Tucker: cosseno entre os dois vetores de cargas
    return a @ b / np.sqrt((a @ a) * (b @ b))


def casar(L_base, L, k):
    # outra ACP pode trocar a ordem e o sinal das componentes; fico com a
    # permutação das k primeiras mais parecida com a base
    perm = max(itertools.permutations(range(k)),
               key=lambda pm: sum(abs(congruencia(L_base[:, j], L[:, pm[j]])) for j in range(k)))
    L = L[:, list(perm)]
    phi = np.array([congruencia(L_base[:, j], L[:, j]) for j in range(k)])
    return L * np.sign(phi), perm, np.abs(phi)


def kmedias(Y, k, rng):
    # Lloyd com início k-means++, repetido N_INICIOS vezes; fica a solução de menor SQ.
    # Não uso o KMeans do sklearn porque nestes dados a solução dele muda com a
    # semente em k = 3 e k = 5, e a silhueta de k = 5 oscila entre 0,30 e 0,31.
    n = len(Y)
    melhor_sq, melhor = np.inf, None
    for _ in range(N_INICIOS):
        C = [Y[rng.integers(n)]]
        for _ in range(1, k):
            dist2 = np.min(((Y[:, None] - np.array(C)[None]) ** 2).sum(-1), axis=1)
            C.append(Y[rng.choice(n, p=dist2 / dist2.sum())])
        C = np.array(C)
        for _ in range(300):
            rot = ((Y[:, None] - C[None]) ** 2).sum(-1).argmin(axis=1)
            novo = np.array([Y[rot == j].mean(axis=0) if np.any(rot == j) else C[j] for j in range(k)])
            if np.allclose(novo, C):
                break
            C = novo
        sq = sum(((Y[rot == j] - C[j]) ** 2).sum() for j in range(k))
        if sq < melhor_sq - 1e-9:
            melhor_sq, melhor = sq, rot
    return melhor, melhor_sq


SIGLAS = {"HC", "FMUSP", "BP", "SA", "INCOR", "IDPC", "USP", "AME", "HU", "UNICAMP", "UNESP", "SUS"}
MINUSCULAS = {"DA", "DE", "DO", "DAS", "DOS", "E"}


def arrumar_nome(nome):
    # o cadastro vem em maiúsculas; passo para caixa de título e mantenho as siglas
    palavras = []
    for i, w in enumerate(str(nome).split()):
        if w.upper() in SIGLAS:
            w = w.upper()
        elif i > 0 and w.upper() in MINUSCULAS:
            w = w.lower()
        else:
            w = w.capitalize()
        palavras.append(w)
    return " ".join(palavras)


# ---- figuras ----
def fig_distribuicoes(d, assim):
    fig, eixos = plt.subplots(2, 4, figsize=(7.0, 4.3))
    for ax, k in zip(eixos.ravel(), CHAVES):
        x = d[k].values
        grade = np.linspace(x.min(), x.max(), 400)
        kde = stats.gaussian_kde(x)
        dens = kde(grade)
        ax.fill_between(grade, dens, color=AZUL_CLARO, linewidth=0)
        ax.plot(grade, dens, color=AZUL, linewidth=1.6)
        ax.plot([np.median(x)] * 2, [0, kde(np.median(x))[0]], color=PRETO,
                linewidth=1.0, linestyle=(0, (3, 2)))
        ax.set_title(ROTULO[k])
        ax.set_xlabel(VARIAVEIS[k][2].replace("\\", ""), fontsize=NOTA)
        ax.set_yticks([])
        ax.grid(False)
        ax.set_ylim(0, dens.max() * 1.18)
        ax.text(0.98, 0.97, "assimetria " + br(assim[k], 2, "\u2212"), transform=ax.transAxes,
                ha="right", va="top", fontsize=NOTA, color=CINZA_70)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        eixo_virgula(ax, "x")
        ax.tick_params(axis="x", labelsize=NOTA)
    fig.tight_layout(h_pad=1.6)
    return fig


def fig_atipicos(d, D2, corte, Z):
    n = len(D2)
    ordem = np.argsort(D2)
    q = stats.chi2.ppf((np.arange(1, n + 1) - 0.5) / n, P)
    acima = D2[ordem] > corte

    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_axes([0.05, 0.1, 0.40, 0.80])
    lim = q.max() * 1.05
    ax.plot([0, lim], [0, lim], color=CINZA_60, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.axhline(corte, color=LARANJA, linewidth=1.0)
    ax.scatter(q[~acima], D2[ordem][~acima], s=16, color=CINZA_40, edgecolor=BRANCO, linewidth=0.4, zorder=3)
    ax.scatter(q[acima], D2[ordem][acima], s=26, color=LARANJA, edgecolor=BRANCO, linewidth=0.4, zorder=4)
    ax.text(0.5, corte - 1.2, "corte: quantil 0,975", fontsize=NOTA, color=LARANJA_TEXTO, va="top")
    ax.text(lim * 0.97, lim * 0.97 + 2, "reta y = x", fontsize=NOTA, color=CINZA_60, ha="right")
    rotulos = []
    for j in ordem[-2:]:     # só os dois mais distantes; o resto está na tabela
        pos = np.where(ordem == j)[0][0]
        rotulos.append(ax.text(q[pos] - 0.45, D2[j], str(d.loc[j, "cnes"]), ha="right", va="center",
                               fontsize=NOTA, color=CINZA_70))
    separar_rotulos(ax, rotulos)
    ax.set_xlabel("quantil teórico da χ² com 8 graus de liberdade")
    ax.set_ylabel("distância de Mahalanobis ao quadrado")
    ax.set_title("(a) Distâncias contra o esperado sob normalidade")
    eixo_virgula(ax)

    atip = np.argsort(D2)[::-1][: int((D2 > corte).sum())]
    M = Z[atip]
    ax2 = fig.add_axes([0.60, 0.1, 0.33, 0.80])
    im = ax2.imshow(np.clip(M, -6, 6), cmap=DIVERGENTE, vmin=-6, vmax=6, aspect="auto")
    ax2.set_xticks(range(P))
    ax2.set_xticklabels([ROTULO[k] for k in CHAVES], rotation=55, ha="right", fontsize=NOTA)
    abrev = {"OSS": "OSS", "Filantrópico": "Filant.", "Direta": "Direta", "Universitário": "Univ."}
    ax2.set_yticks(range(len(atip)))
    ax2.set_yticklabels([f"{d.loc[j, 'cnes']} ({abrev[d.loc[j, 'rede']]})" for j in atip], fontsize=8)
    ax2.grid(False)
    for borda in ax2.spines.values():
        borda.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(P):
            if abs(M[i, j]) >= 3:
                fundo = to_hex(DIVERGENTE((np.clip(M[i, j], -6, 6) + 6) / 12))
                ax2.text(j, i, br(M[i, j], 0, "\u2212"), ha="center", va="center", fontsize=7.5,
                         color=cor_texto(fundo))
    ax2.set_title(f"(b) Escore z das {len(atip)} atípicas")
    cb = fig.colorbar(im, ax=ax2, fraction=0.05, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=NOTA)
    cb.set_label("escore z (truncado em \u00b16)", fontsize=NOTA)
    eixo_virgula(cb.ax, "y")
    return fig


def fig_retencao(lam, boot, p95):
    k = np.arange(1, P + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.5))

    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    a.axhline(1, color=CINZA_60, linewidth=1.0, linestyle=(0, (4, 3)))
    a.text(P + 0.15, 1.03, "Kaiser (\u03bb = 1)", ha="right", va="bottom", fontsize=NOTA, color=CINZA_60)
    # a referência normal fica por baixo desta, então só desenho a de permutação
    a.plot(k, p95, color=LARANJA, linewidth=1.6, marker="s", markersize=4)
    a.vlines(k, lo, hi, color=PRETO, linewidth=1.2)
    a.plot(k, lam, color=PRETO, linewidth=2.0, marker="o", markersize=6, zorder=5)
    a.annotate("observado (barra: IC bootstrap 95%)", (1.25, lam[0]), xytext=(8, 0),
               textcoords="offset points", fontsize=NOTA, color=PRETO, va="center")
    a.annotate("análise paralela (percentil 95)", (5, p95[4]), xytext=(-4, 14),
               textcoords="offset points", fontsize=NOTA, color=LARANJA_TEXTO, ha="left")
    a.set_xticks(k)
    a.set_xlabel("componente")
    a.set_ylabel("autovalor")
    a.set_ylim(0, max(hi.max(), lam.max()) * 1.08)
    a.set_title("(a) Autovalores de R")
    eixo_virgula(a)

    acum = 100 * np.cumsum(lam) / lam.sum()
    b.bar(k, acum, color=[PRETO if i < 3 else CINZA_40 for i in range(P)], width=0.66, zorder=2)
    for i, v in enumerate(acum):
        b.text(k[i], v + 1.0, br(v, 0) + "%", ha="center", va="bottom", fontsize=NOTA,
               color=CINZA_70, zorder=4,
               bbox=dict(boxstyle="square,pad=0.05", facecolor=BRANCO, edgecolor="none"))
    for ref in (70, 80):
        b.axhline(ref, color=CINZA_60, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        b.text(P + 0.75, ref, f"{ref}%", fontsize=NOTA, color=CINZA_60, va="center", ha="left")
    b.set_ylim(0, 110)
    b.set_xticks(k)
    b.set_yticks([0, 20, 40, 60, 80, 100])
    b.set_xlabel("número de componentes")
    b.set_ylabel("% da variância total (acumulada)")
    b.set_title("(b) Variância explicada acumulada")
    eixo_virgula(b)
    fig.tight_layout(w_pad=3)
    return fig


def fig_cargas(cargas, lam, k):
    M = cargas[:, :k]
    h2 = (M ** 2).sum(axis=1)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    im = ax.imshow(M, cmap=DIVERGENTE, vmin=-1, vmax=1, aspect="auto")
    for i in range(P):
        for j in range(k):
            fundo = to_hex(DIVERGENTE((M[i, j] + 1) / 2))
            ax.text(j, i, br(M[i, j], 2, "\u2212"), ha="center", va="center", fontsize=NOTA,
                    color=cor_texto(fundo))
        ax.text(k - 0.3, i, "   " + br(h2[i], 2), ha="left", va="center", fontsize=NOTA, color=CINZA_70)
    ax.text(k - 0.3, -0.75, "   comunalidade", ha="left", va="bottom", fontsize=NOTA, color=CINZA_70)
    pct = 100 * lam / lam.sum()
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"CP{j + 1}\n({br(pct[j], 1)}%)" for j in range(k)])
    ax.xaxis.tick_top()
    ax.set_yticks(range(P))
    ax.set_yticklabels([ROTULO[c] for c in CHAVES])
    ax.set_xlim(-0.5, k + 0.9)
    ax.grid(False)
    for borda in ax.spines.values():
        borda.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.set_label("correlação entre variável e componente", fontsize=NOTA)
    eixo_virgula(cb.ax, "y")
    return fig


def fig_biplot(d, Y, lam, cargas, escala):
    F = Y / np.sqrt(lam)     # escores com variância 1
    pct = 100 * lam / lam.sum()
    fig, eixos = plt.subplots(1, 2, figsize=(7.0, 4.5))
    paineis = []
    for ax, (c1, c2) in zip(eixos, [(0, 1), (0, 2)]):
        ax.axhline(0, color=CINZA_20, linewidth=0.8)
        ax.axvline(0, color=CINZA_20, linewidth=0.8)
        ax.grid(False)
        for rede in ["Filantrópico", "OSS", "Direta", "Universitário"]:
            m = (d["rede"] == rede).values
            ax.scatter(F[m, c1], F[m, c2], s=20, marker=MARCADOR_REDE[rede], color=CORES_REDE[rede],
                       edgecolor=BRANCO, linewidth=0.35, zorder=3)
        rotulos, pontas = [], []
        for i, ch in enumerate(CHAVES):
            x, y = escala * cargas[i, c1], escala * cargas[i, c2]
            ax.annotate("", xy=(x, y), xytext=(0, 0), zorder=6,
                        arrowprops=dict(arrowstyle="-|>", color=PRETO, lw=1.3, mutation_scale=10))
            ux, uy = x / np.hypot(x, y), y / np.hypot(x, y)
            if abs(ux) < 0.35:
                ha, va = "center", ("bottom" if uy > 0 else "top")
            else:
                ha, va = ("left" if ux > 0 else "right"), "center"
            rotulos.append(ax.text(x + 0.10 * ux, y + 0.10 * uy, ROTULO_BIPLOT[ch], fontsize=NOTA,
                                   color=PRETO, ha=ha, va=va, zorder=7,
                                   bbox=dict(boxstyle="round,pad=0.10", facecolor=BRANCO, edgecolor="none")))
            pontas.append((x, y))
        ax.set_xlabel(f"CP{c1 + 1} ({br(pct[c1], 1)}% da variância)")
        ax.set_ylabel(f"CP{c2 + 1} ({br(pct[c2], 1)}% da variância)")
        ax.set_title(f"({'ab'[c2 - 1]}) CP{c1 + 1} e CP{c2 + 1}")
        ax.set_aspect("equal", adjustable="datalim")
        eixo_virgula(ax)
        paineis.append((ax, rotulos, pontas))
    legenda = [Line2D([], [], linestyle="none", marker=MARCADOR_REDE[r], markersize=7, color=CORES_REDE[r],
                      markeredgecolor=BRANCO, label=f"{r} ({int((d['rede'] == r).sum())})") for r in REDES]
    fig.legend(handles=legenda, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=3)
    # só dá para afastar os rótulos depois do tight_layout, que muda a escala dos eixos
    for ax, rotulos, pontas in paineis:
        antes = [r.get_position() for r in rotulos]
        separar_rotulos(ax, rotulos)
        for r, (x0, y0), (px, py) in zip(rotulos, antes, pontas):
            x1, y1 = r.get_position()
            if np.hypot(x1 - x0, y1 - y0) > 0.08:   # rótulo que andou ganha um fio até a seta
                ax.plot([px, x1], [py, y1], color=CINZA_60, linewidth=0.6, zorder=5)
    return fig


def fig_sensibilidade(cargas_var, congr, k):
    estilos = {"base": (PRETO, "o", "base (correlação de Pearson)"),
               "sem_atipicos": (LARANJA, "s", "sem as observações atípicas"),
               "log_fat": (MARROM, "D", "log do faturamento por saída"),
               "postos": (ROSA, "^", "correlação de postos (Spearman)")}
    deslocamento = {"base": 0.27, "sem_atipicos": 0.09, "log_fat": -0.09, "postos": -0.27}
    fig, eixos = plt.subplots(1, k, figsize=(7.0, 4.4), sharey=True)
    ypos = np.arange(P)[::-1]
    for j, ax in enumerate(eixos):
        ax.axvline(0, color=CINZA_40, linewidth=0.8)
        ax.grid(axis="x", visible=True, color=CINZA_20, linewidth=0.7, linestyle=(0, (4, 4)))
        ax.grid(axis="y", visible=False)
        for nome, (cor, marcador, _) in estilos.items():
            ax.scatter(cargas_var[nome][:, j], ypos + deslocamento[nome], color=cor, marker=marcador, s=30,
                       edgecolor=BRANCO, linewidth=0.4, zorder=4 if nome == "base" else 3)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xticks([-1, 0, 1])
        menor = min(congr[nome][j] for nome in estilos if nome != "base")
        ax.set_title(f"CP{j + 1}: mín. {br(menor, 2)}")
        eixo_virgula(ax, "x")
    eixos[0].set_yticks(ypos)
    eixos[0].set_yticklabels([ROTULO[c] for c in CHAVES])
    eixos[1].set_xlabel("correlação entre variável e componente")
    legenda = [Line2D([], [], linestyle="none", marker=mk, markersize=7, color=cor, markeredgecolor=BRANCO,
                      label=txt) for cor, mk, txt in estilos.values()]
    fig.legend(handles=legenda, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    return fig


def fig_grupos(perfil_z, composicao, tamanhos):
    g = len(perfil_z)
    fig = plt.figure(figsize=(7.0, 3.3 + 0.35 * g))

    ax = fig.add_axes([0.13, 0.30, 0.47, 0.55])
    M = perfil_z.values
    ax.imshow(np.clip(M, -2, 2), cmap=DIVERGENTE, vmin=-2, vmax=2, aspect="auto")
    for i in range(g):
        for j in range(P):
            fundo = to_hex(DIVERGENTE((np.clip(M[i, j], -2, 2) + 2) / 4))
            ax.text(j, i, br(M[i, j], 1, "\u2212"), ha="center", va="center", fontsize=NOTA,
                    color=cor_texto(fundo))
    ax.set_xticks(range(P))
    ax.set_xticklabels([ROTULO[c] for c in CHAVES], rotation=35, ha="right", fontsize=NOTA)
    ax.set_yticks(range(g))
    ax.set_yticklabels([f"Grupo {i + 1} (n = {tamanhos.iloc[i]})" for i in range(g)])
    ax.grid(False)
    for borda in ax.spines.values():
        borda.set_visible(False)
    ax.set_title("(a) Mediana do escore z de cada variável no grupo")

    ax2 = fig.add_axes([0.70, 0.30, 0.27, 0.55])
    esquerda = np.zeros(g)
    for rede in REDES:
        v = composicao[rede].values
        ax2.barh(np.arange(g), v, left=esquerda, color=CORES_REDE[rede], height=0.62,
                 edgecolor=BRANCO, linewidth=0.6)
        for i in range(g):
            if v[i] >= 9:     # número só nas faixas largas
                ax2.text(esquerda[i] + v[i] / 2, i, br(v[i], 0), ha="center", va="center",
                         fontsize=NOTA, color=cor_texto(CORES_REDE[rede]))
        esquerda += v
    ax2.set_ylim(g - 0.5, -0.5)
    ax2.set_yticks(range(g))
    ax2.set_yticklabels([f"Grupo {i + 1}" for i in range(g)])
    ax2.set_xlim(0, 100)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xlabel("% dos hospitais do grupo")
    ax2.grid(False)
    ax2.set_title("(b) Composição por rede")
    legenda = [Line2D([], [], linestyle="none", marker="s", markersize=9, color=CORES_REDE[r], label=r)
               for r in REDES]
    ax2.legend(handles=legenda, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.32))
    return fig


# ---- dados ----
DIR_FIG.mkdir(exist_ok=True)
DIR_RES.mkdir(exist_ok=True)
log("ME731 P1: ACP de indicadores hospitalares do SUS-SP", "-" * 60,
    f"Python {platform.python_version()} | numpy {np.__version__} | scipy {scipy.__version__} | "
    f"pandas {pd.__version__} | matplotlib {matplotlib.__version__} | semente {SEMENTE}", "")

painel = pd.read_csv(URL, encoding="utf-8-sig")
d = painel[painel["ano"] == ANO].copy()
assert len(d) == d["cnes"].nunique()
assert (d["leitos_sus"] + d["uti_sus"] == d["total_leitos_sus"]).all()
assert np.allclose(100 * d["dias_perm"] / (d["leitos_sus"] * 365), d["ocupacao_internacao"])
assert np.allclose(d["dias_sem_covid"] / d["qtde_sem_covid"], d["tmp"])
assert np.allclose(d["valor_sem_covid"] / d["qtde_sem_covid"], d["custo_saida"])
# em 2025 não há internação de covid, então tanto faz a versão com ou sem covid
assert (d["qtde_covid"] == 0).all()

d["uti"] = 100 * d["uti_sus"] / d["total_leitos_sus"]
d["giro"] = d["qtde_sem_covid"] / d["total_leitos_sus"]
for k, (coluna, *_) in VARIAVEIS.items():
    if coluna:
        d[k] = d[coluna]
d["alta"] *= 100
d["mort"] *= 100
d["rede"] = d["modelo_gestao_proxy"]
# o nome do hospital só aparece em 2020 e 2021, pego o último disponível
nomes = (painel.dropna(subset=["nome_fantasia"]).sort_values("ano")
         .groupby("cnes")["nome_fantasia"].last())
d["nome"] = d["cnes"].map(nomes).fillna("(nome não informado)")
assert d[CHAVES].notna().all().all()
d = d.reset_index(drop=True)

X = d[CHAVES].to_numpy(dtype=float)
n = len(X)
log(f"[dados] {len(painel)} linhas no painel; {n} hospitais em {ANO}; p = {P}")
macro("vN", str(n), "hospitais em 2025")
macro("vSemente", str(SEMENTE))
macro("vNInicios", str(N_INICIOS), "inícios do k-médias")
macro("vNPainel", br(len(painel), 0), "linhas do painel")
macro("vNHospPainel", str(painel["cnes"].nunique()), "hospitais no painel")

contagem = d["rede"].value_counts()
assert contagem.sum() == n and set(contagem.index) == set(REDES)
for rede in REDES:
    macro("vN" + SIGLA_REDE[rede], str(contagem[rede]), rede)
macro("vNSemUTI", str((d["uti_sus"] == 0).sum()), "sem leito de UTI SUS")
macro("vNAltaZero", str((d["alta"] == 0).sum()), "sem saída de alta complexidade")
macro("vNOcupCem", str((d["ocup"] > 100).sum()), "ocupação acima de 100%")

# variáveis que ficaram de fora, o texto cita essas correlações
r_barc = np.corrcoef(d["total_leitos_sus"], d["pont_barcelona"])[0, 1]
r_vol = np.corrcoef(d["total_leitos_sus"], d["qtde"])[0, 1]
r_mort = stats.spearmanr(d["mort_all"], d["mort_sem_excl"])[0]
macro("vRBarcelona", br(r_barc, 2), "leitos x Barcelona")
macro("vRVolume", br(r_vol, 2), "leitos x saídas")
macro("vRMortes", br(r_mort, 3), "Spearman mortalidade geral x ajustada")
log(f"[fora] r(leitos, Barcelona) = {r_barc:.3f}; r(leitos, saídas) = {r_vol:.3f}; "
    f"Spearman(mort_all, mort_sem_excl) = {r_mort:.4f}")

# ocupação = (100/365) * giro * TMP / (1 - fração de UTI), exata (equação 1)
dif = np.max(np.abs(np.log(d["ocup"]) - (np.log(100 / 365) + np.log(d["giro"]) + np.log(d["tmp"])
                                         - np.log(1 - d["uti"] / 100))))
assert dif < 1e-9
log(f"[identidade] diferença máxima em log = {dif:.2e}")

# ---- descritivas ----
assim = {k: stats.skew(d[k]) for k in CHAVES}
linhas = []
for k, (_, rotulo, unidade, c) in VARIAVEIS.items():
    x = d[k]
    linhas.append(f"{rotulo} & {unidade} & {br(x.mean(), c)} & {br(x.std(ddof=1), c)} & "
                  f"{br(x.median(), c)} & {br(x.min(), c)} & {br(x.max(), c)} & {br(assim[k], 2)} \\\\")
    log(f"[descr] {k:7s} média {x.mean():10.3f}  dp {x.std(ddof=1):10.3f}  "
        f"mediana {x.median():10.3f}  assimetria {assim[k]:6.2f}")
tabela_latex("tab_descritivas.tex", "Variável & Unidade & Média & DP & Mediana & Mínimo & Máximo & Assimetria",
             linhas, "llrrrrrr")
macro("vAssimFat", br(assim["fat"], 2))
macro("vAssimLeitos", br(assim["leitos"], 2))
macro("vAssimAlta", br(assim["alta"], 2))
macro("vFatMediana", br(d["fat"].median(), 0))
macro("vFatMax", br(d["fat"].max(), 0))
macro("vNAssimForte", POR_EXTENSO[sum(abs(a) > 1 for a in assim.values())], "variáveis com |assimetria| > 1")
assert sorted(assim, key=lambda k: -abs(assim[k]))[:3] == ["leitos", "fat", "alta"]

R = np.corrcoef(X, rowvar=False)
linhas = []
for i, k in enumerate(CHAVES):
    celulas = [br(R[i, j], 2) if j < i else ("1" if j == i else "") for j in range(P)]
    linhas.append(f"({i + 1}) {ROTULO[k]} & " + " & ".join(celulas) + " \\\\")
tabela_latex("tab_correlacao.tex", "Variável & " + " & ".join(f"({j + 1})" for j in range(P)),
             linhas, "l" + "r" * P)
macro("vRAltaFat", br(R[idx["alta"], idx["fat"]], 2))
macro("vRTmpMort", br(R[idx["tmp"], idx["mort"]], 2))
macro("vRTmpGiro", br(R[idx["tmp"], idx["giro"]], 2))
macro("vRUtiOcup", br(R[idx["uti"], idx["ocup"]], 2))
triang = R[np.triu_indices(P, 1)]
macro("vRMaxAbs", br(np.abs(triang).max(), 2))
macro("vNRAcimaTres", str((np.abs(triang) > 0.3).sum()), "pares com |r| > 0,3")
macro("vNPares", str(len(triang)))
macro("vNRPos", str((triang > 0).sum()), "pares com r positivo")
assert (triang > 0).sum() > len(triang) / 2

# ---- diagnóstico: Mardia, Bartlett, KMO ----
b1, b2, A, gl_m, p_A, z_m, p_z = mardia(X)
log(f"[Mardia] b1 = {b1:.3f}, A = {A:.1f} (gl {gl_m:.0f}), p = {p_A:.2e}; "
    f"b2 = {b2:.2f} (normal: {P * (P + 2)}), z = {z_m:.2f}, p = {p_z:.2e}")
assert p_A < 0.001 and p_z < 0.001
macro("vMardiaBUm", br(b1, 2))
macro("vMardiaA", br(A, 1))
macro("vMardiaGl", str(int(gl_m)))
macro("vMardiaBDois", br(b2, 2))
macro("vMardiaBDoisEsp", str(P * (P + 2)))
macro("vMardiaZ", br(z_m, 1))

# esfericidade de Bartlett
det_R = np.linalg.det(R)
qui2_b = -(n - 1 - (2 * P + 5) / 6) * np.log(det_R)
gl_b = P * (P - 1) / 2
p_b = stats.chi2.sf(qui2_b, gl_b)
log(f"[Bartlett] qui2 = {qui2_b:.1f}, gl = {gl_b:.0f}, p = {p_b:.2e}, det R = {det_R:.5f}")
assert p_b < 0.001
macro("vBartlett", br(qui2_b, 1))
macro("vBartlettGl", str(int(gl_b)))
macro("vDetR", br(det_R, 4))

kmo_g, msa, smc = kmo(R)
log(f"[KMO] global = {kmo_g:.3f}; MSA = " + ", ".join(f"{k} {m:.2f}" for k, m in zip(CHAVES, msa)))
log("[SMC] " + ", ".join(f"{k} {s:.3f}" for k, s in zip(CHAVES, smc)))
tabela_latex("tab_kmo.tex", "Variável & MSA & $R^2_j$ (SMC)",
             [f"{ROTULO[k]} & {br(msa[i], 2)} & {br(smc[i], 2)} \\\\" for i, k in enumerate(CHAVES)], "lrr")
macro("vKMO", br(kmo_g, 3))
i_min = np.argmin(msa)
macro("vMSAMinVar", ROTULO[CHAVES[i_min]].lower())
macro("vMSAMin", br(msa[i_min], 2))
macro("vSMCOcup", br(smc[idx["ocup"]], 2))
macro("vSMCFat", br(smc[idx["fat"]], 2))
# as três menores MSA são as da identidade
assert {CHAVES[i] for i in np.argsort(msa)[:3]} == {"giro", "ocup", "tmp"}

# ---- atípicos ----
Xc = X - X.mean(axis=0)
D2 = np.einsum("ij,jk,ik->i", Xc, np.linalg.inv(np.cov(X, rowvar=False)), Xc)
corte = stats.chi2.ppf(Q_CORTE, P)
corte999 = stats.chi2.ppf(0.999, P)
atip = D2 > corte
n_atip = atip.sum()
log(f"[Mahalanobis] corte {corte:.2f}: {n_atip} atípicos (esperado {n * (1 - Q_CORTE):.1f}); "
    f"acima de {corte999:.2f}: {(D2 > corte999).sum()}; máximo {D2.max():.1f}")
macro("vCorteDois", br(corte, 2))
macro("vCorteTres", br(corte999, 2))
macro("vNAtip", str(n_atip))
macro("vPctAtip", br(100 * n_atip / n, 1))
macro("vNAtipEsp", br(n * (1 - Q_CORTE), 1), "esperado sob normalidade")
macro("vNAtipTres", str((D2 > corte999).sum()))
macro("vDMax", br(D2.max(), 1))

Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
linhas = []
for j in np.argsort(D2)[::-1][:10]:
    extremos = [f"{ROTULO[CHAVES[i]]} ({br(Z[j, i], 1)})" for i in np.argsort(-np.abs(Z[j]))
                if abs(Z[j, i]) >= 2.5][:3]
    nome = arrumar_nome(d.loc[j, "nome"]).replace("&", "\\&")
    linhas.append(f"{d.loc[j, 'cnes']} & {nome} & {d.loc[j, 'rede']} & {br(D2[j], 1)} & "
                  + "; ".join(extremos) + " \\\\")
tabela_latex("tab_atipicos.tex",
             "CNES & Nome (cadastro de 2021) & Rede & $D^2$ & Variáveis com $|z| \\ge 2{,}5$", linhas,
             "r>{\\raggedright\\arraybackslash}p{4.6cm}lr>{\\raggedright\\arraybackslash}p{4.2cm}")
d.assign(D2=D2, atipico=atip)[["cnes", "nome", "rede", "D2", "atipico"] + CHAVES].to_csv(
    DIR_RES / "distancias_mahalanobis.csv", index=False, encoding="utf-8")
atip_rede = d.loc[atip, "rede"].value_counts()
log("[atípicos por rede] " + ", ".join(f"{r} {atip_rede.get(r, 0)}" for r in REDES))
macro("vNAtipUniv", str(atip_rede.get("Universitário", 0)))
macro("vPctUnivAtip", br(100 * atip_rede.get("Universitário", 0) / contagem["Universitário"], 0))

# ---- ACP ----
lam, E = autos(R)
cargas = E * np.sqrt(lam)
Y = Z @ E

# proposições da Seção 3 conferidas nos dados
assert np.isclose(lam.sum(), P)
assert np.allclose(np.cov(Y, rowvar=False), np.diag(lam), atol=1e-10)
assert np.allclose(E.T @ E, np.eye(P))
assert np.allclose(np.corrcoef(Z, Y, rowvar=False)[:P, P:], cargas)
_, s, Vt = np.linalg.svd(Z / np.sqrt(n - 1), full_matrices=False)
assert np.allclose(s ** 2, lam) and np.allclose(np.abs(Vt.T), np.abs(E))
log("[ACP] proposições conferidas (traço, Var(Y), ortonormalidade, cargas, SVD)")

pct = 100 * lam / lam.sum()
acum = np.cumsum(pct)
for k in range(P):
    macro(f"vLam{EXTENSO[k + 1]}", br(lam[k], 2), f"autovalor {k + 1}")
    macro(f"vPct{EXTENSO[k + 1]}", br(pct[k], 1))
    macro(f"vAcum{EXTENSO[k + 1]}", br(acum[k], 1))
macro("vLamOitoTres", br(lam[7], 3), "menor autovalor com 3 casas")
log("[ACP] autovalores: " + ", ".join(f"{v:.3f}" for v in lam))
log("[ACP] % acumulada: " + ", ".join(f"{v:.1f}" for v in acum))

# quantas componentes; cada sorteio tem o seu gerador, assim mexer num não muda os outros
K = 3
k_kaiser = (lam > 1).sum()
k_07 = (lam > 0.7).sum()
k_70 = np.argmax(acum >= 70) + 1
k_80 = np.argmax(acum >= 80) + 1
p95_perm = paralela(X, np.random.default_rng([SEMENTE, 1]), permutar=True)
p95_norm = paralela(X, np.random.default_rng([SEMENTE, 2]), permutar=False)
k_perm = n_paralela(lam, p95_perm)
k_norm = n_paralela(lam, p95_norm)
log(f"[retenção] Kaiser {k_kaiser}; lambda > 0,7: {k_07}; 70%: {k_70}; 80%: {k_80}; "
    f"paralela (permutação) {k_perm}; paralela (normal) {k_norm}")
log("[paralela] p95 permutação: " + ", ".join(f"{v:.3f}" for v in p95_perm))
log("[paralela] p95 normal:     " + ", ".join(f"{v:.3f}" for v in p95_norm))
# o texto: 3 por Kaiser, paralela e 70%; 4 por 80% e por lambda > 0,7
assert k_kaiser == k_perm == k_norm == k_70 == K
assert k_80 == 4 and k_07 == 4
macro("vParPermTres", br(p95_perm[2], 2))
macro("vParPermQuatro", br(p95_perm[3], 2))
macro("vParNormTres", br(p95_norm[2], 2))
macro("vParNormQuatro", br(p95_norm[3], 2))
macro("vParPermUm", br(p95_perm[0], 2))
macro("vBPar", br(B_PARALELA, 0))

rng = np.random.default_rng([SEMENTE, 3])
boot = np.array([autovalores(X[rng.integers(0, n, n)]) for _ in range(B_BOOT)])
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
lo3, hi3 = np.percentile(100 * boot[:, :K].sum(axis=1) / P, [2.5, 97.5])
pct_tres = 100 * np.mean((boot > 1).sum(axis=1) == K)
assert lo[2] > 1 and hi[3] < 1
macro("vBBoot", br(B_BOOT, 0))
for k in range(4):
    macro(f"vLamLo{EXTENSO[k + 1]}", br(lo[k], 2))
    macro(f"vLamHi{EXTENSO[k + 1]}", br(hi[k], 2))
macro("vAcumTresLo", br(lo3, 1))
macro("vAcumTresHi", br(hi3, 1))
macro("vPctBootKaiser", br(pct_tres, 1), "% das reamostras com exatamente 3 autovalores > 1")
log("[bootstrap] IC 95% dos autovalores 1 a 4: "
    + "; ".join(f"[{a:.2f}, {b:.2f}]" for a, b in zip(lo[:4], hi[:4])))
log(f"[bootstrap] % acumulada com 3: [{lo3:.1f}, {hi3:.1f}]; exatamente 3 > 1 em {pct_tres:.1f}%")

linhas = [f"CP{k + 1} & {br(lam[k], 2)} & [{br(lo[k], 2)}; {br(hi[k], 2)}] & {br(pct[k], 1)} & "
          f"{br(acum[k], 1)} & {br(p95_perm[k], 2)} & {br(p95_norm[k], 2)} \\\\" for k in range(P)]
tabela_latex("tab_autovalores.tex",
             "Componente & $\\hat\\lambda_k$ & IC 95\\% bootstrap & \\% var. & \\% acum. & "
             "Paralela (perm.) & Paralela (normal)", linhas, "lrcrrrr")

h2 = (cargas[:, :K] ** 2).sum(axis=1)
linhas = []
for i, k in enumerate(CHAVES):
    coefs = " & ".join(br(E[i, j], 2) for j in range(K))
    cors = " & ".join(br(cargas[i, j], 2) for j in range(K))
    linhas.append(f"{ROTULO[k]} & {coefs} & {cors} & {br(h2[i], 2)} \\\\")
tabela_latex("tab_cargas.tex",
             "& \\multicolumn{3}{c}{Coeficientes $\\hat e_{ik}$} & \\multicolumn{3}{c}{Correlações $r_{Z_i,Y_k}$} & \\\\\n"
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nVariável & CP1 & CP2 & CP3 & CP1 & CP2 & CP3 & $h_i^2$",
             linhas, "lrrrrrrr")
pd.DataFrame(np.column_stack([E, cargas]), index=[ROTULO[k] for k in CHAVES],
             columns=[f"e{j + 1}" for j in range(P)] + [f"carga{j + 1}" for j in range(P)]
             ).to_csv(DIR_RES / "autovetores_e_cargas.csv", encoding="utf-8")
for j in range(K):
    log(f"[cargas] CP{j + 1}: " + ", ".join(f"{k} {cargas[i, j]:.2f}" for i, k in enumerate(CHAVES)))
log("[comunalidade] " + ", ".join(f"{k} {v:.2f}" for k, v in zip(CHAVES, h2)))
for i, k in enumerate(CHAVES):
    for j in range(K):
        macro(f"vC{EXTENSO[j + 1]}{k.capitalize()}", br(cargas[i, j], 2), f"carga de {k} na CP{j + 1}")
    macro(f"vHDois{k.capitalize()}", br(h2[i], 2), f"comunalidade de {k}")
macro("vMenorHDois", br(h2.min(), 2))
macro("vLeitosCQuatro", br(cargas[idx["leitos"], 3], 2), "carga de leitos na CP4")

# a interpretação do texto depende destes sinais e ordens; se algum mudar, revisar a Seção 5
c = cargas
assert all(c[i, 0] > 0 for i in range(P) if CHAVES[i] != "giro") and abs(c[idx["giro"], 0]) < 0.15
assert np.argmax(c[:, 0]) == idx["fat"]
assert c[idx["giro"], 1] > 0.8 and c[idx["tmp"], 1] < -0.5 and c[idx["mort"], 1] < -0.4
assert c[idx["alta"], 2] > 0.5 and c[idx["fat"], 2] > 0.4
assert c[idx["ocup"], 2] < -0.5 and c[idx["mort"], 2] < -0.4 and c[idx["giro"], 2] < -0.3
assert np.argmin(h2) == idx["leitos"]
outras = [i for i in range(P) if i != idx["leitos"]]
assert CHAVES[outras[np.argmin(h2[outras])]] == "uti" and CHAVES[outras[np.argmax(h2[outras])]] == "giro"
assert np.argmax(np.abs(c[:, 3])) == idx["leitos"] and abs(c[idx["leitos"], 3]) > 0.7
assert np.sort(np.abs(c[:, 3]))[-2] < 0.3       # a CP4 é praticamente só leitos

# as duas últimas componentes são as redundâncias
e8, e7 = E[:, -1], E[:, -2]
log("[CP8] " + ", ".join(f"{k} {e8[i]:.3f}" for i, k in enumerate(CHAVES)))
log("[CP7] " + ", ".join(f"{k} {e7[i]:.3f}" for i, k in enumerate(CHAVES)))
for k in ("ocup", "giro", "tmp", "uti"):
    macro(f"vEOito{k.capitalize()}", br(e8[idx[k]], 2), f"coeficiente de {k} na CP8")
resto = [idx[k] for k in ("leitos", "alta", "fat", "mort")]
macro("vEOitoOutrosMax", br(np.abs(e8[resto]).max(), 2))
assert np.abs(e8[resto]).max() < 0.1
macro("vESeteAlta", br(e7[idx["alta"]], 2))
macro("vESeteFat", br(e7[idx["fat"]], 2))

# com log de ocupação, giro e TMP e -log(1 - uti) a identidade fica linear, e o menor
# autovalor vai a zero (aqui sai ~1e-30; a folga é para outra BLAS)
Xlog = X.copy()
for k in ("ocup", "giro", "tmp"):
    Xlog[:, idx[k]] = np.log(X[:, idx[k]])
Xlog[:, idx["uti"]] = -np.log(1 - X[:, idx["uti"]] / 100)
menor_log = autovalores(Xlog)[-1]
log(f"[CP8] menor autovalor com as variáveis em log: {menor_log:.2e}")
assert abs(menor_log) < 1e-10

sem_ocup = [i for i in range(P) if i != idx["ocup"]]
kmo_so, msa_so, _ = kmo(np.corrcoef(X[:, sem_ocup], rowvar=False))
lam_so = autovalores(X[:, sem_ocup])
log(f"[KMO] sem a ocupação: {kmo_so:.3f}; autovalores: " + ", ".join(f"{v:.3f}" for v in lam_so))
assert kmo_g < 0.5 < kmo_so
macro("vKMOSemOcup", br(kmo_so, 3))
macro("vMSAGiroSemOcup", br(msa_so[sem_ocup.index(idx["giro"])], 2))
macro("vLamMinSemOcup", br(lam_so[-1], 2))

# D2 = soma dos escores padronizados ao quadrado (equação 2); quem é atípico só nas últimas CP
F = Y / np.sqrt(lam)
assert np.allclose((F ** 2).sum(axis=1), D2)
escondidos = np.where(atip & ~(np.abs(F[:, :K]) >= 2.5).any(axis=1))[0]
cp_maior = np.abs(F[escondidos]).argmax(axis=1) + 1
log(f"[D2] atípicos sem escore >= 2,5 nas 3 primeiras CP: {len(escondidos)} de {n_atip}: "
    + ", ".join(f"{d.loc[j, 'cnes']} (CP{cp})" for j, cp in zip(escondidos, cp_maior)))
macro("vNAtipInvisivel", str(len(escondidos)))
macro("vNAtipInvisivelUltimas", str((cp_maior >= 6).sum()), "maior escore nas CP6 a CP8")
macro("vNAtipInvisivelQuatro", str((cp_maior == 4).sum()), "maior escore na CP4")

escores = pd.DataFrame(Y[:, :K], columns=["CP1", "CP2", "CP3"])
pd.concat([d[["cnes", "rede"]], escores], axis=1).to_csv(DIR_RES / "escores.csv", index=False, encoding="utf-8")
medianas = escores.groupby(d["rede"]).median()
log("[escores por rede, mediana]\n" + medianas.round(2).to_string())
tabela_latex("tab_escores_rede.tex", "Rede & $n$ & CP1 & CP2 & CP3",
             [f"{r} & {contagem[r]} & " + " & ".join(br(medianas.loc[r, f"CP{j + 1}"], 2) for j in range(K))
              + " \\\\" for r in REDES], "lrrrr")
for r in REDES:
    for j in range(K):
        macro(f"vMed{SIGLA_REDE[r]}{EXTENSO[j + 1]}", br(medianas.loc[r, f"CP{j + 1}"], 2))
assert medianas["CP1"].idxmax() == "Universitário" and medianas["CP1"].idxmin() == "Filantrópico"
assert medianas["CP2"].idxmin() == "Direta" and medianas["CP3"].idxmin() == "OSS"

# ---- e se fosse com a matriz de covariância? ----
S = np.cov(X, rowvar=False)
lam_S, E_S = autos(S)
pct_S1 = 100 * lam_S[0] / lam_S.sum()
var_fat = 100 * S[idx["fat"], idx["fat"]] / np.trace(S)
log(f"[S] CP1 explica {pct_S1:.1f}%; coeficiente do faturamento {E_S[idx['fat'], 0]:.4f}; "
    f"faturamento = {var_fat:.1f}% do traço")
assert E_S[idx["fat"], 0] > 0.99     # a CP1 de S é praticamente só o faturamento
macro("vPctSUm", br(pct_S1, 1))
macro("vESFat", br(E_S[idx["fat"], 0], 3))
macro("vPctVarFat", br(var_fat, 1))
macro("vDPFat", br(np.sqrt(S[idx["fat"], idx["fat"]]), 0))
macro("vDPTmp", br(np.sqrt(S[idx["tmp"], idx["tmp"]]), 2))

# ---- sensibilidade ----
X_logfat = X.copy()
X_logfat[:, idx["fat"]] = np.log(X[:, idx["fat"]])
variantes = [
    ("base", "Base (Pearson)", "Base", X),
    ("sem_atipicos", "Sem as atípicas", "Sem", X[~atip]),
    ("log_fat", "Log do faturamento", "Log", X_logfat),
    ("postos", "Postos (Spearman)", "Postos", stats.rankdata(X, axis=0)),
]
cargas_var, congr, linhas = {}, {}, []
for i, (nome, rotulo, tag, Xv) in enumerate(variantes):
    lv, Ev = autos(np.corrcoef(Xv, rowvar=False))
    Lc, perm, phi = casar(cargas[:, :K], Ev * np.sqrt(lv), K)
    cargas_var[nome], congr[nome] = Lc, phi
    ref = paralela(Xv, np.random.default_rng([SEMENTE, 10 + i]), permutar=True)
    k_par = n_paralela(lv, ref)
    k_kai = (lv > 1).sum()
    acum3 = 100 * lv[:K].sum() / P
    linhas.append(f"{rotulo} & {len(Xv)} & " + " & ".join(br(v, 2) for v in lv[:4])
                  + f" & {br(acum3, 1)} & {k_kai} & {k_par} & " + " & ".join(br(v, 2) for v in phi) + " \\\\")
    log(f"[sens] {nome:13s} n={len(Xv)} lam={np.round(lv[:4], 3)} acum3={acum3:.1f} "
        f"Kaiser={k_kai} paralela={k_par} ordem={perm} congruência={np.round(phi, 3)}")
    assert perm == (0, 1, 2)     # as componentes não trocaram de lugar
    macro(f"vSens{tag}Acum", br(acum3, 1))
    macro(f"vSens{tag}LamTres", br(lv[2], 2))
    macro(f"vSens{tag}Kaiser", str(k_kai))
    macro(f"vSens{tag}Par", str(k_par))
    macro(f"vSens{tag}ParTres", br(ref[2], 2))
    for j in range(K):
        macro(f"vSens{tag}Phi{EXTENSO[j + 1]}", br(phi[j], 2))
    if nome == "sem_atipicos":
        macro("vSemNAtip", str(len(Xv)))
        assert k_kai == 3 and k_par == 2
    if nome == "log_fat":
        assert k_kai == 3 and k_par == 3
    if nome == "postos":
        assert k_kai == 2 and k_par == 2
        macro("vPostosCUmGiro", br(Lc[idx["giro"], 0], 2))
        macro("vPostosCUmLeitos", br(Lc[idx["leitos"], 0], 2))
        macro("vPostosLamUm", br(lv[0], 2))
        macro("vPostosPctUm", br(100 * lv[0] / P, 1))
tabela_latex("tab_sensibilidade.tex",
             "Variante & $n$ & $\\hat\\lambda_1$ & $\\hat\\lambda_2$ & $\\hat\\lambda_3$ & $\\hat\\lambda_4$ & "
             "\\% acum. (3) & Kaiser & Paralela & $\\phi_1$ & $\\phi_2$ & $\\phi_3$", linhas, "lrrrrrrrrrrr")
outras_var = [v for v in congr if v != "base"]
phi_min = min(congr[v][j] for v in outras_var for j in range(K))
phi_12 = min(congr[v][j] for v in outras_var for j in range(2))
phi_3 = min(congr[v][2] for v in outras_var)
log(f"[sens] menor congruência: geral {phi_min:.3f}, CP1 e CP2 {phi_12:.3f}, CP3 {phi_3:.3f}")
# Lorenzo-Seva e ten Berge: >= 0,95 iguais; 0,85 a 0,94 parecidas
assert phi_12 >= 0.95 and 0.85 <= phi_3 < 0.95
macro("vPhiMin", br(phi_min, 2))
macro("vPhiMinTres", br(phi_3, 2))
macro("vPhiMinUmDois", br(phi_12, 2))

# ---- análise fatorial (eixos principais, 3 fatores) ----
heywood, iteracao, h2_af = eixos_principais(R, K)
i_hey = np.argmax(h2_af)
log(f"[AF] Heywood = {heywood} na iteração {iteracao}; comunalidades: "
    + ", ".join(f"{k} {v:.3f}" for k, v in zip(CHAVES, h2_af)))
assert heywood
macro("vAFIter", str(iteracao))
macro("vAFVar", ROTULO[CHAVES[i_hey]].lower())
macro("vAFHDois", br(h2_af[i_hey], 3))

# ---- agrupamento nos escores das 3 componentes ----
Yk = Y[:, :K]
grupos_k, sq, sil = {}, {}, {}
for g in range(2, 7):
    grupos_k[g], sq[g] = kmedias(Yk, g, np.random.default_rng([SEMENTE, 100 + g]))
    sil[g] = silhouette_score(Yk, grupos_k[g])
log("[k-médias] silhueta: " + ", ".join(f"k={g} {s:.3f}" for g, s in sil.items()))
log("[k-médias] SQ dentro: " + ", ".join(f"k={g} {s:.1f}" for g, s in sq.items()))
# a silhueta quase não muda de k = 2 a 6, nenhum k se destaca; fico com 4 pela interpretação
assert max(sil.values()) - min(sil.values()) < 0.02 and max(sil.values()) < 0.35
macro("vSilMin", br(min(sil.values()), 2))
macro("vSilMax", br(max(sil.values()), 2))
for g in sil:
    macro(f"vSil{EXTENSO[g]}", br(sil[g], 2))
for g in range(3, 7):
    macro(f"vRedSQ{EXTENSO[g]}", br(100 * (sq[g - 1] - sq[g]) / sq[g - 1], 0), f"redução da SQ de {g - 1} para {g}")
g_max = max(sil, key=sil.get)
macro("vGSil", POR_EXTENSO[g_max])

N_GRUPOS = 4
rot = grupos_k[N_GRUPOS]
# numero os grupos pela mediana da CP1, para a numeração não depender do sorteio
ordem_g = pd.Series(Yk[:, 0]).groupby(rot).median().sort_values().index
rot = pd.Series(rot).map({g: i for i, g in enumerate(ordem_g)}).to_numpy()
ward = fcluster(linkage(Yk, method="ward"), N_GRUPOS, criterion="maxclust")
ari = adjusted_rand_score(rot, ward)
macro("vARI", br(ari, 2), "Rand ajustado k-médias x Ward")
log(f"[grupos] maior silhueta em k = {g_max}; usei k = {N_GRUPOS}; Rand ajustado com Ward = {ari:.3f}")

tam = pd.Series(rot).value_counts().sort_index()
perfil_z = pd.DataFrame(Z, columns=CHAVES).groupby(rot).median()
perfil = d[CHAVES].groupby(rot).median()
comp = pd.crosstab(rot, d["rede"]).reindex(columns=REDES, fill_value=0)
comp_pct = 100 * comp.div(comp.sum(axis=1), axis=0)
med_g = pd.DataFrame(Yk).groupby(rot).median()
log("[grupos] tamanhos: " + ", ".join(f"G{i + 1} {t}" for i, t in tam.items()))
log("[grupos] medianas:\n" + perfil.round(2).to_string())
log("[grupos] por rede:\n" + comp.to_string())
log("[grupos] mediana dos escores:\n" + med_g.round(2).to_string())

# o que o texto diz de cada grupo
assert perfil.loc[0, "uti"] == 0 and perfil.loc[0, "alta"] == 0
assert perfil["leitos"].idxmin() == 0 and perfil["ocup"].idxmin() == 0 and med_g[0].idxmin() == 0
assert perfil["giro"].idxmax() == 1 and abs(med_g.loc[1, 0]) < 0.5 and med_g.loc[1, 1] > 0.5
assert perfil["tmp"].idxmax() == 2 and perfil["mort"].idxmax() == 2
assert med_g.loc[2, 0] > 0.5 and med_g.loc[2, 1] < -0.5
assert perfil["alta"].idxmax() == 3 and perfil["fat"].idxmax() == 3
assert med_g[0].idxmax() == 3 and med_g[2].idxmax() == 3

linhas = []
for i in range(N_GRUPOS):
    g = EXTENSO[i + 1]
    macro(f"vG{g}N", str(tam.iloc[i]))
    for k in CHAVES:
        macro(f"vG{g}{k.capitalize()}", br(perfil.loc[i, k], VARIAVEIS[k][3]))
    for r in REDES:
        macro(f"vG{g}Pct{SIGLA_REDE[r]}", br(comp_pct.loc[i, r], 0))
        macro(f"vG{g}N{SIGLA_REDE[r]}", str(comp.loc[i, r]))
    linhas.append(f"Grupo {i + 1} & {tam.iloc[i]} & "
                  + " & ".join(br(perfil.loc[i, k], VARIAVEIS[k][3]) for k in CHAVES) + " \\\\")
tabela_latex("tab_grupos.tex", "Grupo & $n$ & Leitos & UTI (\\%) & Alta (\\%) & Fat. (R\\$) & TMP & "
             "Mort. (\\%) & Ocup. (\\%) & Saídas/leito", linhas, "lrrrrrrrrr")
pd.DataFrame({"cnes": d["cnes"], "rede": d["rede"], "grupo": rot + 1}).to_csv(
    DIR_RES / "grupos.csv", index=False, encoding="utf-8")
for r in REDES:
    macro(f"vPctGeral{SIGLA_REDE[r]}", br(100 * contagem[r] / n, 0))

# ---- figuras ----
ESCALA_SETAS = 2.5
figuras = {
    "fig01_distribuicoes.pdf": fig_distribuicoes(d, assim),
    "fig02_atipicos.pdf": fig_atipicos(d, D2, corte, Z),
    "fig03_retencao.pdf": fig_retencao(lam, boot, p95_perm),
    "fig04_cargas.pdf": fig_cargas(cargas, lam, K),
    "fig05_biplot.pdf": fig_biplot(d, Y, lam, cargas, ESCALA_SETAS),
    "fig06_sensibilidade.pdf": fig_sensibilidade(cargas_var, congr, K),
    "fig07_grupos.pdf": fig_grupos(perfil_z, comp_pct, tam),
}
for nome, fig in figuras.items():
    fig.savefig(DIR_FIG / nome)
    plt.close(fig)
macro("vEscalaBiplot", br(ESCALA_SETAS, 1))
kb = {nome: (DIR_FIG / nome).stat().st_size / 1024 for nome in figuras}
log(f"[figuras] {len(kb)} arquivos, {sum(kb.values()):.0f} kB: "
    + ", ".join(f"{nome} {v:.0f} kB" for nome, v in kb.items()))

cabecalho = ("% Gerado por codigo_P1.py (não editar à mão)\n"
             f"% Python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, "
             f"pandas {pd.__version__}, matplotlib {matplotlib.__version__}; semente {SEMENTE}\n")
comandos = [f"\\newcommand{{\\{nome}}}{{{valor}}}" for nome, valor in MACROS.items()]
(DIR_RES / "valores.tex").write_text(cabecalho + "\n".join(comandos) + "\n", encoding="utf-8")
(DIR_RES / "saida.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
print(f"\n{len(MACROS)} valores gravados em resultados/valores.tex")
