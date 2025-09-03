import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Supondo que os seus módulos estejam na estrutura src/
from src.ValidadorNEES.tri.estimador import EstimadorTRI
from src.ValidadorNEES.tri.validador import BinDificuldade, ValidadorTRI

# --- 1. CONFIGURAÇÃO DOS CAMINHOS ---
current_path = Path.cwd()
enunciados_path = (
    current_path / "data" / "01_raw" / "ENEM" / "2022" / "2022_com_prob.csv"
)
respostas_path = (
    current_path / "data" / "03_processed" / "resultados_simulacao_2022.csv"
)

# --- 2. PREPARAÇÃO DO DATAFRAME DE PARÂMETROS REAIS (df_real) ---
colunas_para_carregar = [
    "CO_ITEM",
    "NU_PARAM_A",
    "NU_PARAM_B",
    "TP_LINGUA",
    "PROB_ACERTO",
]
df_enunciados = pd.read_csv(enunciados_path, usecols=colunas_para_carregar)  # type: ignore

df_enunciados_filtrado = df_enunciados[
    (df_enunciados["TP_LINGUA"] != 0) & (df_enunciados["TP_LINGUA"] != 1)
].copy()

colunas_finais = ["CO_ITEM", "NU_PARAM_A", "NU_PARAM_B", "PROB_ACERTO"]
df_parametros_reais = df_enunciados_filtrado[colunas_finais]

df_real = df_parametros_reais.rename(
    columns={
        "CO_ITEM": "ID_QUESTÃO",
        "NU_PARAM_A": "A",
        "NU_PARAM_B": "B",
    }
)

# --- 3. PREPARAÇÃO DO DATAFRAME DE PARÂMETROS SIMULADOS (df_simulado) ---
df_respostas = pd.read_csv(respostas_path)
ids_questoes_validas = df_real["ID_QUESTÃO"].unique()
df_respostas_filtrado = df_respostas[df_respostas["item_id"].isin(ids_questoes_validas)]
df_simulado = EstimadorTRI.estimar_parametros(df_respostas_filtrado)  # type: ignore

# --- 4. EXECUÇÃO DA VALIDAÇÃO ---
print("\n--- Iniciando a Validação dos Parâmetros ---")
validador = ValidadorTRI(df_real, df_simulado)
dados_continuos = validador.obter_dados_continuos_para_relatorio()
dados_discretos = validador.obter_dados_discretos_para_relatorio()

# --- 5. GERAÇÃO DO RELATÓRIO VISUAL ---
print("--- Gerando o relatório visual ---")

# Configurar a figura com 2 linhas e 2 colunas para os gráficos
fig, axes = plt.subplots(2, 2, figsize=(18, 16))
fig.suptitle(
    "Relatório de Validação",
    fontsize=20,
    weight="bold",
)

# Gráfico 1: Dispersão para Parâmetro Probabilidade de Acerto
ax1 = axes[0, 0]
reais_prob = dados_continuos["dados_brutos"]["prob_de_acerto_real"]
simulado_prob = dados_continuos["dados_brutos"]["prob_de_acerto_simulada"]
sns.regplot(x=reais_prob, y=simulado_prob, ax=ax1, scatter_kws={"alpha": 0.5})
ax1.plot(
    [reais_prob.min(), reais_prob.max()],
    [reais_prob.min(), reais_prob.max()],
    "r--",
    label="Recuperação Perfeita",
)
corr_prob = dados_continuos["metricas"]["prob_de_acerto_spearman"]
ax1.set_title("Probabilidade de Acerto", fontsize=16)
ax1.set_xlabel("Valor Real", fontsize=12)
ax1.set_ylabel("Valor Simulado", fontsize=12)
ax1.legend()
ax1.text(
    0.05,
    0.95,
    f"Corr = {corr_prob:.3f}",
    transform=ax1.transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
)

# Gráfico 3: Matriz de Confusão
ax3 = axes[1, 0]
labels = dados_discretos["labels"]
# CORREÇÃO: Removido o til de 'matriz_confusao'
matriz = dados_discretos["matriz_confusao"]
sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    ax=ax3,
)
ax3.set_title("Matriz de Confusão (% de acerto)", fontsize=16)
ax3.set_xlabel("Classificação Prevista", fontsize=12)
ax3.set_ylabel("Classificação Real", fontsize=12)

# Painel 4: Resumo das Métricas
ax4 = axes[1, 1]
ax4.axis("off")  # Oculta os eixos do gráfico
metricas_texto = (
    f"**ANÁLISE CONTÍNUA**\n"
    f"RMSE (% de acerto): {dados_continuos['metricas']['prob_de_acerto_rmse']:.3f}\n"
    f"BIAS (% de acerto): {dados_continuos['metricas']['prob_de_acerto_bias']:.3f}\n"
    f"**ANÁLISE DISCRETA**\n"
    f"Acurácia: {dados_discretos['metricas']['acuracia']:.2%}"
)
ax4.text(
    0.0,
    0.7,
    metricas_texto,
    fontsize=14,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=1", fc="aliceblue", alpha=0.8),
)
ax4.set_title("Resumo das Métricas", fontsize=16)

# Salvar o relatório
plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # type: ignore
nome_arquivo_relatorio = "relatorio_visual.png"
plt.savefig(nome_arquivo_relatorio, dpi=300)

print(f"\n--- Relatório visual salvo com sucesso em '{nome_arquivo_relatorio}' ---")

# plt.show() # Descomente esta linha se quiser ver o gráfico interativamente
