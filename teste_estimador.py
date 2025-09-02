import json
from pathlib import Path

import pandas as pd

from src.ValidadorNEES.tri.estimador import EstimadorTRI
from src.ValidadorNEES.tri.validador import ValidadorTRI

# --- 1. CONFIGURAÇÃO DOS CAMINHOS ---
current_path = Path.cwd()
enunciados_path = (
    current_path / "data" / "01_raw" / "ENEM" / "2022" / "itens_com_enunciados.csv"
)
respostas_path = current_path / "data" / "03_processed" / "resultados_simulacao.csv"

# --- 2. PREPARAÇÃO DO DATAFRAME DE PARÂMETROS REAIS (df_real) ---

# Melhoria de Desempenho: Carregar apenas as colunas necessárias desde o início
colunas_para_carregar = ["CO_ITEM", "NU_PARAM_A", "NU_PARAM_B", "TP_LINGUA"]
df_enunciados = pd.read_csv(enunciados_path, usecols=colunas_para_carregar)  # type: ignore

# Filtro para remover as questões de língua estrangeira (Inglês/Espanhol)
df_enunciados_filtrado = df_enunciados[
    (df_enunciados["TP_LINGUA"] != 0) & (df_enunciados["TP_LINGUA"] != 1)
].copy()

# Correção do Erro Principal: Selecionar colunas usando a sintaxe correta
colunas_finais = ["CO_ITEM", "NU_PARAM_A", "NU_PARAM_B"]
df_parametros_reais = df_enunciados_filtrado[colunas_finais]

# Renomear colunas para o formato esperado pelo Validador
df_real = df_parametros_reais.rename(
    columns={
        "CO_ITEM": "ID_QUESTÃO",
        "NU_PARAM_A": "A",
        "NU_PARAM_B": "B",
    }
)

# --- 3. PREPARAÇÃO DO DATAFRAME DE PARÂMETROS SIMULADOS (df_simulado) ---

df_respostas = pd.read_csv(respostas_path)

# Correção do Erro Lógico: Filtrar as respostas simuladas para que correspondam
# apenas às questões presentes no DataFrame de parâmetros reais.
ids_questoes_validas = df_real["ID_QUESTÃO"].unique()
df_respostas_filtrado = df_respostas[df_respostas["item_id"].isin(ids_questoes_validas)]
df_simulado = EstimadorTRI.estimar_parametros(df_respostas_filtrado)  # type: ignore

# --- 4. EXECUÇÃO DA VALIDAÇÃO ---

print("\n--- Iniciando a Validação dos Parâmetros ---")
validador = ValidadorTRI(df_real, df_simulado)
dados_relatorio = validador.obter_dados_para_relatorio()

# Imprime as métricas do relatório

print("\n--- Resultados da Validação ---")
print(json.dumps(dados_relatorio["metricas"], indent=4))
