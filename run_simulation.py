# Arquivo: run_simulation.py (na pasta raiz do projeto)

from pathlib import Path

from dotenv import load_dotenv

# Importações do LangChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from src.ValidadorNEES.gerador.gerador_prova import GeradorProva
from src.ValidadorNEES.gerador.gerador_respondentes import GeradorRespondentes
from src.ValidadorNEES.infraestrutura.provedor_llm import get_llm
from src.ValidadorNEES.simulador.simulador import Simulador

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent
LLM_PROVIDER = "google"
LLM_MODEL = "gemini-1.5-flash-8b"
NUM_RESPONDENTES = 500
CAMINHO_PROVA = (
    PROJECT_ROOT
    / "data"
    / "01_raw"
    / "ENEM"
    / "2022"
    / "2019_ENUNCIADOS_SEM_IMAGEM.csv"
)
CAMINHO_HABILIDADES = (
    PROJECT_ROOT / "data" / "01_raw" / "ENEM" / "2022" / "habilidades_alunos.csv"
)
CAMINHO_SAIDA_RESULTADOS = (
    PROJECT_ROOT / "data" / "03_processed" / "resultados_simulacao_2019.csv"
)


# para organizar a cadeia no Runnable
def criar_lista_de_mensagens(inputs: dict) -> list:
    """Função que conecta a persona do Respondente com a pergunta do Item."""
    respondente = inputs["respondente"]
    item = inputs["item"]
    return [respondente.get_system_message(), item.get_human_message()]


# orquestrador das chamadas de funções
def main():
    print("--- INICIANDO SIMULAÇÃO TRI COM LLM (VERSÃO OTIMIZADA) ---")

    print("1. Configurando LLM e montando a cadeia LangChain...")
    llm = get_llm(provider=LLM_PROVIDER, model_name=LLM_MODEL, temperature=1.0)

    responder_chain = RunnableLambda(criar_lista_de_mensagens) | llm | StrOutputParser()

    print("2. Gerando população e carregando a prova...")

    gerador_populacao = GeradorRespondentes(
        caminho_habilidades=str(CAMINHO_HABILIDADES)
    )
    populacao = gerador_populacao.gerar_respondentes(numero_de_alunos=NUM_RESPONDENTES)

    gerador_prova = GeradorProva(caminho_prova=str(CAMINHO_PROVA))
    prova = gerador_prova.carregar_prova_ingles()

    # EXECUÇÃO DA SIMULAÇÃO
    print(
        f"3. Iniciando a simulação para {len(populacao)} alunos e {len(prova.itens)} itens..."
    )
    simulador = Simulador(responder_chain=responder_chain)
    df_resultados = simulador.executar(prova, populacao)

    # SALVANDO RESULTADOS
    print(f"\n4. Simulação concluída. Foram geradas {len(df_resultados)} respostas.")
    CAMINHO_SAIDA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_csv(CAMINHO_SAIDA_RESULTADOS, index=False)
    print(f"Resultados salvos com sucesso em: {CAMINHO_SAIDA_RESULTADOS}")
    print("\nAmostra dos resultados:")
    print(df_resultados.head())


if __name__ == "__main__":
    main()
