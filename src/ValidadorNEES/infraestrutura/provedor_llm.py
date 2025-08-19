import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm(provider: str, model_name: str, **kwargs) -> BaseChatModel:
    """
    Fábrica de LLMs que retorna uma instância de um modelo de chat do LangChain
    com base no provedor especificado.

    Exige que as chaves de API estejam configuradas como variáveis de ambiente.
    Ex: GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY.

    Args:
        provider (str): O nome do provedor (ex: 'google', 'openai', 'anthropic').
        model_name (str): O nome específico do modelo.
        **kwargs: Argumentos adicionais para passar ao construtor do modelo (ex: temperature=0.7).

    Returns:
        BaseChatModel: Uma instância do modelo de chat pronta para uso.

    Raises:
        ValueError: Se o provedor não for suportado.
    """
    provider = provider.lower()

    if provider == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("A variável de ambiente GOOGLE_API_KEY não foi definida.")
        return ChatGoogleGenerativeAI(model=model_name, **kwargs)

    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("A variável de ambiente OPENAI_API_KEY não foi definida.")
        return ChatOpenAI(model=model_name, **kwargs)

    else:
        raise ValueError(
            f"Provedor de LLM '{provider}' não é suportado. "
            "Opções válidas: 'google', 'openai'"
        )
