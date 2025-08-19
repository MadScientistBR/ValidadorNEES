from typing import List

from langchain_core.messages import HumanMessage


class Item:
    """
    Representa um item individual (questão) em uma prova.

    Essa classe armazena informações sobre o enunciado da questão, link para
    eventuais imagens, texto das alternativas, link para eventuais imagens nas
    alternativas e o gabarito correto da questão.

    Attributes:
        id_item (str): Identificador único da questão
        co_posicao (int): Posição do item na prova
        ano (int): Ano da prova em que a questão apareceu
        tx_enunciado (str): Texto do enunciado
        tx_introducao_alternativas (str): Texto de introdução as alternativas
        arquivos_enunciado (List[str]): Lista contendo os link das imagens do enunciado
        tx_alternativas (List[str]): Lista contendo o enunciado das alternativas
        gabarito (str): A resposta correta do item
    """

    def __init__(
        self,
        id_item: str,
        co_posicao: int,
        ano: int,
        tx_enunciado: str,
        tx_introducao_alternativas: str,
        arquivos_enunciado: List[str],
        tx_alternativas: List[str],
        gabarito: str,
    ) -> None:
        """
        Inicializa uma nova instância do item.

        Args:
            id_item (str): Identificador único da questão
            co_posicao (int): Posição do item na prova
            ano (int): Ano da prova em que a questão apareceu
            tx_enunciado (str): Texto do enunciado
            tx_introducao_alternativas (str): Texto de introdução as alternativas
            arquivos_enunciado (List[str]): Lista contendo os link das imagens do enunciado
            tx_alternativas (List[str]): Lista contendo o enunciado das alternativas
            gabarito (str): A resposta correta do item
        """

        self.id_item = id_item
        self.co_posicao = co_posicao
        self.ano = ano
        self.tx_enunciado = tx_enunciado
        self.tx_introducao_alternativas = tx_introducao_alternativas
        self.arquivos_enunciado = arquivos_enunciado
        self.tx_alternativas = tx_alternativas
        self.gabarito = gabarito

    def __repr__(self) -> str:
        """
        Retorna uma representação em string do objeto Item, útil para depuração.
        """
        return (
            f"Item(id_item={self.id_item!r}, "
            f"co_posicao={self.co_posicao!r}, "
            f"ano={self.ano!r}, "
            f"tx_enunciado={self.tx_enunciado!r}, "
            f"tx_introducao_alternativas={self.tx_introducao_alternativas!r}, "
            f"arquivos_enunciado={self.arquivos_enunciado!r}, "
            f"tx_alternativas={self.tx_alternativas!r}, "
            f"gabarito={self.gabarito!r})"
        )

    def get_human_message(self) -> HumanMessage:
        """
        Retorna um objeto langchain_core.messages.human.HumanMessage contendo o enunciado
        da questão e o link para as imagens para a LLM processar o prompt e retornar uma
        resposta.
        """
        content = []

        # Enunciado da questão
        content.append({"type": "text", "text": self.tx_enunciado})

        # Arquivos do enunciado
        for arquivo in self.arquivos_enunciado:
            content.append({"type": "image_url", "image_url": arquivo})

        # Introdução às alternativas
        content.append({"type": "text", "text": self.tx_introducao_alternativas})

        # Adicionando texto das alternativas
        letras = ["A", "B", "C", "D", "E"]
        for letra, texto_alternativa in zip(letras, self.tx_alternativas):
            content.append({"type": "text", "text": f"{letra}) - {texto_alternativa}"})

        return HumanMessage(content=content)
