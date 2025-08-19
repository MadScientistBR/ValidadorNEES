from typing import List

from .item import Item


class Prova:
    """
    Classe que representa uma prova (Por exemplo, do ENEM).

    Attributes:
        itens (List[Item]): Uma lista contendo todas as questões
    """

    def __init__(self, itens: List[Item]) -> None:
        self.itens = itens
