import ast
import os

import pandas as pd

from ..core.item import Item
from ..core.prova import Prova


class GeradorProva:
    """
    Classe responsável por ler um arquivo (.csv) contendo os dados das questões e gerar
    um objeto (Prova) contendo uma lista contendo todas as questões (objetos de Item).

    Attribute:
        caminho_prova (str): caminho contendo os dados da prova
    """

    def __init__(self, caminho_prova: str) -> None:
        if not (os.path.isfile(caminho_prova)):
            raise ValueError("O Caminho fornecido para o arquivo não existe!")

        self.caminho_prova = caminho_prova

    def carregar_prova_ingles(self) -> Prova:
        """
        Gera um objeto (Prova) desconsiderando as questões de espanhol.
        """
        df_questoes = pd.read_csv(self.caminho_prova)

        # Removendo questões de espanhol
        df_questoes = df_questoes[df_questoes["TP_LINGUA"] != 0]

        itens = []

        for _, row in df_questoes.iterrows():
            id_item = row["CO_ITEM"]
            ano = row["ANO"]
            co_posicao = row["CO_POSICAO"]
            tx_enunciado = row["TX_ENUNCIADO"]
            tx_introducao_alternativas = row["TX_INTRODUCAO_ALTERNATIVAS"]

            arquivos_enunciado = []
            if pd.notna(row["ARQUIVOS_ENUNCIADO"]):  # type: ignore
                arquivos_enunciado = ast.literal_eval(str(row["ARQUIVOS_ENUNCIADO"]))

            tx_alternativas = [
                str(row[f"TX_ALTERNATIVA_{letra}"])
                for letra in ["A", "B", "C", "D", "E"]
            ]
            gabarito = row["TX_GABARITO"]

            itens.append(
                Item(
                    id_item=str(id_item),
                    co_posicao=int(co_posicao),
                    ano=int(ano),
                    tx_enunciado=str(tx_enunciado),
                    tx_introducao_alternativas=str(tx_introducao_alternativas),
                    arquivos_enunciado=arquivos_enunciado,
                    tx_alternativas=tx_alternativas,
                    gabarito=str(gabarito),
                )
            )

        return Prova(itens=itens)
