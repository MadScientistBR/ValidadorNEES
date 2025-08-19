import os
from typing import List, cast

import numpy as np
import pandas as pd

from ..core.respondente import Respondente


class GeradorRespondentes:
    """
    Classe responsável por ler um arquivo (.csv) contendo as habilidades de uma
    população e retorna uma lista de respondentes contendo a mesma distribuição.

    Attribute:
        caminho_habilidades (str): caminho contendo os dados das habilidades dos alunos
    """

    def __init__(self, caminho_habilidades: str) -> None:
        if not (os.path.isfile(caminho_habilidades)):
            raise ValueError("O Caminho fornecido para o arquivo não existe!")

        self.caminho_habilidades = caminho_habilidades

    # def _gerar_sample_habilidades(self, numero_de_alunos: int) -> pd.Series:
    #     """
    #     Retorna um pd.Series sample (mantendo a mesma distribuição
    #     do arquivo original) contendo a habilidade para um número
    #     de respondentes especificado.
    #     """
    #     df_habilidades = pd.read_csv(self.caminho_habilidades)
    #     df_habilidades = df_habilidades.dropna(subset=["HABILIDADE"])
    #
    #     resultado_sample = df_habilidades["HABILIDADE"].sample(n=numero_de_alunos)
    #
    #     return cast(pd.Series, resultado_sample)
    #

    def _gerar_sample_habilidades(self, numero_de_alunos: int) -> pd.Series:
        """
        Gera uma amostra de habilidades (theta) para um número especificado de alunos
        a partir de uma distribuição normal padrão (média=0, desvio padrão=1).
        Os valores são limitados ao intervalo [-3, 3].
        """
        habilidades_array = np.random.normal(loc=0, scale=1, size=numero_de_alunos)

        habilidades_array_filtrado = np.clip(habilidades_array, a_min=-3, a_max=3)

        habilidades_series = pd.Series(habilidades_array_filtrado, name="habilidades")

        return habilidades_series

    def gerar_respondentes(self, numero_de_alunos: int) -> List[Respondente]:
        """
        Cria uma lista de objetos Respondente mantendo a mesma distribuição
        de habilidades da base de dados original.
        """
        habilidades_sample = self._gerar_sample_habilidades(numero_de_alunos)

        alunos = [
            Respondente(id=indice, habilidade=valor_habilidade)
            for indice, valor_habilidade in enumerate(habilidades_sample)
        ]

        return alunos
