from typing import Final, Set

import pandas as pd
from girth import twopl_jml
from scipy.special import expit


class EstimadorTRI:
    """
    Classe responsável por receber um dataframe contendo as respostas dos alunos
    simulados e retornar um dataframe contendo os parâmetros TRI para um conjunto
    de questões fornecidas.
    """

    COLUNAS_ESPERADAS: Final[Set[str]] = {"respondente_id", "item_id", "acertou"}

    def __init__(self) -> None:
        pass

    @staticmethod
    def _verificar_esquema(df_simulado: pd.DataFrame) -> None:
        """
        Função que verifica se o Dataframe fornecido está no formato adequado.
        """
        colunas_df = set(df_simulado.columns)
        colunas_que_faltam = EstimadorTRI.COLUNAS_ESPERADAS - colunas_df

        if colunas_que_faltam:
            raise ValueError(
                f"""Erro: As seguintes colunas estão faltando no dataframe
                dos alunos simulados {colunas_que_faltam}"""
            )

    @staticmethod
    def estimar_parametros(df_simulado: pd.DataFrame) -> pd.DataFrame:
        """
        Função que recebe um dataframe com as respostas dos alunos simulados
        e retorna um dataframe com os parâmetros de discriminação (a) e
        dificuldade (b) e porcentagem de acerto (%).
        """
        EstimadorTRI._verificar_esquema(df_simulado)

        # se a tabela está no formato adequado
        df_pivotado = df_simulado.pivot(
            index="item_id", columns="respondente_id", values="acertou"
        )

        index_series = df_pivotado.index

        prob_acerto = pd.Series(df_pivotado.mean(axis=1))
        prob_acerto = EstimadorTRI.ajustar_probabilidade_sigmoidal(prob_acerto)

        tri_data = twopl_jml(dataset=df_pivotado.astype(int).to_numpy())

        tri_dataframe = pd.DataFrame(
            {
                "A": tri_data["Discrimination"],
                "B": tri_data["Difficulty"],
                # obs: o 0.20 é para corrigir um vies que observei na LLM. O ideal é fazer
                # um modelo (logístico ou algo do tipo) que relaciona a p llm (probabilidade
                # prevista pela LLM) com a p real.
                "PROB_ACERTO": prob_acerto,
                "ID_QUESTÃO": index_series,
            },
        )

        return tri_dataframe

    @staticmethod
    def ajustar_probabilidade_sigmoidal(
        prob_series: pd.Series, fator_contraste: float = 4.0
    ) -> pd.Series:
        prob_centralizada = prob_series - 0.50

        prob_ajustada = expit(prob_centralizada * fator_contraste)

        return prob_ajustada
