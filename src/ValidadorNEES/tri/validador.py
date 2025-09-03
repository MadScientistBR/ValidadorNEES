from enum import Enum
from typing import Any, Dict, Final, Literal, Set, TypeAlias

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

ParametroInteresse: TypeAlias = Literal[
    "A", "B", "PROB_ACERTO"
]  # parâmetros TRI de interesse
CorrelationMethod: TypeAlias = Literal["pearson", "spearman"]


class BinDificuldade(Enum):
    FACIL = "Fácil"
    MEDIA = "Média"
    DIFICIL = "Dificil"


class ValidadorTRI:
    """
    Uma classe para validar a recuperação de parâmetros de um modelo TRI
    comparando um conjunto de parâmetros reais com um conjunto simulado/estimado.

    Attributes:
        df_real: pd.DataFrame = DataFrame contendo os parâmetros reais das questões.
        Deve conter apenas as seguintes colunas [ID_QUESTÃO, A, B].

        df_simulado: pd.DataFrame = DataFrame contendo os parâmetros simulados das
        questões. Deve conter apenas as seguintes colunas [ID_QUESTÃO, A, B].
    """

    COLUNAS_ESPERADAS: Final[Set[str]] = {"A", "B", "ID_QUESTÃO", "PROB_ACERTO"}

    def __init__(
        self, df_parametros_reais: pd.DataFrame, df_parametros_simulados: pd.DataFrame
    ) -> None:
        """
        Inicializa um objeto ValidadorTRI.

        Args:
            df_parametros_reais: pd.DataFrame = Dataframe contendo os parâmetros (a, b) reais.

            df_parametros_simulados: pd.DataFrame = Dataframe contendo os parâmetros (a, b) simulados.
        """
        df_real = df_parametros_reais.copy(deep=True)
        df_simulado = df_parametros_simulados.copy(deep=True)

        self._verificar_esquema(df_real, df_simulado)
        self._alinhar_dataframes(df_parametros_reais, df_parametros_simulados)
        self._calcular_limites_dificuldade()

    def _verificar_esquema(
        self, df_real: pd.DataFrame, df_simulado: pd.DataFrame
    ) -> None:
        """
        Função que verifica se os DataFrames estão no formato solicitado pelo Validador.
        """
        dataframes = {"DataFrame Real": df_real, "DataFrame Simulado": df_simulado}

        lista_de_erros = []

        for nome, df in dataframes.items():
            colunas_df = set(df.columns)
            colunas_que_faltam = self.COLUNAS_ESPERADAS - colunas_df

            if colunas_que_faltam:
                lista_de_erros.append(
                    f"Erro no {nome}, as seguintes colunas estão faltando = {colunas_que_faltam}\n"
                )

        if lista_de_erros:
            mensagem_final = (
                "Foram encontrados os seguintes erros de esquema:\n"
                + "\n".join(lista_de_erros)
            )
            raise ValueError(mensagem_final)

    def _alinhar_dataframes(
        self, df_real: pd.DataFrame, df_simulado: pd.DataFrame
    ) -> None:
        """
        Função que confirma que ambos os DataFrames se referem ao mesmo conjunto de questões.
        Para tal análise, vamos utilizar o ID_QUESTÃO.
        """
        df_real = df_real.set_index("ID_QUESTÃO")
        df_real = df_real.sort_index()

        df_simulado = df_simulado.set_index("ID_QUESTÃO")
        df_simulado = df_simulado.sort_index()

        if not df_real.index.equals(df_simulado.index):
            raise ValueError(
                "ERRO ao criar a classe: "
                "Os DataFrames não representam as mesmas questões!"
            )

        self.df_real = df_real
        self.df_simulado = df_simulado

    def _calcular_correlacao(
        self,
        method: CorrelationMethod,
        parametro: ParametroInteresse,
    ) -> float:
        """
        Método base que prepara, valida e calcula a correlação entre duas séries
        de parâmetros da TRI (e.g., discriminação, dificuldade).
        """

        # obs: cast "forçado" para evitar erro do PyRight
        parametros_reais = pd.Series(self.df_real[parametro])
        parametros_simulados = pd.Series(self.df_simulado[parametro])

        correlation = parametros_reais.corr(parametros_simulados, method=method)
        return float(correlation)

    def _calcular_bias(self, parametro: ParametroInteresse) -> float:
        """
        Calcula o BIAS (Erro médio) entre as questões.
        """
        diferenca = self.df_simulado[parametro] - self.df_real[parametro]
        return float(np.mean(diferenca))

    def _calcular_rmse(self, parametro: ParametroInteresse) -> float:
        """
        Calcula o RMSE (Raiz do erro quadrático médio) entre os parâmetros.
        """
        diferenca_quadratica = (
            self.df_simulado[parametro] - self.df_real[parametro]
        ) ** 2
        return float(np.sqrt(np.mean(diferenca_quadratica)))

    def _calcular_limites_dificuldade(self) -> None:
        """Calcula os pontos de corte de dificuldade com base nos quantis dos dados REAIS."""
        quantis = [0, 0.30, 0.75, 1.0]

        # Calcula os valores da 'probabilidade de acerto' que correspondem a esses quantis
        limites = self.df_real["PROB_ACERTO"].quantile(quantis).tolist()

        # Garante que os limites sejam únicos para evitar erros no pd.cut
        limites_unicos = sorted(list(set(limites)))
        limites_unicos[0] = -np.inf  # O limite inferior deve ser infinito
        limites_unicos[-1] = np.inf  # O limite superior deve ser infinito

        self.limites_dificuldades = limites_unicos

    def _categorizar_dificuldade(self, series_prob: pd.Series) -> pd.Series:
        """
        Categoriza as questões em fáceis, médias e difíceis com base no parâmetro
        B. No caso, analisa-se os quantis da seguinte forma:

        Questões fáceis: 25% (Ponto de corte: 1° quartil)
        Questões médias: 50% (1° a 3° quartil)
        Questões difíceis: 25% (3° quartil ao final)
        """
        labels = [
            BinDificuldade.FACIL.value,
            BinDificuldade.MEDIA.value,
            BinDificuldade.DIFICIL.value,
        ]

        num_categorias = len(self.limites_dificuldades) - 1

        categorias = pd.cut(
            x=series_prob,
            bins=self.limites_dificuldades,
            labels=labels[:num_categorias],
            include_lowest=True,
        )

        return pd.Series(categorias)

    def obter_dados_continuos_para_relatorio(self) -> Dict[str, Any]:
        """
        Retorna os dados necessários para criação de um relatório que analisa
        a qualidade da simulação.
        """

        df_comp = self.df_real.join(
            self.df_simulado, lsuffix="_real", rsuffix="_simulado"
        )

        dados = {
            # "metricas": {
            #     "a_correlacao_spearman": self._calcular_correlacao("spearman", "A"),
            #     "a_correlacao_pearson": self._calcular_correlacao("pearson", "A"),
            #     "a_bias": self._calcular_bias("A"),
            #     "a_rmse": self._calcular_rmse("A"),
            #     "b_correlacao_spearman": self._calcular_correlacao("spearman", "B"),
            #     "b_correlacao_pearson": self._calcular_correlacao("pearson", "B"),
            #     "b_bias": self._calcular_bias("B"),
            #     "b_rmse": self._calcular_rmse("B"),
            # },
            # "dados_brutos": {
            #     "a_real": self.df_real["A"],
            #     "a_simulado": self.df_simulado["A"],
            #     "b_real": self.df_real["B"],
            #     "b_simulado": self.df_simulado["B"],
            # },
            "metricas": {
                "prob_de_acerto_spearman": self._calcular_correlacao(
                    "spearman", "PROB_ACERTO"
                ),
                "prob_de_acerto_pearson": self._calcular_correlacao(
                    "pearson", "PROB_ACERTO"
                ),
                "prob_de_acerto_bias": self._calcular_bias("PROB_ACERTO"),
                "prob_de_acerto_rmse": self._calcular_rmse("PROB_ACERTO"),
            },
            "dados_brutos": {
                "prob_de_acerto_real": self.df_real["PROB_ACERTO"],
                "prob_de_acerto_simulada": self.df_simulado["PROB_ACERTO"],
            },
            "df_comparativo": df_comp,
        }

        return dados

    def obter_dados_discretos_para_relatorio(self) -> Dict[str, Any]:
        """Retorna as métricas e dados para a análise discreta da dificuldade."""
        prob_de_acerto_real_cat = self._categorizar_dificuldade(
            pd.Series(self.df_real["PROB_ACERTO"])
        )
        prob_acerto_real_cat = self._categorizar_dificuldade(
            pd.Series(self.df_simulado["PROB_ACERTO"])
        )

        labels = [
            BinDificuldade.FACIL.value,
            BinDificuldade.MEDIA.value,
            BinDificuldade.DIFICIL.value,
        ]

        # Calcular a matriz de confusão e a acurácia
        matriz = confusion_matrix(
            prob_de_acerto_real_cat, prob_acerto_real_cat, labels=labels
        )
        acuracia = accuracy_score(prob_de_acerto_real_cat, prob_acerto_real_cat)

        return {
            "metricas": {"acuracia": acuracia},
            "matriz_confusao": matriz,
            "categorias": {
                "real": prob_de_acerto_real_cat,
                "simulado": prob_acerto_real_cat,
            },
            "labels": labels,
        }
