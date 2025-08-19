import time
from typing import List

import pandas as pd
from langchain_core.runnables import Runnable
from tqdm import tqdm

from ..core.prova import Prova
from ..core.respondente import Respondente


class Simulador:
    def __init__(self, responder_chain: Runnable):
        self.chain = responder_chain

    def executar(
        self,
        prova: Prova,
        populacao: List[Respondente],
        tamanho_lote: int = 45,
        delay_segundos: int = 2,
    ) -> pd.DataFrame:
        """
        Executa a simulação completa, processando em lotes controlados com delay.
        """
        lista_de_inputs = [
            {"respondente": respondente, "item": item}
            for respondente in populacao
            for item in prova.itens
        ]

        print(
            f"\nIniciando simulação com {len(lista_de_inputs)} respostas "
            f"(lotes de {tamanho_lote} com delay de {delay_segundos}s)..."
        )

        respostas_geradas_total = []
        total_de_inputs = len(lista_de_inputs)

        # MODIFICAÇÃO 3: Substituir a chamada única de .batch() por um loop com lotes
        # O range avança em passos do tamanho do lote
        # O tqdm cria uma barra de progresso visual
        for i in tqdm(
            range(0, total_de_inputs, tamanho_lote), desc="Processando lotes"
        ):
            # Pega o pedaço (lote) da lista de inputs
            lote_inputs = lista_de_inputs[i : i + tamanho_lote]

            # Executa o batch APENAS para o lote atual
            respostas_do_lote = self.chain.batch(
                lote_inputs, config={"max_concurrency": 45}
            )

            # Adiciona os resultados deste lote à lista total
            respostas_geradas_total.extend(respostas_do_lote)

            # Adiciona o delay, mas apenas se este não for o último lote
            if i + tamanho_lote < total_de_inputs:
                time.sleep(delay_segundos)

        # O restante do código permanece o mesmo, mas agora usa a lista preenchida em lotes
        resultados = []
        for i, resposta in enumerate(respostas_geradas_total):
            respondente = lista_de_inputs[i]["respondente"]
            item = lista_de_inputs[i]["item"]

            resultados.append(
                {
                    "respondente_id": respondente.id,
                    "habilidade_respondente": respondente.habilidade,
                    "item_id": item.id_item,
                    "resposta_gerada": resposta.strip().upper(),
                    "gabarito": item.gabarito.strip().upper(),
                    "acertou": (
                        1
                        if resposta.strip().upper() == item.gabarito.strip().upper()
                        else 0
                    ),
                }
            )

        return pd.DataFrame(resultados)
