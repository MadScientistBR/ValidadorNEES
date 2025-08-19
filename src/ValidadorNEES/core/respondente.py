import textwrap

from langchain_core.messages import SystemMessage


class Respondente:
    """
    Representa um respondente (aluno) em uma prova.

    Essa classe armazena o seu id (identificador único) e sua habilidade (-3 a 3) conforme
    a TRI.

    Attributes:
        id (int): Identificador único do aluno
        habilidade (float): Habilidade do aluno
    """

    def __init__(self, id: int, habilidade: float) -> None:
        self.id = id
        self.habilidade = habilidade

    def __repr__(self) -> str:
        """
        Retorna a representação oficial em string do objeto Respondente.
        """
        # Formata a string para ser informativa e parecer código Python
        return f"Respondente(id={self.id}, habilidade={self.habilidade:.2f})"

    def _get_habilidade(self) -> str:
        theta = (
            self.habilidade - 0.6
        )  # testando alguns offsets para ver se a simulação melhora

        if theta <= -0.8:
            return "muito baixo"
        elif theta <= -0.3:
            return "baixo"
        elif theta <= 1.0:
            return "médio"
        else:
            return "alto"

    def get_system_message(self) -> SystemMessage:
        """
        Retorna um objeto langchain_core.messages.SystemMessage contendo as
        peculiaridades do aluno (conforme sua habilidade).
        """

        prompt_content = f"""
        Você é um simulador de respostas de alunos para questões de Língua Portuguesa do ENEM.

        Persona e Objetivo

        Você deve simular as respostas de um estudante com a habilidade (theta) fornecida, mas com um toque de realismo. Nenhum aluno é perfeito. Sua tarefa é incorporar falhas humanas, vieses e a pressão do momento, garantindo que o desempenho não seja ideal, mas também não seja excessivamente penalizado. Priorize uma simulação autêntica de um aluno que pode errar, em vez de um robô que sempre acerta ou sempre cai nos mesmos erros.

        Perfis de habilidade (Ajustados)
        - *Theta Muito Baixo: compreensão extremamente limitada, respostas praticamente aleatórias.
        - *Theta Baixo: compreensão limitada, respostas baseadas em pistas superficiais, distrações simples parecem corretas.
        - *Theta Médio: entende o tema central, mas erra comandos complexos, distratores fortes confundem, análise superficial.
        - *Theta Alto: compreensão completa, escolhe a correta com segurança e reconhece erros das outras.

        Instruções de simulação

        Leia o enunciado e as alternativas.

        Incorpore a mentalidade de um aluno real com o nível de habilidade (e falibilidade) especificado. Pense no cansaço, na desatenção e na possibilidade de uma má interpretação.

        Escolha a alternativa que esse aluno provavelmente marcaria, considerando tanto sua capacidade quanto sua propensão ao erro.

        Retorne apenas a letra (A, B, C, D ou E).

        Entrada

        Habilidade (theta): {self._get_habilidade()}

        Saída esperada

        APENAS a letra maiúscula (A, B, C, D ou E).

        Sem explicações, textos ou pontuação.
        """

        prompt_limpo = textwrap.dedent(prompt_content)

        return SystemMessage(content=prompt_limpo)
