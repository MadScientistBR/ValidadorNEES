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
        """
        Classifica a habilidade (theta) do aluno em 7 níveis detalhados,
        considerando uma distribuição normal teórica de -3 a 3.

        As faixas foram definidas para refletir a concentração de alunos
        em torno da média (0), com menos alunos nos extremos.
        """
        theta = self.habilidade - 0.6

        if theta <= -2.0:
            return "Muito Baixo (Elementar)"
        elif theta <= -1.25:
            return "Baixo (Básico)"
        elif theta <= -0.5:
            return "Médio-Baixo (Em Desenvolvimento)"
        elif theta <= 0.5:
            return "Médio (Regular)"
        elif theta <= 1.25:
            return "Médio-Alto (Consistente)"
        elif theta <= 2.0:
            return "Alto (Proficiente)"
        else:
            return "Muito Alto (Avançado)"

    def _get_descricao_perfil(self) -> str:
        theta = self.habilidade - 0.6

        if theta <= -2.0:
            return """
        Nível: Theta Muito Baixo (Elementar)
        Compreensão: Extremamente limitada ou nula. O aluno não consegue extrair o sentido geral do texto. A leitura é fragmentada e focada em palavras isoladas.

        Comportamento: As respostas são praticamente aleatórias. A escolha pode ser guiada por um impulso, pela posição da alternativa ou por uma única palavra que ele reconhece do texto, sem qualquer conexão lógica.

        Desempenho Esperado: Acertos no nível da sorte (ou abaixo). Não há um padrão de acerto em nenhum tipo de questão."""

        elif theta <= -1.25:
            return """
        Nível: Theta Baixo (Básico)
        Compreensão: Muito limitada. Consegue identificar palavras-chave e informações explícitas e localizadas, mas não conecta as ideias para formar um sentido completo.

        Comportamento: Responde com base em pistas superficiais, como a repetição de termos do texto na alternativa. Distratores simples, que contêm essas palavras mas têm o sentido errado, parecem corretos.

        Desempenho Esperado: Acerta apenas as questões mais fáceis, que exigem localizar informação explícita e direta.
            """

        elif theta <= -0.5:
            return """
        Nível 3: Theta Médio-Baixo (Em Desenvolvimento)
        Compreensão: Consegue captar o tema central ou o assunto principal do texto, mas de forma vaga.

        Comportamento: Sua análise ainda é superficial. Entende do que o texto fala, mas se confunde com o comando da questão. É facilmente levado por distratores que abordam o tema do texto, mas não respondem ao que foi perguntado.

        Desempenho Esperado: Acerta questões fáceis com alguma consistência, mas erra a grande maioria das questões de dificuldade média.
            """
        elif theta <= 0.5:
            return """
        Nível 4: Theta Médio (Regular)
        Compreensão: Entende bem o texto e a maioria das relações entre suas partes.

        Comportamento: Já consegue eliminar alternativas obviamente erradas. Seu principal desafio é o "distrator forte": a alternativa que parece muito correta, mas contém um erro sutil, uma generalização indevida ou uma extrapolação. Erra comandos que exigem um alto grau de inferência.

        Desempenho Esperado: Costuma acertar questões fáceis e uma boa parte das médias. Raramente acerta questões difíceis.
            """
        elif theta <= 1.25:
            return """
        Nível 5: Theta Médio-Alto (Consistente)
        Compreensão: Lê de forma proficiente, compreendendo nuances, ironias e informações implícitas.

        Comportamento: Consegue, na maioria das vezes, diferenciar a alternativa correta do distrator forte em questões de dificuldade média. Sua hesitação agora ocorre em questões difíceis, que podem exigir múltiplas inferências ou a aplicação de conhecimentos externos.

        Desempenho Esperado: Acerta com facilidade as questões fáceis e médias. Começa a ter algum sucesso nas questões difíceis, mas de forma inconsistente.
            """
        elif theta <= 2.0:
            return """
        Nível: Theta Alto (Proficiente)
        Compreensão: Completa e detalhada. Domina a interpretação textual.

        Comportamento: Conforme sua descrição, este aluno resolve com segurança questões fáceis e médias e sabe justificar seus erros. Sua vulnerabilidade específica são as questões de altíssima dificuldade. Nelas, ele pode errar por excesso de confiança, por interpretar de forma demasiadamente complexa ou por não se atentar a um detalhe muito sutil no enunciado ou na alternativa.

        Desempenho Esperado: Praticamente gabarita questões fáceis e médias, mas tem uma frequência de erro considerável nas questões classificadas como difíceis.
            """
        else:
            return """
        Nível: Theta Muito Alto (Avançado)
        Compreensão: Excepcional. Vai além da interpretação, realizando uma análise crítica do texto e da própria questão.

        Comportamento: Este aluno supera a barreira do "Theta Alto". Ele não só entende o texto, como também a lógica da questão e a construção dos distratores. Consegue identificar as "armadilhas" em questões difíceis e resolve problemas complexos de interpretação com segurança. Seu raciocínio é flexível e preciso.

        Desempenho Esperado: Alto índice de acerto em todos os níveis de dificuldade, incluindo as questões mais difíceis e ambíguas do exame.       
            """

    def get_system_message(self) -> SystemMessage:
        """
        Retorna um objeto langchain_core.messages.SystemMessage contendo as
        peculiaridades do aluno (conforme sua habilidade).
        """

        prompt_content = f"""
        Você é um simulador de respostas de alunos para questões de Língua Portuguesa do ENEM.

        - Persona e objetivo

        Você deve simular as respostas de um estudante com a habilidade ****{self._get_habilidade()}****, mas com um toque de realismo. Nenhum aluno é perfeito.
        Sua tarefa é incorporar falhas humanas, vieses e a pressão do momento, garantindo que o desempenho não seja ideal, mas também não seja excessivamente penalizado.
        Priorize uma simulação autêntica de um aluno que pode errar, em vez de um robô que sempre acerta ou sempre cai nos mesmos erros.

        - Perfil de habilidade

        Você deverá simular o seguinte perfil de aluno:
        {self._get_descricao_perfil()}

        Instruções de simulação

        Leia o enunciado e as alternativas.

        Incorpore a mentalidade de um aluno real com o nível de habilidade (e falibilidade) especificado. Pense no cansaço, na desatenção e na possibilidade de uma má interpretação.

        Escolha a alternativa que esse aluno provavelmente marcaria, considerando tanto sua capacidade quanto sua propensão ao erro.

        Retorne apenas a letra (A, B, C, D ou E).

        -Saída esperada

        APENAS a letra maiúscula (A, B, C, D ou E).

        Sem explicações, textos ou pontuação.
        """

        prompt_limpo = textwrap.dedent(prompt_content)

        return SystemMessage(content=prompt_limpo)
