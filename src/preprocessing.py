import re
import string
import unicodedata

class TextPreprocessor:
    """
    Classe responsável pelo pré-processamento de texto e higienização conforme LGPD (RN04 e RNF05).
    Aplica técnicas de Processamento de Linguagem Natural (PLN) para normalização,
    remoção de ruídos, tratamento de stopwords em português e anonimização de dados sensíveis.
    """

    # Lista consolidada de stopwords em português (independente de downloads externos)
    STOPWORDS_PT = {
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não",
        "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi",
        "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito",
        "há", "nos", "já", "está", "eu", "também", "só", "pelo", "pela", "até", "isso",
        "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem",
        "nas", "me", "esse", "eles", "estão", "você", "tinha", "foram", "essa", "num",
        "nem", "suas", "meu", "às", "minha", "têm", "numa", "pelos", "elas", "havia",
        "seja", "qual", "será", "nós", "tenho", "lhe", "deles", "essas", "esses", "pelas",
        "este", "fosse", "dele", "tu", "te", "vocês", "vos", "lhes", "meus", "minhas",
        "teu", "tua", "teus", "tuas", "nosso", "nossa", "nossos", "nossas", "dela", "delas",
        "esta", "estes", "estas", "aquele", "aquela", "aqueles", "aquelas", "isto", "aquilo"
    }

    def __init__(self):
        # Padrões Regex para mascaramento LGPD
        self.regex_cpf = re.compile(r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b')
        self.regex_telefone = re.compile(r'(\(?\d{2}\)?\s?)?(\d{4,5}-?\d{4})')
        self.regex_email = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')

    def anonymize_lgpd(self, text: str) -> str:
        """
        Substitui informações de identificação pessoal por tokens anônimos (RNF05 e RN10).
        """
        if not text or not isinstance(text, str):
            return ""
        
        texto_anon = self.regex_cpf.sub("[CPF_ANONIMIZADO]", text)
        texto_anon = self.regex_email.sub("[EMAIL_ANONIMIZADO]", texto_anon)
        texto_anon = self.regex_telefone.sub("[TELEFONE_ANONIMIZADO]", texto_anon)
        return texto_anon

    def clean_text(self, text: str) -> str:
        """
        Pipeline de limpeza em PLN (RF02 e RN04):
        - Conversão para caixa baixa (minúsculas)
        - Remoção de pontuação e caracteres especiais
        - Remoção de stopwords em língua portuguesa
        """
        if not text or not isinstance(text, str):
            return ""

        # Conversão para minúsculas
        texto_limpo = text.lower()

        # Normalização unicode (remove acentos para padronização de busca léxica)
        texto_limpo = unicodedata.normalize('NFKD', texto_limpo)
        texto_limpo = "".join([c for c in texto_limpo if not unicodedata.combining(c)])

        # Remoção de pontuações
        for pontuacao in string.punctuation:
            texto_limpo = texto_limpo.replace(pontuacao, " ")

        # Remoção de números residuais e quebras de linha
        texto_limpo = re.sub(r'\d+', ' ', texto_limpo)
        texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()

        # Filtragem de stopwords
        palavras = texto_limpo.split()
        palavras_filtradas = [p for p in palavras if p not in self.STOPWORDS_PT and len(p) > 2]

        return " ".join(palavras_filtradas)

    def prepare(self, text: str) -> tuple[str, str]:
        """
        Executa a anonimização e depois a limpeza para o modelo.
        Retorna (texto_anonimizado, texto_processado_pln)
        """
        texto_anon = self.anonymize_lgpd(text)
        texto_limpo = self.clean_text(texto_anon)
        return texto_anon, texto_limpo
