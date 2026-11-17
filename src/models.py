from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

@dataclass
class Solicitacao:
    """
    Entidade de domínio que representa uma solicitação de atendimento social.
    Emprega Programação Orientada a Objetos para encapsulamento dos dados.
    """
    id: Optional[int]
    nome_morador: str
    contato: str
    mensagem_original: str
    mensagem_anonimizada: str
    categoria_predita: str
    urgencia_predita: str
    confianca_categoria: float
    confianca_urgencia: float
    status: str = "Pendente"  # Pendente, Em Atendimento, Concluído
    data_hora: str = ""

    def __post_init__(self):
        if not self.data_hora:
            self.data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        return asdict(self)
