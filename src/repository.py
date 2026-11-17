import sqlite3
import json
import pandas as pd
from typing import List, Optional, Dict, Any

try:
    from src.models import Solicitacao
except ImportError:
    from models import Solicitacao

class TriageRepository:
    """
    Camada de persistência de dados utilizando SQLite (POO).
    Gerencia o histórico de solicitações triadas e implementa filtros e exportação (RF05, RF06).
    """

    def __init__(self, db_path: str = "triage.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Inicializa a tabela de solicitações com índices para rápida recuperação."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS solicitacoes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome_morador TEXT NOT NULL,
                    contato TEXT,
                    mensagem_original TEXT NOT NULL,
                    mensagem_anonimizada TEXT NOT NULL,
                    categoria_predita TEXT NOT NULL,
                    urgencia_predita TEXT NOT NULL,
                    confianca_categoria REAL,
                    confianca_urgencia REAL,
                    status TEXT DEFAULT 'Pendente',
                    data_hora TEXT NOT NULL
                )
            """)
            conn.commit()

    def save(self, item: Solicitacao) -> int:
        """Salva uma nova solicitação triada no banco."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO solicitacoes (
                    nome_morador, contato, mensagem_original, mensagem_anonimizada,
                    categoria_predita, urgencia_predita, confianca_categoria,
                    confianca_urgencia, status, data_hora
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.nome_morador,
                item.contato,
                item.mensagem_original,
                item.mensagem_anonimizada,
                item.categoria_predita,
                item.urgencia_predita,
                item.confianca_categoria,
                item.confianca_urgencia,
                item.status,
                item.data_hora
            ))
            conn.commit()
            return cursor.lastrowid

    def list_all(self, order_by_urgency: bool = True) -> List[Solicitacao]:
        """
        Retorna todas as solicitações. Se order_by_urgency=True,
        ordena com 'Alta' primeiro, seguido de 'Média' e 'Baixa' (RF05).
        """
        sql = """
            SELECT * FROM solicitacoes
            ORDER BY 
                CASE urgencia_predita
                    WHEN 'Alta' THEN 1
                    WHEN 'Média' THEN 2
                    WHEN 'Baixa' THEN 3
                    ELSE 4
                END ASC,
                id DESC
        """ if order_by_urgency else "SELECT * FROM solicitacoes ORDER BY id DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            return [self._row_to_entity(r) for r in rows]

    def filter_requests(
        self,
        categoria: Optional[str] = None,
        urgencia: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Solicitacao]:
        """Filtra as solicitações conforme critérios do gestor (RF05)."""
        query = "SELECT * FROM solicitacoes WHERE 1=1"
        params = []

        if categoria and categoria != "Todas":
            query += " AND categoria_predita = ?"
            params.append(categoria)
        if urgencia and urgencia != "Todas":
            query += " AND urgencia_predita = ?"
            params.append(urgencia)
        if status and status != "Todos":
            query += " AND status = ?"
            params.append(status)

        query += """
            ORDER BY 
                CASE urgencia_predita
                    WHEN 'Alta' THEN 1
                    WHEN 'Média' THEN 2
                    WHEN 'Baixa' THEN 3
                    ELSE 4
                END ASC,
                id DESC
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_entity(r) for r in rows]

    def update_status(self, solicitacao_id: int, new_status: str):
        """Atualiza o status de atendimento de uma requisição."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE solicitacoes SET status = ? WHERE id = ?",
                (new_status, solicitacao_id)
            )
            conn.commit()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Retorna indicadores quantitativos para o dashboard."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM solicitacoes")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM solicitacoes WHERE urgencia_predita = 'Alta'")
            total_alta = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM solicitacoes WHERE status = 'Pendente'")
            total_pendentes = cursor.fetchone()[0]

            cursor.execute("SELECT categoria_predita, COUNT(*) FROM solicitacoes GROUP BY categoria_predita")
            por_categoria = dict(cursor.fetchall())

            return {
                "total": total,
                "urgencia_alta": total_alta,
                "pendentes": total_pendentes,
                "por_categoria": por_categoria
            }

    def export_csv(self) -> str:
        """Exporta os dados em formato CSV (RF06)."""
        with self._get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM solicitacoes ORDER BY id DESC", conn)
            return df.to_csv(index=False)

    def export_json(self) -> str:
        """Exporta os dados em formato JSON (RF06)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM solicitacoes ORDER BY id DESC")
            rows = [dict(r) for r in cursor.fetchall()]
            return json.dumps(rows, ensure_ascii=False, indent=2)

    def _row_to_entity(self, row: sqlite3.Row) -> Solicitacao:
        return Solicitacao(
            id=row["id"],
            nome_morador=row["nome_morador"],
            contato=row["contato"],
            mensagem_original=row["mensagem_original"],
            mensagem_anonimizada=row["mensagem_anonimizada"],
            categoria_predita=row["categoria_predita"],
            urgencia_predita=row["urgencia_predita"],
            confianca_categoria=row["confianca_categoria"],
            confianca_urgencia=row["confianca_urgencia"],
            status=row["status"],
            data_hora=row["data_hora"]
        )
