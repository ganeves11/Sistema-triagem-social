import unittest
import os
import tempfile
import json
import time

try:
    from src.preprocessing import TextPreprocessor
    from src.classifier import TriageClassifier
    from src.repository import TriageRepository
    from src.models import Solicitacao
except ImportError:
    from preprocessing import TextPreprocessor
    from classifier import TriageClassifier
    from repository import TriageRepository
    from models import Solicitacao

class TestTriageSystem(unittest.TestCase):
    """
    Suíte completa de testes automatizados para verificação e validação do sistema
    conforme o plano de testes do Capítulo 05 e requisitos RF01-RF06 e RNF01-RNF05.
    """

    @classmethod
    def setUpClass(cls):
        cls.preprocessor = TextPreprocessor()
        # Busca dataset em data/ ou na pasta raiz
        cls.dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_sintetico.csv")
        if not os.path.exists(cls.dataset_path):
            cls.dataset_path = os.path.join(os.path.dirname(__file__), "data", "dataset_sintetico.csv")
        if not os.path.exists(cls.dataset_path):
            cls.dataset_path = os.path.join(os.path.dirname(__file__), "dataset_sintetico.csv")

        cls.classifier = TriageClassifier()
        if os.path.exists(cls.dataset_path):
            cls.classifier.train(cls.dataset_path)

    def setUp(self):
        # Cria banco SQLite temporário isolado para cada teste
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.repo = TriageRepository(self.temp_db.name)

    def tearDown(self):
        if os.path.exists(self.temp_db.name):
            try:
                os.remove(self.temp_db.name)
            except PermissionError:
                pass

    # =========================================================================
    # 1. TESTES UNITÁRIOS: PRÉ-PROCESSAMENTO E LGPD (RF02, RNF05, RN04)
    # =========================================================================
    def test_text_cleaning_removes_stopwords_and_accents(self):
        raw = "Nós estamos precisando com muita urgência de alimentação e água!"
        cleaned = self.preprocessor.clean_text(raw)
        # 'nós', 'estamos', 'com', 'muita', 'de', 'e' são stopwords
        self.assertNotIn("estamos", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertIn("precisando", cleaned)
        self.assertIn("urgencia", cleaned)
        self.assertIn("alimentacao", cleaned)

    def test_lgpd_anonymization_masks_cpf_phone_email(self):
        raw = "Meu nome é João, CPF 123.456.789-00, tel 11 98888-7777 e email joao@teste.com"
        anon = self.preprocessor.anonymize_lgpd(raw)
        self.assertNotIn("123.456.789-00", anon)
        self.assertNotIn("98888-7777", anon)
        self.assertNotIn("joao@teste.com", anon)
        self.assertIn("[CPF_ANONIMIZADO]", anon)
        self.assertIn("[TELEFONE_ANONIMIZADO]", anon)
        self.assertIn("[EMAIL_ANONIMIZADO]", anon)

    # =========================================================================
    # 2. TESTES UNITÁRIOS: CLASSIFICAÇÃO MACHINE LEARNING (RF03, RF04, RN01, RN02)
    # =========================================================================
    def test_classifier_training_metrics_exist(self):
        if not self.classifier.is_trained:
            self.skipTest("Dataset sintético não encontrado na pasta atual")
        metrics = self.classifier.metrics
        self.assertIn("categoria", metrics)
        self.assertIn("urgencia", metrics)
        self.assertGreater(metrics["categoria"]["acuracia"], 0.70)
        self.assertGreater(metrics["urgencia"]["acuracia"], 0.60)

    def test_classifier_predicts_valid_category_and_urgency(self):
        if not self.classifier.is_trained:
            self.skipTest("Dataset sintético não encontrado na pasta atual")
        text = "Estamos sem comida nenhuma em casa e as crianças estão chorando de fome socorro"
        res = self.classifier.predict(text)
        self.assertTrue(res["sucesso"])
        self.assertIn(res["categoria"], ["Alimentação", "Vestuário", "Apoio Jurídico", "Saúde/Orientação"])
        self.assertIn(res["urgencia"], ["Baixa", "Média", "Alta"])
        self.assertEqual(res["categoria"], "Alimentação")
        self.assertEqual(res["urgencia"], "Alta")

    # =========================================================================
    # 3. TESTES NÃO-FUNCIONAIS: DESEMPENHO E CONFIABILIDADE (RNF01, RNF04)
    # =========================================================================
    def test_rnf01_performance_under_two_seconds(self):
        """Valida que o tempo de resposta é inferior a 2 segundos (RNF01)."""
        if not self.classifier.is_trained:
            self.skipTest("Dataset sintético não encontrado na pasta atual")
        text = "Preciso de auxílio jurídico urgente para medida protetiva contra ameaça."
        res = self.classifier.predict(text)
        self.assertTrue(res["atende_rnf01"])
        self.assertLess(res["tempo_resposta_segundos"], 2.0)
        self.assertLess(res["tempo_resposta_segundos"], 0.5)

    def test_rnf04_reliability_handles_empty_or_whitespace_input(self):
        """Valida tratamento gracioso de inputs vazios sem interrupção (RNF04)."""
        res_empty = self.classifier.predict("")
        self.assertFalse(res_empty["sucesso"])
        self.assertIn("vazia", res_empty["erro"].lower())

        res_spaces = self.classifier.predict("     ")
        self.assertFalse(res_spaces["sucesso"])

    # =========================================================================
    # 4. TESTES DE INTEGRAÇÃO: BANCO DE DADOS E EXPORTAÇÃO (RF05, RF06)
    # =========================================================================
    def test_repository_save_and_order_by_urgency(self):
        """Valida persistência e ordenação priorizada (Alta primeiro) (RF05)."""
        req_baixa = Solicitacao(
            id=None, nome_morador="Ana", contato="11999990000",
            mensagem_original="Dúvida sobre palestra", mensagem_anonimizada="Dúvida...",
            categoria_predita="Saúde/Orientação", urgencia_predita="Baixa",
            confianca_categoria=0.9, confianca_urgencia=0.8
        )
        req_alta = Solicitacao(
            id=None, nome_morador="Carlos", contato="11988880000",
            mensagem_original="Emergência alimentar", mensagem_anonimizada="Emergência...",
            categoria_predita="Alimentação", urgencia_predita="Alta",
            confianca_categoria=0.95, confianca_urgencia=0.9
        )
        self.repo.save(req_baixa)
        self.repo.save(req_alta)

        pedidos = self.repo.list_all(order_by_urgency=True)
        self.assertEqual(len(pedidos), 2)
        self.assertEqual(pedidos[0].urgencia_predita, "Alta")
        self.assertEqual(pedidos[1].urgencia_predita, "Baixa")

    def test_repository_export_csv_and_json(self):
        """Valida que a exportação gera arquivos íntegros com registros (RF06)."""
        req = Solicitacao(
            id=None, nome_morador="Maria", contato="11977776666",
            mensagem_original="Preciso de cobertor", mensagem_anonimizada="Preciso de...",
            categoria_predita="Vestuário", urgencia_predita="Média",
            confianca_categoria=0.85, confianca_urgencia=0.75
        )
        self.repo.save(req)

        csv_content = self.repo.export_csv()
        json_content = self.repo.export_json()

        self.assertIn("Vestuário", csv_content)
        self.assertIn("Maria", csv_content)

        parsed_json = json.loads(json_content)
        self.assertEqual(len(parsed_json), 1)
        self.assertEqual(parsed_json[0]["categoria_predita"], "Vestuário")

if __name__ == "__main__":
    unittest.main()
