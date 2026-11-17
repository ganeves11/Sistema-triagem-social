try:
    from src.preprocessing import TextPreprocessor
except ImportError:
    from preprocessing import TextPreprocessor

import time
import os
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TriageClassifier:
    """
    Componente de Inteligência Artificial para classificação temática e atribuição
    de prioridade/urgência supervisionada utilizando TF-IDF e Multinomial Naive Bayes.
    Atende aos requisitos RF02, RF03, RF04, RNF01 e RN04.
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),     # Unigramas e bigramas para captar contexto como 'falta de ar'
            sublinear_tf=True,      # Atenua o peso de palavras excessivamente frequentes
            min_df=1
        )
        self.clf_categoria = MultinomialNB(alpha=0.5)
        self.clf_urgencia = MultinomialNB(alpha=0.5)
        self.is_trained = False
        self.metrics: Dict[str, Any] = {}

    def train(self, dataset_path: str, test_size: float = 0.25, random_state: int = 42) -> Dict[str, Any]:
        """
        Carrega o dataset sintético, realiza pré-processamento, vetorização TF-IDF
        e treina os classificadores supervisionados para Categoria e Urgência.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset não encontrado no caminho: {dataset_path}")

        df = pd.read_csv(dataset_path)

        # Pré-processamento das mensagens de treino
        df['texto_processado'] = df['texto'].apply(self.preprocessor.clean_text)

        X = df['texto_processado']
        y_cat = df['categoria']
        y_urg = df['urgencia']

        # Divisão em treino e teste estratificada para avaliação científica
        X_train, X_test, y_cat_train, y_cat_test, y_urg_train, y_urg_test = train_test_split(
            X, y_cat, y_urg, test_size=test_size, random_state=random_state, stratify=y_cat
        )

        # Vetorização TF-IDF ajustada nos dados de treino
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Treinamento dos modelos
        self.clf_categoria.fit(X_train_vec, y_cat_train)
        self.clf_urgencia.fit(X_train_vec, y_urg_train)
        self.is_trained = True

        # Predições de teste para cálculo de métricas
        y_cat_pred = self.clf_categoria.predict(X_test_vec)
        y_urg_pred = self.clf_urgencia.predict(X_test_vec)

        # Consolidação das métricas de desempenho
        self.metrics = {
            "total_amostras": len(df),
            "amostras_treino": len(X_train),
            "amostras_teste": len(X_test),
            "categoria": {
                "acuracia": float(accuracy_score(y_cat_test, y_cat_pred)),
                "relatorio": classification_report(y_cat_test, y_cat_pred, output_dict=True, zero_division=0),
                "classes": list(self.clf_categoria.classes_),
                "matriz_confusao": confusion_matrix(y_cat_test, y_cat_pred, labels=self.clf_categoria.classes_).tolist()
            },
            "urgencia": {
                "acuracia": float(accuracy_score(y_urg_test, y_urg_pred)),
                "relatorio": classification_report(y_urg_test, y_urg_pred, output_dict=True, zero_division=0),
                "classes": list(self.clf_urgencia.classes_),
                "matriz_confusao": confusion_matrix(y_urg_test, y_urg_pred, labels=self.clf_urgencia.classes_).tolist()
            }
        }

        return self.metrics

    def predict(self, raw_text: str) -> Dict[str, Any]:
        """
        Classifica uma nova solicitação e estima a urgência com medição de tempo de resposta.
        Garante tratamento para textos vazios (RNF04) e mede o tempo de inferência (RNF01).
        """
        start_time = time.perf_counter()

        # Validação de robustez (RNF04)
        if not raw_text or not raw_text.strip():
            return {
                "sucesso": False,
                "erro": "Mensagem vazia ou inválida. Por favor, descreva a necessidade.",
                "tempo_resposta_ms": 0.0
            }

        if not self.is_trained:
            raise RuntimeError("O modelo precisa ser treinado antes de realizar predições.")

        # Pipeline de PLN e Anonimização
        texto_anon, texto_limpo = self.preprocessor.prepare(raw_text)

        if not texto_limpo.strip():
            # Fallback caso a limpeza remova todos os caracteres (apenas stopwords/símbolos)
            texto_limpo = raw_text.lower().strip()

        # Vetorização e inferência
        vec = self.vectorizer.transform([texto_limpo])

        cat_pred = self.clf_categoria.predict(vec)[0]
        urg_pred = self.clf_urgencia.predict(vec)[0]

        # Probabilidades/Confiança estatística
        cat_probs = self.clf_categoria.predict_proba(vec)[0]
        urg_probs = self.clf_urgencia.predict_proba(vec)[0]

        cat_idx = list(self.clf_categoria.classes_).index(cat_pred)
        urg_idx = list(self.clf_urgencia.classes_).index(urg_pred)

        conf_cat = float(cat_probs[cat_idx])
        conf_urg = float(urg_probs[urg_idx])

        elapsed_seconds = time.perf_counter() - start_time

        return {
            "sucesso": True,
            "texto_original": raw_text,
            "texto_anonimizado": texto_anon,
            "texto_limpo": texto_limpo,
            "categoria": cat_pred,
            "confianca_categoria": conf_cat,
            "urgencia": urg_pred,
            "confianca_urgencia": conf_urg,
            "tempo_resposta_segundos": elapsed_seconds,
            "tempo_resposta_ms": elapsed_seconds * 1000.0,
            "atende_rnf01": elapsed_seconds < 2.0  # RNF01: resposta em menos de 2s
        }
