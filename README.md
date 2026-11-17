# Sistema Inteligente de Triagem e Priorização Automatizada de Atendimentos em Organizações Comunitárias

**Projeto de Extensão Universitária em Sistemas Inteligentes**  
**Instituição:** UNINOVE — Centro Universitário Nove de Julho  
**Curso:** Ciência da Computação / Sistemas de Informação  
**Aluno:** Gabriel Neves Moura  
**RA:** 2225102570  

---

## 📌 Visão Geral do Projeto

Este software implementa uma solução baseada em **Processamento de Linguagem Natural (PLN)** e **Machine Learning Supervisionado (Multinomial Naive Bayes + TF-IDF)** para automatizar a recepção, classificação temática e priorização por grau de urgência de solicitações de auxílio recebidas por ONGs e centros comunitários.

### Categorias Suportadas (RN01 / RF03):
- **Alimentação** (Cestas básicas, refeições emergenciais, leite infantil, etc.)
- **Vestuário** (Agasalhos, roupas infantis, cobertores, calçados)
- **Apoio Jurídico** (Medidas protetivas, regularização civil, orientações trabalhistas)
- **Saúde/Orientação** (Medicamentos, apoio psicológico, fraldas geriátricas, primeiros socorros)

### Níveis de Prioridade (RN02 / RF04):
- 🔴 **Alta:** Risco iminente à vida, integridade física, desabastecimento severo imediato.
- 🟡 **Média:** Necessidades substanciais com prazo de dias/semanas.
- 🟢 **Baixa:** Dúvidas cadastrais, informações de oficinas, agendamentos rotineiros.

---

## 🏗️ Arquitetura e Tecnologias

- **Linguagem:** Python 3.10+
- **Machine Learning & PLN:** `scikit-learn` (TF-IDF Vectorizer + MultinomialNB)
- **Interface Web:** `Streamlit`
- **Banco de Dados:** `SQLite` (persistência relacional local sem dependência de servidores)
- **Engenharia de Dados:** `pandas`
- **Conformidade LGPD:** Máscaras por expressões regulares para CPF, telefone e e-mail.

---

## 🚀 Como Executar o Projeto

### 1. Criar e Ativar o Ambiente Virtual

```bash
# No terminal (PowerShell / Prompt de Comando)
cd "C:\Users\Gabriel Neves Moura\.gemini\antigravity\scratch\projeto-triagem-social"

python -m venv venv
.\venv\Scripts\activate
```

### 2. Instalar as Dependências

```bash
pip install -r requirements.txt
```

### 3. Rodar a Bateria de Testes Automatizados

```bash
python -m unittest discover tests
```

### 4. Iniciar a Aplicação Web Interativa

```bash
streamlit run app.py
```

A aplicação abrirá automaticamente no seu navegador no endereço `http://localhost:8501`.

---

## 🧪 Estrutura de Arquivos

```
projeto-triagem-social/
│
├── data/
│   └── dataset_sintetico.csv       # Base sintética balanceada de mensagens comunitárias
│
├── src/
│   ├── preprocessing.py            # Pré-processamento, Stopwords e Anonimização LGPD
│   ├── models.py                   # Entidade Solicitacao (POO)
│   ├── classifier.py               # Pipeline TF-IDF + Multinomial Naive Bayes
│   └── repository.py               # Persistência SQLite e exportações CSV/JSON
│
├── tests/
│   └── test_triage.py              # Suíte de testes unitários, integração e desempenho
│
├── app.py                          # Interface Web completa em Streamlit
├── requirements.txt                # Dependências do projeto
└── README.md                       # Documentação técnica do sistema
```
