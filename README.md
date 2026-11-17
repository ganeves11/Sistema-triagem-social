# Sistema - Triagem - Social

### Plataforma de Triagem e Priorização Automatizada de Demandas Comunitárias com Inteligência Artificial

Projeto de Extensão Universitária em Sistemas Inteligentes  
Centro Universitário Nove de Julho (UNINOVE)  
Curso: Bacharelado em Ciência da Computação / Sistemas de Informação  
Autor: Gabriel Neves Moura  

---

## 1. Descrição do Projeto

Este software consiste em uma solução computacional desenvolvida para automatizar o processo de triagem, categorização temática e estratificação de prioridade de solicitações assistenciais recebidas por organizações da sociedade civil e instituições comunitárias.

Em muitas entidades do terceiro setor, os pedidos de apoio chegam em formato de texto não estruturado (e-mails, formulários web e mensageria instantânea). A análise puramente manual dessas mensagens gera sobrecarga operacional e introduz o risco de solicitações emergenciais críticas ficarem represadas em filas de espera sem o devido destaque.

Para mitigar esse gargalo, a aplicação emprega técnicas de Processamento de Linguagem Natural (PLN) e Aprendizado de Máquina Supervisionado (classificador Multinomial Naive Bayes combinado com vetorização estatística TF-IDF), permitindo a predição imediata da categoria da demanda e seu grau de criticidade, fornecendo suporte técnico à decisão dos gestores sociais.

---

## 2. Categorias e Níveis de Prioridade

### 2.1 Categorias Temáticas
* **Alimentação:** Solicitações de cestas básicas, alimentos não perecíveis, refeições e fórmulas infantis.
* **Vestuário:** Demanda por agasalhos, cobertores, calçados e vestimentas de frio.
* **Apoio Jurídico:** Orientações legais, medidas protetivas emergenciais, questões de guarda e regularização civil.
* **Saúde/Orientação:** Encaminhamento médico, medicamentos de uso contínuo, fraldas geriátricas e apoio psicológico.

### 2.2 Estratificação de Prioridade
* **Prioridade Alta:** Situações de desabastecimento severo de alimentos, risco à integridade física, urgências infantis ou vulnerabilidade extrema com necessidade de resposta imediata.
* **Prioridade Média:** Necessidades materiais substanciais com prazo de atendimento estimado em dias.
* **Prioridade Baixa:** Dúvidas cadastrais, pedidos de informação de oficinas, cursos comunitários e agendamentos rotineiros.

---

## 3. Arquitetura e Tecnologias Utilizadas

A solução foi desenvolvida adotando o paradigma de Programação Orientada a Objetos (POO), garantindo separação clara de responsabilidades entre as camadas de interface, inteligência artificial e persistência de dados.

* **Linguagem Principal:** Python 3.10+
* **Processamento de Linguagem Natural e ML:** `scikit-learn` (TfidfVectorizer com unigramas e bigramas; classificador MultinomialNB)
* **Engenharia de Dados:** `pandas`
* **Interface Web:** `streamlit` (interface com estética de prontuário e regulação clínica)
* **Banco de Dados Relacional:** `SQLite` (persistência local embarcada sem necessidade de serviços externos)
* **Testes Automatizados:** `unittest` / `pytest`
* **Privacidade e LGPD:** Módulo de higienização com expressões regulares (`re`) para detecção e mascaramento de dados de identificação pessoal (CPFs, telefones e e-mails).

---

## 4. Estrutura do Repositório

```text
sistema-triagem-social/
│
├── data/
│   └── dataset_sintetico.csv       # Base de treinamento com instâncias balanceadas em português
│
├── docs/
│   └── documento_completo_uninove.html # Documentação técnica unificada e relatórios acadêmicos
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Higienização de texto, remoção de stopwords e anonimização LGPD
│   ├── models.py                   # Entidade de domínio Solicitacao (POO)
│   ├── classifier.py               # Pipeline de treino, inferência e métricas estatísticas (TF-IDF + Naive Bayes)
│   └── repository.py               # Camada de persistência relacional SQLite e exportação CSV/JSON
│
├── tests/
│   ├── __init__.py
│   └── test_triage.py              # Suíte de testes unitários, testes de integração e testes de sistema
│
├── .streamlit/
│   └── config.toml                 # Configurações de tema de alto contraste da interface
│
├── app.py                          # Aplicação web interativa (Streamlit)
├── requirements.txt                # Lista de bibliotecas e dependências do ambiente
├── .gitignore                      # Regras de exclusão de artefatos temporários do versionamento
└── README.md                       # Documentação técnica do projeto
```

---

## 5. Instruções de Instalação e Execução

### 5.1 Pré-requisitos
* Python 3.10 ou superior instalado no sistema operacional.
* Gerenciador de pacotes `pip` atualizado.

### 5.2 Passo a Passo

1. **Clonar o repositório:**
   ```bash
   git clone https://github.com/ganeves11/sistema-triagem-social.git
   cd sistema-triagem-social
   ```

2. **Criar e ativar o ambiente virtual (recomendado):**
   ```bash
   # No Windows (PowerShell):
   python -m venv venv
   .\venv\Scripts\activate

   # No Linux / macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Executar a aplicação web:**
   ```bash
   streamlit run app.py
   ```
   A interface será carregada automaticamente no navegador através do endereço `http://localhost:8501`.

---

## 6. Execução da Suíte de Testes Automatizados

O projeto conta com testes unitários, testes de integração e testes de sistema para validar a conformidade dos requisitos funcionais e não-funcionais (tempo de inferência inferior a 2,0 segundos, imunidade a SQL injection e robustez contra entradas vazias).

Para rodar todos os testes automatizados, execute no terminal:

```bash
python -m unittest discover tests
```

---

## 7. Segurança e Conformidade com a LGPD

O software incorpora salvaguardas alinhadas à Lei Geral de Proteção de Dados Pessoais (Lei nº 13.709/2018):
* **Minimização:** Coleta apenas dados estritamente necessários para o atendimento assistencial (nome, telefone e relato da necessidade).
* **Anonimização em Tempo de Execução:** O pipeline de pré-processamento substitui automaticamente dados sensíveis identificáveis por marcadores anônimos antes de submeter o texto à análise de inteligência artificial.
* **Operação Local:** Os dados são mantidos exclusivamente em banco de dados local (`triage.db`), sem transmissão de dados para servidores externos ou nuvens de terceiros.
* **Defesa contra Injeção de Código:** Consultas parametrizadas (*Prepared Statements*) são empregadas em todas as rotinas de banco de dados, neutralizando riscos de SQL Injection.

---

## 8. Autor e Identificação

* **Desenvolvedor:** Gabriel Neves Moura ([@ganeves11](https://github.com/ganeves11))
* **Instituição:** Centro Universitário Nove de Julho (UNINOVE)
* **Curso:** Bacharelado em Ciência da Computação / Sistemas de Informação
* **Disciplina:** Projeto de Extensão em Sistemas Inteligentes
