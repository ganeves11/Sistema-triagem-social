import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime

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

# Configuração da página estilo Portal Clínico
st.set_page_config(
    page_title="Portal de Triagem Clínica e Social",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS de alto contraste e legibilidade clínica total
st.markdown("""
<style>
    /* ========================================================= */
    /* 1. ÁREA PRINCIPAL (Fundo claro, textos escuros de alto contraste) */
    /* ========================================================= */
    .stApp {
        background-color: #f8fafc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    section.main h1, section.main h2, section.main h3, section.main h4, section.main h5, section.main h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px !important;
    }
    
    section.main p, 
    section.main div[data-testid="stMarkdownContainer"] > p,
    section.main span {
        color: #1e293b !important;
        font-size: 0.95rem !important;
    }
    
    section.main .stCaption, 
    section.main [data-testid="stCaptionContainer"] p {
        color: #475569 !important;
        font-size: 0.85rem !important;
    }

    /* Rótulos dos Campos de Entrada (Labels) */
    section.main .stTextInput label p, 
    section.main .stTextArea label p, 
    section.main .stSelectbox label p {
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Caixas de Entrada (Inputs e TextAreas) */
    section.main .stTextInput input, 
    section.main .stTextArea textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        font-size: 0.9rem !important;
    }
    section.main .stTextInput input:focus, 
    section.main .stTextArea textarea:focus {
        border-color: #0d9488 !important;
        box-shadow: 0 0 0 2px rgba(13, 148, 136, 0.2) !important;
    }
    section.main .stTextInput input::placeholder, 
    section.main .stTextArea textarea::placeholder {
        color: #94a3b8 !important;
    }

    /* Abas do Streamlit - Visibilidade e Contraste */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        padding: 10px 16px !important;
    }
    button[data-baseweb="tab"] div[data-testid="stMarkdownContainer"] p {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #0d9488 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] div[data-testid="stMarkdownContainer"] p {
        color: #0f766e !important;
        font-weight: 700 !important;
    }

    /* Barra Superior do Portal Clínico */
    .clinical-header {
        background: linear-gradient(135deg, #0f766e 0%, #0369a1 100%) !important;
        border-radius: 8px !important;
        padding: 22px 28px !important;
        margin-bottom: 22px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    }
    .clinical-header * {
        color: #ffffff !important;
    }
    .clinical-header-title {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        margin: 0 !important;
        color: #ffffff !important;
    }
    .clinical-header-subtitle {
        font-size: 0.92rem !important;
        color: #e0f2fe !important;
        margin-top: 6px !important;
    }
    .clinical-tag {
        display: inline-block !important;
        background-color: rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
        padding: 3px 10px !important;
        border-radius: 12px !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        margin-top: 8px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    /* ========================================================= */
    /* 2. BARRA LATERAL (SIDEBAR) - FORÇAR BRANCO E PRATEADO BRILHANTE */
    /* ========================================================= */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
    }
    
    /* Todos os textos, parágrafos, spans e strongs dentro da sidebar */
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] strong,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #ffffff !important;
        font-size: 0.92rem !important;
    }

    /* Títulos da Barra Lateral */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Legendas e textos secundários da sidebar em prateado luminoso */
    [data-testid="stSidebar"] .stCaption, 
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    [data-testid="stSidebar"] caption {
        color: #cbd5e1 !important;
        font-size: 0.82rem !important;
        opacity: 0.95 !important;
    }

    /* Linhas divisórias da sidebar */
    [data-testid="stSidebar"] hr {
        border-color: #334155 !important;
        opacity: 0.6 !important;
    }

    /* ========================================================= */
    /* 3. BOTÕES, BADGES E CARDS */
    /* ========================================================= */
    .stButton > button {
        background-color: #0d9488 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 20px !important;
        font-size: 0.95rem !important;
    }
    .stButton > button p {
        color: #ffffff !important;
    }
    .stButton > button:hover {
        background-color: #0f766e !important;
        box-shadow: 0 4px 6px -1px rgba(13, 148, 136, 0.3) !important;
    }

    .badge-risco-alto {
        background-color: #fef2f2 !important;
        color: #991b1b !important;
        border: 1px solid #fecaca !important;
        padding: 4px 12px !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
    }
    .badge-risco-medio {
        background-color: #fffbeb !important;
        color: #92400e !important;
        border: 1px solid #fde68a !important;
        padding: 4px 12px !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
    }
    .badge-risco-baixo {
        background-color: #f0fdfa !important;
        color: #0f766e !important;
        border: 1px solid #99f6e4 !important;
        padding: 4px 12px !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
    }

    .medical-confirmation {
        background-color: #f0fdfa !important;
        border: 1px solid #99f6e4 !important;
        border-radius: 6px !important;
        padding: 14px 18px !important;
        margin-top: 15px !important;
        color: #0f766e !important;
        font-weight: 600 !important;
    }

    section.main div[data-testid="stContainer"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# Inicialização dos serviços em cache de sessão
@st.cache_resource
def get_classifier():
    classifier = TriageClassifier()
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "dataset_sintetico.csv")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(os.path.dirname(__file__), "dataset_sintetico.csv")
    
    if os.path.exists(dataset_path):
        classifier.train(dataset_path)
    return classifier

@st.cache_resource
def get_repository():
    db_path = os.path.join(os.path.dirname(__file__), "triage.db")
    return TriageRepository(db_path)

classifier = get_classifier()
repository = get_repository()

# Barra Lateral (Sidebar) com textos explícitos de alto contraste
with st.sidebar:
    st.markdown("### Unidade de Atendimento")
    st.caption("Ambulatório e Gestão de Demandas Comunitárias")
    st.markdown("**Instituição:** UNINOVE — Centro Universitário")
    st.markdown("**Responsável Técnico:** Gabriel Neves Moura")
    st.markdown("**Registro Acadêmico (RA):** 2225102570")
    st.divider()

    st.markdown("#### Status do Motor de Triagem")
    if classifier.is_trained:
        st.markdown("**Algoritmo Preditivo:** Multinomial Naive Bayes")
        st.caption(f"Base de conhecimento: {classifier.metrics.get('total_amostras', 0)} protocolos analisados")
        st.markdown("**Situação:** Operacional / Calibrado")
    else:
        st.warning("Motor de triagem aguardando inicialização da base.")

    st.divider()
    st.markdown("#### Diretrizes de Acolhimento")
    st.caption(
        "A classificação automatizada funciona como triagem preliminar de risco. "
        "A consulta, o diagnóstico e a conduta assistencial definitiva cabem estritamente aos profissionais de saúde e serviço social."
    )

# Banner Superior de Clínica / Saúde Comunitária
st.markdown("""
<div class="clinical-header">
    <div class="clinical-header-title">Portal de Triagem e Acolhimento Comunitário</div>
    <div class="clinical-header-subtitle">Sistema Informatizado de Classificação de Risco, Encaminhamento Social e Apoio à Decisão Médica</div>
    <div class="clinical-tag">Unidade de Saúde Integrada e Apoio Humanitário</div>
</div>
""", unsafe_allow_html=True)

# Abas de Navegação Clínica
aba1, aba2, aba3, aba4 = st.tabs([
    "1. Ficha de Acolhimento e Triagem",
    "2. Fila de Atendimento e Regulação",
    "3. Indicadores Clínicos e Estatísticos",
    "4. Prontuários e Exportação"
])

# =========================================================================
# ABA 1: FICHA DE ACOLHIMENTO E TRIAGEM CLÍNICA (RF01 - RF04, RNF01, RNF05)
# =========================================================================
with aba1:
    st.markdown("#### Ficha de Admissão e Anamnese Preliminar")
    st.caption("Cadastre os dados de contato do cidadão/paciente e descreva o quadro ou necessidade relatada.")

    with st.form("form_solicitacao", clear_on_submit=False):
        col_nome, col_contato = st.columns([2, 1])
        with col_nome:
            nome = st.text_input("Nome Completo do Paciente / Solicitante:", placeholder="Exemplo: Maria de Lourdes da Silva")
        with col_contato:
            contato = st.text_input("Telefone ou Contato de Emergência:", placeholder="Exemplo: (11) 98765-4321")

        mensagem = st.text_area(
            "Relato Livre do Caso / Sintomas / Necessidade Emergencial:",
            placeholder="Descreva o quadro: falta de alimentos, necessidade de medicamentos contínuos, risco físico, fraldas geriátricas ou apoio emergencial.",
            height=120
        )

        submitted = st.form_submit_button("Processar Triagem Clínica e Social", use_container_width=True)

    if submitted:
        if not mensagem.strip():
            st.error("Campo obrigatório: É necessário descrever o relato da situação para a triagem.")
        else:
            # Inferência via modelo
            resultado = classifier.predict(mensagem)

            if resultado["sucesso"]:
                # Persistência relacional
                nova_solicitacao = Solicitacao(
                    id=None,
                    nome_morador=nome.strip() if nome.strip() else "Identificação Reservada",
                    contato=contato.strip() if contato.strip() else "Não informado",
                    mensagem_original=mensagem,
                    mensagem_anonimizada=resultado["texto_anonimizado"],
                    categoria_predita=resultado["categoria"],
                    urgencia_predita=resultado["urgencia"],
                    confianca_categoria=resultado["confianca_categoria"],
                    confianca_urgencia=resultado["confianca_urgencia"]
                )
                solic_id = repository.save(nova_solicitacao)

                st.markdown(f"""
                <div class="medical-confirmation">
                    Ficha de Acolhimento registrada com sucesso. Protocolo Clínico Nº {solic_id}.
                </div>
                """, unsafe_allow_html=True)

                st.write("")
                st.markdown("##### Avaliação da Triagem Automatizada")
                c_cat, c_urg, c_perf = st.columns(3)

                with c_cat:
                    st.metric(
                        label="Especialidade / Domínio Identificado",
                        value=resultado["categoria"],
                        help=f"Confiabilidade preditiva do modelo: {resultado['confianca_categoria']*100:.1f}%"
                    )

                with c_urg:
                    risco_nome = "Risco Alto (Emergência)" if resultado["urgencia"] == "Alta" else ("Risco Moderado" if resultado["urgencia"] == "Média" else "Risco Habitual (Baixo)")
                    st.metric(
                        label="Classificação de Risco Preliminar",
                        value=risco_nome,
                        help=f"Score estatístico: {resultado['confianca_urgencia']*100:.1f}%"
                    )

                with c_perf:
                    st.metric(
                        label="Tempo de Resposta do Sistema",
                        value=f"{resultado['tempo_resposta_ms']:.2f} ms",
                        delta="Conforme protocolo de agilidade (< 2.0s)"
                    )

                with st.expander("Inspeção de Privacidade e Segurança do Prontuário (LGPD)"):
                    st.markdown("**Texto Higienizado com Mascaramento de Identificadores (LGPD):**")
                    st.code(resultado["texto_anonimizado"])
                    st.markdown("**Vetor de Termos Estruturados para Análise Estatística:**")
                    st.code(resultado["texto_limpo"])
            else:
                st.error(resultado["erro"])

# =========================================================================
# ABA 2: FILA DE REGULAÇÃO E ATENDIMENTO CLÍNICO (RF05, RN05)
# =========================================================================
with aba2:
    st.markdown("#### Painel de Regulação e Gestão da Fila")
    st.caption("Acompanhamento das demandas em tempo real com triagem de risco ponderada.")

    # Painel de Métricas da Clínica
    stats = repository.get_summary_stats()
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Fichas Cadastradas", stats["total"])
    kpi2.metric("Risco Alto (Emergência)", stats["urgencia_alta"])
    kpi3.metric("Aguardando Consulta/Ação", stats["pendentes"])
    kpi4.metric("Especialidades Ativas", len(stats["por_categoria"]))

    st.divider()

    # Filtros de Busca e Triagem
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        f_cat = st.selectbox("Especialidade / Domínio:", ["Todas", "Alimentação", "Vestuário", "Apoio Jurídico", "Saúde/Orientação"])
    with f_col2:
        f_urg = st.selectbox("Classificação de Risco:", ["Todas", "Alta", "Média", "Baixa"])
    with f_col3:
        f_status = st.selectbox("Situação do Acolhimento:", ["Todos", "Pendente", "Em Atendimento", "Concluído"])

    pedidos = repository.filter_requests(f_cat, f_urg, f_status)

    if not pedidos:
        st.info("Nenhuma ficha de acolhimento encontrada para os filtros selecionados.")
    else:
        st.markdown(f"Exibindo **{len(pedidos)}** prontuários regulados:")

        for p in pedidos:
            if p.urgencia_predita == "Alta":
                badge_html = '<span class="badge-risco-alto">RISCO ALTO / PRIORIDADE VERMELHA</span>'
            elif p.urgencia_predita == "Média":
                badge_html = '<span class="badge-risco-medio">RISCO MODERADO / PRIORIDADE AMARELA</span>'
            else:
                badge_html = '<span class="badge-risco-baixo">RISCO HABITUAL / PRIORIDADE VERDE</span>'

            with st.container():
                col_cab_esq, col_cab_dir = st.columns([3, 1])
                with col_cab_esq:
                    st.markdown(f"**Prontuário #{p.id} — Paciente: {p.nome_morador}** | Contato: `{p.contato}`")
                    st.caption(f"Horário de Admissão: {p.data_hora} | Domínio: **{p.categoria_predita}** (Confiabilidade: {p.confianca_categoria*100:.0f}%)")
                with col_cab_dir:
                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.caption(f"Conduta Atual: **{p.status}**")

                st.markdown(f"**Relato da Admissão:** *\"{p.mensagem_original}\"*")

                col_st_sel, col_st_btn = st.columns([2, 1])
                with col_st_sel:
                    novo_status = st.selectbox(
                        f"Atualizar Conduta do Protocolo #{p.id}:",
                        ["Pendente", "Em Atendimento", "Concluído"],
                        index=["Pendente", "Em Atendimento", "Concluído"].index(p.status),
                        key=f"status_select_{p.id}"
                    )
                with col_st_btn:
                    st.write("")
                    st.write("")
                    if st.button("Salvar Conduta", key=f"btn_save_{p.id}", use_container_width=True):
                        repository.update_status(p.id, novo_status)
                        st.success(f"Protocolo #{p.id} atualizado para '{novo_status}'.")
                        st.rerun()

# =========================================================================
# ABA 3: INDICADORES CLÍNICOS E ESTATÍSTICOS (RN08)
# =========================================================================
with aba3:
    st.markdown("#### Indicadores Estatísticos e Performance do Modelo")
    st.caption("Validação científica e calibração estatística do classificador supervisionado Naive Bayes.")

    m = classifier.metrics
    if m:
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Amostras de Calibração", m.get("total_amostras", 0))
        m_c2.metric("Casos de Teste / Validação", m.get("amostras_teste", 0))
        m_c3.metric("Sensibilidade / Acurácia (Domínio)", f"{m['categoria']['acuracia']*100:.1f}%")
        m_c4.metric("Concordância de Risco (Urgência)", f"{m['urgencia']['acuracia']*100:.1f}%")

        st.divider()

        col_rep_cat, col_rep_urg = st.columns(2)
        with col_rep_cat:
            st.markdown("##### Relatório de Desempenho: Domínios de Atendimento")
            rep_cat_df = pd.DataFrame(m['categoria']['relatorio']).transpose()
            st.dataframe(rep_cat_df.style.format(precision=2), use_container_width=True)

        with col_rep_urg:
            st.markdown("##### Relatório de Desempenho: Classificação de Gravidade")
            rep_urg_df = pd.DataFrame(m['urgencia']['relatorio']).transpose()
            st.dataframe(rep_urg_df.style.format(precision=2), use_container_width=True)

        st.divider()
        st.markdown("##### Matrizes de Confusão Preditiva")
        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            st.caption("Matriz de Concordância: Domínios")
            cm_cat_df = pd.DataFrame(
                m['categoria']['matriz_confusao'],
                index=m['categoria']['classes'],
                columns=m['categoria']['classes']
            )
            st.dataframe(cm_cat_df, use_container_width=True)

        with col_cm2:
            st.caption("Matriz de Concordância: Graus de Risco")
            cm_urg_df = pd.DataFrame(
                m['urgencia']['matriz_confusao'],
                index=m['urgencia']['classes'],
                columns=m['urgencia']['classes']
            )
            st.dataframe(cm_urg_df, use_container_width=True)
    else:
        st.info("Métricas disponíveis após o treinamento do modelo.")

# =========================================================================
# ABA 4: PRONTUÁRIOS E EXPORTAÇÃO DE RELATÓRIOS (RF06)
# =========================================================================
with aba4:
    st.markdown("#### Histórico de Prontuários e Exportação de Dados")
    st.caption("Extração dos registros de atendimento para auditoria médica e prestação de contas.")

    csv_data = repository.export_csv()
    json_data = repository.export_json()

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.download_button(
            label="Download da Base em Formato CSV (Padrão Hospitalar)",
            data=csv_data,
            file_name=f"prontuarios_clinica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with exp_col2:
        st.download_button(
            label="Download da Base em Formato JSON Interoperável",
            data=json_data,
            file_name=f"prontuarios_clinica_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    st.divider()
    st.markdown("##### Visualização Tabular de Prontuários Registrados")
    df_preview = pd.read_sql_query(
        "SELECT id AS Prontuario, nome_morador AS Paciente, categoria_predita AS Dominio, "
        "urgencia_predita AS Risco, status AS Conduta, data_hora AS Horario "
        "FROM solicitacoes ORDER BY id DESC",
        repository._get_connection()
    )
    st.dataframe(df_preview, use_container_width=True)
