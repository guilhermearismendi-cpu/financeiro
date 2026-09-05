import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import plotly.express as px

# 1. Configuração da Página
st.set_page_config(page_title="Controle Financeiro IA", layout="wide")
st.title("📊 Classificador e Dashboard Financeiro com IA")

# 2. Configuração da API do Gemini (Com o modelo atualizado e estável)
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.6-flash') 
except Exception as e:
    st.error("Erro ao configurar a API. Verifique se a GEMINI_API_KEY está nos secrets.")
    st.stop()

# 3. Interface de Upload
st.write("Faça o upload do seu extrato bancário (formato CSV).")
arquivo_upload = st.file_uploader("Escolha um arquivo CSV", type=["csv"])

if arquivo_upload is not None:
    # Lê o arquivo CSV
    df = pd.read_csv(arquivo_upload)
    
    st.write("### Pré-visualização dos Dados")
    st.dataframe(df.head())
    
    # Seleção das colunas essenciais
    col1, col2 = st.columns(2)
    with col1:
        coluna_descricao = st.selectbox("Qual coluna contém a DESCRIÇÃO da compra?", df.columns)
    with col2:
        coluna_valor = st.selectbox("Qual coluna contém o VALOR do gasto?", df.columns)
    
    if st.button("Processar, Classificar e Gerar Dashboard"):
        with st.spinner("A IA está analisando, estruturando e categorizando seus gastos..."):
            
            # Prepara a lista de descrições únicas
            descricoes = df[coluna_descricao].dropna().unique().tolist()
            texto_gastos = "\n".join([str(d) for d in descricoes])
            
            # Prompt estruturado para retornar um mapeamento limpo (Descrição;Categoria)
            prompt = f"""
            Você é um consultor financeiro especialista. Sua tarefa é categorizar estritamente a seguinte lista de descrições de extrato bancário.
            Retorne APENAS linhas no formato exato: `Descrição Exata;Categoria` (separado por ponto e vírgula, sem markdown adicional, sem bullets).
            
            Categorias permitidas: Alimentação, Moradia, Transporte, Investimentos, Lazer, Saúde, Negócios, Manutenção, Outros.
            
            Exemplos de raciocínio lógico que você deve aplicar:
            - Custos com 'Terra Nativa' ou 'Car & Bike Hunter' -> Negócios
            - Compras em concessionárias, peças para 'V-Strom', 'Triumph' ou postos de combustível -> Manutenção
            - Uber, passagens, passagens aéreas, combustíveis gerais -> Transporte
            - Transferências para corretoras, ou aportes em 'BTLG11', 'CPTS11', 'HGLG11' -> Investimentos
            
            Lista de gastos para classificar:
            {texto_gastos}
            """
            
            try:
                resposta = model.generate_content(prompt)
                linhas = resposta.text.strip().split('\n')
                
                # Cria um dicionário de mapeamento entre a descrição original e a categoria da IA
                mapa_categorias = {}
                for linha in linhas:
                    if ';' in linha:
                        partes = linha.split(';', 1)
                        desc = partes[0].strip()
                        cat = partes[1].strip()
                        mapa_categorias[desc] = cat
                
                # Mapeia de volta para o DataFrame principal
                df['Categoria'] = df[coluna_descricao].map(mapa_categorias).fillna('Outros')
                
                # Tratamento básico da coluna de valor para garantir formato numérico
                df[coluna_valor] = pd.to_numeric(
                    df[coluna_valor].astype(str).str.replace('R$', '', regex=True)
                    .str.replace('.', '', regex=True)
                    .str.replace(',', '.', regex=True), 
                    errors='coerce'
                ).fillna(0)
                
                # Salva no estado da sessão do Streamlit para manter os dados visíveis
                st.session_state['df_processado'] = df
                st.session_state['coluna_valor'] = coluna_valor
                st.success("Gastos classificados e estruturados com sucesso!")
                
            except Exception as e:
                st.error(f"Erro ao processar com a IA: {e}")

# 4. Renderização do Dashboard se os dados já estiverem processados
if 'df_processado' in st.session_state:
    df_proc = st.session_state['df_processado']
    col_val = st.session_state['coluna_valor']
    
    st.markdown("---")
    st.header("📈 Dashboard Analítico de Gastos")
    
    # Métricas Resumidas no topo
    total_movimentado = df_proc[col_val].sum()
    total_categorias = df_proc['Categoria'].nunique()
    total_linhas = len(df_proc)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Valor Total Analisado", f"R$ {total_movimentado:,.2f}")
    m2.metric("Categorias Ativas", f"{total_categorias}")
    m3.metric("Total de Lançamentos", f"{total_linhas}")
    
    # Gráficos e Tabela Detalhada
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Distribuição por Categoria")
        gasto_por_cat = df_proc.groupby('Categoria')[col_val].sum().reset_index()
        fig_pie = px.pie(gasto_por_cat, names='Categoria', values=col_val, hole=0.4, color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with g2:
        st.subheader("Tabela Classificada Completa")
        st.dataframe(df_proc, use_container_width=True)
        
    # 5. Seção de Inteligência Artificial para Otimização de Custos
    st.markdown("---")
    st.header("💡 Insights e Otimização de Custos por IA")
    st.write("Clique no botão abaixo para pedir à IA uma análise profunda dos seus gastos com recomendações de economia.")
    
    if st.button("Gerar Plano de Otimização de Custos"):
        with st.spinner("Analisando padrões financeiros e calculando margens de otimização..."):
            resumo_financeiro = df_proc.groupby('Categoria')[col_val].sum().to_string()
            
            prompt_otimizacao = f"""
            Com base nos seguintes totais agregados por categoria de um extrato bancário, atue como um consultor financeiro pessoal de elite. 
            Forneça um plano estratégico, direto e prático de otimização de custos, identificando potenciais ralos de dinheiro e propondo cortes inteligentes:
            
            {resumo_financeiro}
            
            Organize a resposta de forma limpa, destacando áreas críticas de atenção e sugestões acionáveis de economia.
            """
            
            resposta_otimizacao = model.generate_content(prompt_otimizacao)
            st.markdown(resposta_otimizacao.text)
