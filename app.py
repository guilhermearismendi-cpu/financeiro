import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import plotly.express as px

# 1. Configuração da Página
st.set_page_config(page_title="Controle Financeiro IA", layout="wide")
st.title("📊 Classificador e Dashboard Financeiro com IA")

# 2. Configuração da API do Gemini
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash') 
except Exception as e:
    st.error("Erro ao configurar a API. Verifique se a GEMINI_API_KEY está nos secrets.")
    st.stop()

# 3. Interface de Upload
st.write("Faça o upload do seu extrato bancário (formato CSV).")
arquivo_upload = st.file_uploader("Escolha um arquivo CSV", type=["csv"])

if arquivo_upload is not None:
    # Lê o arquivo CSV com detecção de separador
    try:
        df = pd.read_csv(arquivo_upload, sep=None, engine='python')
    except Exception:
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
        with st.spinner("A IA está analisando e categorizando seus gastos..."):
            
            # Prepara a lista de descrições únicas
            descricoes = df[coluna_descricao].dropna().unique().tolist()
            texto_gastos = "\n".join([str(d) for d in descricoes])
            
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
                
                mapa_categorias = {}
                for linha in linhas:
                    if ';' in linha:
                        partes = linha.split(';', 1)
                        desc = partes[0].strip()
                        cat = partes[1].strip()
                        mapa_categorias[desc] = cat
                
                # Mapeia de volta para o DataFrame principal
                df['Categoria'] = df[coluna_descricao].map(mapa_categorias).fillna('Outros')
                
                # Conversão robusta de valores para evitar o erro de valor zerado
                def limpar_e_converter_valor(val):
                    if pd.isna(val):
                        return 0.0
                    if isinstance(val, (int, float)):
                        return float(val)
                    
                    # Remove R$, espaços e símbolos monetários
                    val_str = str(val).replace('R$', '').replace('$', '').strip()
                    
                    # Trata formatos brasileiros (ex: 1.234,56) vs americanos (ex: 1234.56)
                    if '.' in val_str and ',' in val_str:
                        # Assume formato 1.234,56
                        val_str = val_str.replace('.', '').replace(',', '.')
                    elif ',' in val_str and '.' not in val_str:
                        # Assume formato 1234,56
                        val_str = val_str.replace(',', '.')
                    
                    try:
                        return float(val_str)
                    except ValueError:
                        return 0.0

                df[coluna_valor] = df[coluna_valor].apply(limpar_e_converter_valor)
                
                st.session_state['df_processado'] = df
                st.session_state['coluna_valor'] = coluna_valor
                st.success("Gastos classificados com sucesso!")
                
            except Exception as e:
                st.error(f"Erro ao processar com a IA: {e}")

# 4. Renderização do Dashboard e Editor Manual
if 'df_processado' in st.session_state:
    df_proc = st.session_state['df_processado']
    col_val = st.session_state['coluna_valor']
    
    st.markdown("---")
    st.header("🛠️ Refinamento Manual (Ajuste de 'Outros')")
    st.write("Abaixo você pode visualizar e editar diretamente as categorias dos lançamentos (especialmente os que caíram em 'Outros'). Qualquer alteração feita na tabela abaixo atualiza o dashboard instantaneamente.")
    
    # Tabela interativa para editar categorias manualmente
    lista_categorias_permitidas = ['Alimentação', 'Moradia', 'Transporte', 'Investimentos', 'Lazer', 'Saúde', 'Negócios', 'Manutenção', 'Outros']
    
    # Usando st.data_editor para edição interativa
    df_editado = st.data_editor(
        df_proc,
        column_config={
            "Categoria": st.column_config.SelectboxColumn(
                "Categoria",
                help="Selecione a categoria correta",
                options=lista_categorias_permitidas,
                required=True
            )
        },
        use_container_width=True,
        key="editor_categorias"
    )
    
    # Atualiza o dataframe principal com as edições manuais do usuário
    st.session_state['df_processado'] = df_editado
    
    st.markdown("---")
    st.header("📈 Dashboard Analítico de Gastos")
    
    total_movimentado = df_editado[col_val].sum()
    total_categorias = df_editado['Categoria'].nunique()
    total_linhas = len(df_editado)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Valor Total Analisado", f"R$ {total_movimentado:,.2f}")
    m2.metric("Categorias Ativas", f"{total_categorias}")
    m3.metric("Total de Lançamentos", f"{total_linhas}")
    
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Distribuição por Categoria")
        gasto_por_cat = df_editado.groupby('Categoria')[col_val].sum().reset_index()
        fig_pie = px.pie(gasto_por_cat, names='Categoria', values=col_val, hole=0.4, color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with g2:
        st.subheader("Gastos Totais por Categoria (Barras)")
        fig_bar = px.bar(gasto_por_cat, x='Categoria', y=col_val, color='Categoria', color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    # 5. Seção de Inteligência Artificial para Otimização de Custos
    st.markdown("---")
    st.header("💡 Insights e Otimização de Custos por IA")
    
    if st.button("Gerar Plano de Otimização de Custos"):
        with st.spinner("Analisando padrões financeiros refinados..."):
            resumo_financeiro = df_editado.groupby('Categoria')[col_val].sum().to_string()
            
            prompt_otimizacao = f"""
            Com base nos seguintes totais agregados por categoria de um extrato bancário (já refinados e corrigidos pelo usuário), atue como um consultor financeiro pessoal de elite. 
            Forneça um plano estratégico, direto e prático de otimização de custos, identificando potenciais ralos de dinheiro e propondo cortes inteligentes:
            
            {resumo_financeiro}
            
            Organize a resposta de forma limpa, destacando áreas críticas de atenção e sugestões acionáveis de economia.
            """
            
            resposta_otimizacao = model.generate_content(prompt_otimizacao)
            st.markdown(resposta_otimizacao.text)
