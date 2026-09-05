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
    model = genai.GenerativeModel('gemini-3.1-pro') 
except Exception as e:
    st.error("Erro ao configurar a API. Verifique se a GEMINI_API_KEY está nos secrets.")
    st.stop()

# 3. Gerenciamento de Categorias Personalizadas no Estado da Sessão
if 'categorias_permitidas' not in st.session_state:
    st.session_state['categorias_permitidas'] = [
        'Alimentação', 'Moradia', 'Transporte', 'Investimentos', 
        'Lazer', 'Saúde', 'Negócios', 'Manutenção', 'Outros'
    ]

# Sidebar para gerenciar novas categorias personalizadas
st.sidebar.header("🏷️ Gerenciar Categorias")
nova_cat_input = st.sidebar.text_input("Adicionar nova categoria:")
if st.sidebar.button("Cadastrar Categoria"):
    if nova_cat_input and nova_cat_input.strip() not in st.session_state['categorias_permitidas']:
        st.session_state['categorias_permitidas'].append(nova_cat_input.strip())
        st.sidebar.success(f"Categoria '{nova_cat_input.strip()}' adicionada!")
        st.rerun()

st.sidebar.write("Categorias ativas atualmente:")
st.sidebar.write(", ".join(st.session_state['categorias_permitidas']))

# 4. Interface de Upload
st.write("Faça o upload do seu extrato bancário (formato CSV).")
arquivo_upload = st.file_uploader("Escolha um arquivo CSV", type=["csv"])

if arquivo_upload is not None:
    try:
        df = pd.read_csv(arquivo_upload, sep=None, engine='python')
    except Exception:
        df = pd.read_csv(arquivo_upload)
    
    st.write("### Pré-visualização dos Dados")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        coluna_descricao = st.selectbox("Qual coluna contém a DESCRIÇÃO da compra?", df.columns)
    with col2:
        coluna_valor = st.selectbox("Qual coluna contém o VALOR do gasto?", df.columns)
    
    if st.button("Processar, Classificar e Gerar Dashboard"):
        with st.spinner("A IA está analisando e categorizando seus gastos..."):
            
            descricoes = df[coluna_descricao].dropna().unique().tolist()
            texto_gastos = "\n".join([str(d) for d in descricoes])
            lista_str = ", ".join(st.session_state['categorias_permitidas'])
            
            prompt = f"""
            Você é um consultor financeiro especialista. Sua tarefa é categorizar estritamente a seguinte lista de descrições de extrato bancário.
            Retorne APENAS linhas no formato exato: `Descrição Exata;Categoria` (separado por ponto e vírgula, sem markdown adicional, sem bullets).
            
            Categorias permitidas que você pode usar: {lista_str}. Se nenhuma se encaixar perfeitamente, use 'Outros'.
            
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
                        if cat in st.session_state['categorias_permitidas']:
                            mapa_categorias[desc] = cat
                        else:
                            mapa_categorias[desc] = 'Outros'
                
                df['Categoria'] = df[coluna_descricao].map(mapa_categorias).fillna('Outros')
                
                def limpar_e_converter_valor(val):
                    if pd.isna(val):
                        return 0.0
                    if isinstance(val, (int, float)):
                        return float(val)
                    val_str = str(val).replace('R$', '').replace('$', '').strip()
                    if '.' in val_str and ',' in val_str:
                        val_str = val_str.replace('.', '').replace(',', '.')
                    elif ',' in val_str and '.' not in val_str:
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

# 5. Renderização do Dashboard e Editor Manual
if 'df_processado' in st.session_state:
    df_proc = st.session_state['df_processado']
    col_val = st.session_state['coluna_valor']
    
    st.markdown("---")
    st.header("🛠️ Refinamento Manual e Novas Categorias")
    st.write("Ajuste as categorias abaixo utilizando as opções disponíveis (incluindo as que você criou na barra lateral).")
    
    df_editado = st.data_editor(
        df_proc,
        column_config={
            "Categoria": st.column_config.SelectboxColumn(
                "Categoria",
                help="Selecione a categoria correta",
                options=st.session_state['categorias_permitidas'],
                required=True
            )
        },
        use_container_width=True,
        key="editor_categorias"
    )
    
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
        
    # 6. Seção de Inteligência Artificial para Otimização de Custos
    st.markdown("---")
    st.header("💡 Insights e Otimização de Custos por IA")
    
    if st.button("Gerar Plano de Otimização de Custos"):
        with st.spinner("Analisando padrões financeiros refinados..."):
            resumo_financeiro = df_editado.groupby('Categoria')[col_val].sum().to_string()
            
            prompt_otimizacao = f"""
            Com base nos seguintes totais agregados por categoria de um extrato bancário (incluindo categorias personalizadas criadas pelo usuário), atue como um consultor financeiro pessoal de elite. 
            Forneça um plano estratégico, direto e prático de otimização de custos, identificando potenciais ralos de dinheiro e propondo cortes inteligentes:
            
            {resumo_financeiro}
            
            Organize a resposta de forma limpa, destacando áreas críticas de atenção e sugestões acionáveis de economia.
            """
            
            resposta_otimizacao = model.generate_content(prompt_otimizacao)
            st.markdown(resposta_otimizacao.text)
