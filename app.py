
import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# 1. Configuração da Página
st.set_page_config(page_title="Controle Financeiro IA", layout="wide")
st.title("📊 Classificador e Otimizador Financeiro")

# 2. Configuração da API do Gemini
try:
    # Tenta pegar a chave dos secrets do Streamlit
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    # Usando o modelo mais rápido e eficiente para texto
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
    
    # Seleção da coluna que contém a descrição dos gastos
    coluna_descricao = st.selectbox("Qual coluna contém a descrição da compra?", df.columns)
    
    if st.button("Classificar Gastos com IA"):
        with st.spinner("A IA está analisando seus gastos..."):
            
            # Prepara a lista de gastos para enviar à IA (limitando para não estourar o prompt no teste)
            descricoes = df[coluna_descricao].dropna().unique().tolist()
            texto_gastos = "\n".join(descricoes)
            
            # Prompt engenhado para evitar alucinações e dar contexto
            prompt = f"""
            Você é um consultor financeiro. Sua tarefa é categorizar estritamente a seguinte lista de descrições de extrato bancário.
            Retorne APENAS o nome da categoria original ao lado de uma categoria sugerida, no formato: 'Descrição Original -> Categoria'.
            
            Categorias permitidas: Alimentação, Moradia, Transporte, Investimentos, Lazer, Saúde, Negócios, Manutenção, Outros.
            
            Exemplos de raciocínio lógico que você deve aplicar:
            - Custos com 'Terra Nativa' ou 'Car & Bike Hunter' -> Negócios
            - Compras em concessionárias, peças para 'V-Strom', 'Triumph' ou postos de combustível -> Manutenção/Transporte
            - Transferências para corretoras, ou aportes em 'BTLG11', 'CPTS11', 'HGLG11' -> Investimentos
            
            Lista de gastos para classificar:
            {texto_gastos}
            """
            
            try:
                # Chama a IA
                resposta = model.generate_content(prompt)
                
                st.write("### Resultado da Classificação")
                st.text(resposta.text)
                
                st.success("Classificação concluída! O próximo passo será integrar essas categorias de volta à tabela e pedir dicas de otimização.")
                
            except Exception as e:
                st.error(f"Erro ao processar com a IA: {e}")
