import pandas as pd
import yaml
import os

def carregar_config(caminho: str) -> dict:
    with open(caminho, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def carregar_dados(config_data: dict) -> pd.DataFrame:
    caminho = config_data['dataset']['path']
    print(f"Carregando dados de: {caminho}")
    df = pd.read_csv(caminho, low_memory=False)
    print(f"Shape bruto: {df.shape}")
    return df

def filtrar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replica exatamente o filtro do projeto anterior:
    - Apenas leads do programa PGL
    - Apenas leads que interagiram com o chatbot
    """
    filtro_pgl = df['CodPrograma'].str.contains('PGL', case=False, na=False)
    filtro_chatbot = df['Data do último preenchimento de chatbot'].notna()
    df_filtrado = df[filtro_pgl & filtro_chatbot].copy()
    print(f"Shape após filtros PGL + Chatbot: {df_filtrado.shape}")
    return df_filtrado

def selecionar_colunas(df: pd.DataFrame, config_data: dict) -> pd.DataFrame:
    features_num = config_data['features']['numerical']
    features_cat = config_data['features']['categorical']
    target = config_data['dataset']['target_column']
    colunas = [target] + features_num + features_cat
    # Mapeia nome do target original
    df = df.rename(columns={'Fase do ciclo de vida': target})
    df_sel = df[colunas].copy()
    print(f"Colunas selecionadas: {df_sel.shape[1]}")
    return df_sel

def normalizar_cargo(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa cargos raros em 'Outros' — engenharia de feature do projeto anterior."""
    cargos_validos = [
        'Analista', 'Especialista', 'Assistente', 'Coordenador',
        'Gerente', 'Consultor', 'Auxiliar', 'Estagiário',
        'Trainee', 'Diretor', 'Supervisor', 'Superintendente',
        'Não trabalho no momento'
    ]
    df['Cargo'] = df['Cargo'].replace(
        ['Não trabalho no momento.', 'Não trabalho no momento'],
        'Não trabalho no momento'
    )
    mask_outros = ~df['Cargo'].isin(cargos_validos) & df['Cargo'].notna()
    df.loc[mask_outros, 'Cargo'] = 'Outros'
    return df

def criar_target(df: pd.DataFrame, config_data: dict) -> pd.DataFrame:
    target_col = config_data['dataset']['target_column']
    df[target_col] = (df[target_col] == 'Cliente').astype(int)
    print(f"\nDistribuição do target:")
    print(df[target_col].value_counts())
    print(f"Taxa de conversão: {df[target_col].mean()*100:.2f}%")
    return df

def ingerir(caminho_config_data: str = 'configs/data.yaml') -> pd.DataFrame:
    config_data = carregar_config(caminho_config_data)
    df = carregar_dados(config_data)
    df = filtrar_dados(df)
    df = selecionar_colunas(df, config_data)
    df = normalizar_cargo(df)
    df = criar_target(df, config_data)
    return df

if __name__ == '__main__':
    df = ingerir()
    print("\nAmostra dos dados prontos:")
    print(df.head())
    print("\nNulos por coluna:")
    print(df.isnull().sum())