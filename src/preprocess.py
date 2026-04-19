import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def carregar_config(caminho: str) -> dict:
    with open(caminho, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def criar_preprocessador(config_data: dict) -> ColumnTransformer:
    features_num = config_data['features']['numerical']
    features_cat = config_data['features']['categorical']

    transformer_numerico = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    transformer_categorico = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Não informado')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessador = ColumnTransformer(transformers=[
        ('num', transformer_numerico, features_num),
        ('cat', transformer_categorico, features_cat)
    ])

    return preprocessador

def dividir_dados(df: pd.DataFrame, config_data: dict):
    target = config_data['dataset']['target_column']
    test_size = config_data['dataset']['test_size']
    random_state = config_data['dataset']['random_state']

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")
    print(f"Conversão treino: {y_train.mean()*100:.2f}% | Conversão teste: {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    from ingest import ingerir
    config_data = carregar_config('configs/data.yaml')
    df = ingerir()
    X_train, X_test, y_train, y_test = dividir_dados(df, config_data)
    preprocessador = criar_preprocessador(config_data)
    print("\nPreprocessador criado com sucesso!")
    print(f"Features numéricas: {config_data['features']['numerical']}")
    print(f"Features categóricas: {config_data['features']['categorical']}")