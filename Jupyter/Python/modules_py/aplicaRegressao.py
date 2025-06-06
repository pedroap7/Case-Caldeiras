import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error
from category_encoders import TargetEncoder
import re


def sanitizar_nomes_colunas(feature_names):
    """
    Sanitiza os nomes das colunas para remover caracteres especiais.
    :param feature_names: Lista de nomes das colunas.
    :return: Lista de nomes sanitizados.
    """
    return [re.sub(r'[\[\]<>,]', '_', col) for col in feature_names]


def calcular_rmsle(y_true, y_pred):
    """
    Calcula o RMSLE, substituindo previsões negativas ou zero por um valor pequeno e positivo.
    :param y_true: Valores reais.
    :param y_pred: Valores preditos.
    :return: RMSLE.
    """
    # Substituir previsões negativas ou zero por um valor pequeno e positivo
    y_pred = np.where(y_pred < 0, 1e-10, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def aplicar_modelos_regressao(df, target_column, n_folds=4, n_bins=10):
    """
    Aplica modelos de regressão com diferentes encoders (OneHot, OneHot com PCA, Label, Target, Misto) usando validação cruzada com k-folds.
    :param df: DataFrame contendo os dados.
    :param target_column: Nome da coluna alvo.
    :param n_folds: Número de folds para validação cruzada.
    :return: DataFrame com os resultados.
    """
    # Identificando colunas categóricas e numéricas
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    num_cols = df.select_dtypes(include=['number']).columns.drop(target_column)

    # Convertendo valores categóricos para strings (garantindo compatibilidade com encoders)
    df[cat_cols] = df[cat_cols].astype(str)

    # Discretizando a variável target para estratificação
    
    n_samples = len(df)
    min_samples_per_bin = n_folds * 2  # Mínimo de 2 amostras por fold em cada bin
    max_bins = min(n_bins, n_samples // min_samples_per_bin)  # Número máximo de bins possível

    if max_bins < 2:
        raise ValueError(
            f"Não é possível criar estratos com n_folds={n_folds}. "
            f"Dataset muito pequeno (n_samples={n_samples}). "
            f"Reduza n_folds ou aumente o dataset."
        )

    y_binned = pd.qcut(df[target_column], q=max_bins, labels=False)

    # Verificar se todos os bins têm amostras suficientes
    bin_counts = pd.Series(y_binned).value_counts()
    
    if any(bin_counts < n_folds):
        # Ajuste dinâmico - reduz bins até que todos tenham amostras suficientes
        for bins in range(max_bins, 1, -1):
            y_binned = pd.qcut(df[target_column], q=bins, duplicates='drop', labels=False)
            bin_counts = pd.Series(y_binned).value_counts()
            if all(bin_counts >= n_folds):
                break

    # Lista de encoders a serem testados
    encoders = {
        'onehot': OneHotEncoder(handle_unknown='ignore', sparse_output=False),
        'label': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
        'target_enc': TargetEncoder(),
        'mix_enc': None,  # Será definido dinamicamente
        'onehot_pca': None  # Será definido dinamicamente
    }

    # Modelos a serem testados
    modelos = {
        "Regressao Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "Arvore de Decisao": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42),
        "LightGBM": LGBMRegressor(random_state=42),
        "CatBoost": CatBoostRegressor(random_state=42, verbose=0),
        "SVR (Linear Kernel)": SVR(kernel='linear', C=10, epsilon=0.2),
        "SVR (RBF Kernel)": SVR(kernel='rbf', C=10, epsilon=0.2),
        "SVR (Sigmoid Kernel)": SVR(kernel='sigmoid', C=10, epsilon=0.2)
    }

    resultados = []

    # K-Fold Cross Validation
    # kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for encoder_name, encoder in encoders.items():
        print(f"\nProcessando com encoder: {encoder_name}")
        print(f"Distribuição dos estratos: {pd.Series(y_binned).value_counts().to_dict()}")

        # Dicionário para armazenar métricas de cada modelo
        metricas_por_modelo = {nome_modelo: {"R2": [], "MAE": [], "RMSE": [], "RMSLE": [], "Negative_Preds": []} for nome_modelo in modelos.keys()}

        for fold, (train_index, test_index) in enumerate(skf.split(df, y_binned)):
            X_train, X_test = df.iloc[train_index], df.iloc[test_index]
            y_train, y_test = df[target_column].iloc[train_index], df[target_column].iloc[test_index]


            # Tenho que melhorar todo esse código abaixo nesse loop. Pode estar repetitivo

            # Aplicando o encoder
            if encoder_name == 'target_enc':
                # Target Encoding é aplicado apenas no treino
                X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols], y_train)
                X_test[cat_cols] = encoder.transform(X_test[cat_cols])
                preprocessor = make_column_transformer(
                    (StandardScaler(), num_cols),
                    ('passthrough', cat_cols)
                )

            elif encoder_name == 'mix_enc':
                # Encoder misto: OneHot para colunas com até n categorias (n_cat), Label para o restante

                n_cat = 5

                onehot_cols = [col for col in cat_cols if X_train[col].nunique() <= n_cat]
                label_cols = [col for col in cat_cols if X_train[col].nunique() > n_cat]
                preprocessor = make_column_transformer(
                    (StandardScaler(), num_cols),
                    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols),
                    (OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), label_cols)
                )

            elif encoder_name == 'onehot_pca':
                # OneHot Encoding seguido de PCA
                onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                pca = PCA(n_components=0.85)  # Manter 85% da variância
                preprocessor = make_column_transformer(
                    (StandardScaler(), num_cols),
                    (Pipeline(steps=[
                        ('onehot', onehot_encoder),
                        ('pca', pca)
                    ]), cat_cols)
                )

            else:
                # OneHot ou Label Encoding aplicados no dataset completo
                preprocessor = make_column_transformer(
                    (StandardScaler(), num_cols),
                    (encoder, cat_cols)
                )

            # Pré-processar dados
            X_train_preprocessed = preprocessor.fit_transform(X_train)
            X_test_preprocessed = preprocessor.transform(X_test)



            # Converter para DataFrame para manter os nomes das colunas
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out()
            else:
                feature_names = num_cols.tolist() + cat_cols.tolist()

            # Sanitizar nomes das colunas
            feature_names = sanitizar_nomes_colunas(feature_names)

            X_train_preprocessed = pd.DataFrame(X_train_preprocessed, columns=feature_names)
            X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=feature_names)



            for nome_modelo, modelo in modelos.items():

            
                # Treinar modelo
                modelo.fit(X_train_preprocessed, y_train)
                pipeline = Pipeline(steps=[
                    ('regressor', modelo)
                ])

                # Treinar modelo
                pipeline.fit(X_train_preprocessed, y_train) # Transformação log

                # Previsões
                y_pred = pipeline.predict(X_test_preprocessed) # Transformação Inversa

                # Garantir não-negatividade
                y_pred = np.maximum(y_pred, 0)

                # Contar previsões negativas
                negative_preds = np.sum(y_pred < 0)

                # Métricas de avaliação
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                rmsle = calcular_rmsle(y_test, y_pred)
                
                
                

                # Armazenar métricas no dicionário
                metricas_por_modelo[nome_modelo]["R2"].append(r2)
                metricas_por_modelo[nome_modelo]["MAE"].append(mae)
                metricas_por_modelo[nome_modelo]["RMSE"].append(rmse)
                metricas_por_modelo[nome_modelo]["RMSLE"].append(rmsle)
                metricas_por_modelo[nome_modelo]["Negative_Preds"].append(negative_preds)

                # Armazenar resultados por fold
                resultados.append({
                    "Encoder": encoder_name,
                    "Fold": fold + 1,
                    "Modelo": nome_modelo,
                    "R2": r2,
                    "MAE": mae,
                    "RMSE": rmse,
                    "RMSLE": rmsle,
                    "Negative_Preds": negative_preds
                })

        # Calcular médias das métricas por modelo e adicionar ao resultados
        for nome_modelo, metricas in metricas_por_modelo.items():
            media_r2 = np.mean(metricas["R2"])
            media_mae = np.mean(metricas["MAE"])
            media_rmse = np.mean(metricas["RMSE"])
            media_rmsle = np.mean(metricas["RMSLE"])
            total_negative_preds = np.sum(metricas["Negative_Preds"])

            resultados.append({
                "Encoder": encoder_name,
                "Fold": "Média",
                "Modelo": nome_modelo,
                "R2": media_r2,
                "MAE": media_mae,
                "RMSE": media_rmse,
                "RMSLE": media_rmsle,
                "Negative_Preds": total_negative_preds
            })

    return pd.DataFrame(resultados)


# # Exemplo de uso
# data = {
#     'cat_feature_1': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'],
#     'cat_feature_2': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'],
#     'num_feature_1': [1, 2, 3, 4, 5, 6, 7, 8],
#     'num_feature_2': [10, 20, 30, 40, 50, 60, 70, 80],
#     'target': [100, 200, 150, 250, 300, 350, 400, 450]
# }

# df = pd.DataFrame(data)
# resultados = aplicar_modelos_regressao(df, target_column='target', n_folds=4)
# print(resultados)
