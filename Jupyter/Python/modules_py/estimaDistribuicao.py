import pandas as pd
import math
import numpy as np
from scipy.stats import kstest, anderson, cramervonmises, chi2_contingency, gamma, norm, weibull_min, lognorm, expon
from matplotlib import pyplot as plt
from pathlib import Path
import warnings

def estima_distribuicao(datasets, filtros, coluna='SomaDeHorasApontadasUnitario', quant_amostragens=0, fracao=0.6):
    if filtros:
        quant_datasets = len(datasets)
    else:
        quant_datasets = 1

    resultados_resumo = []

    for i in range(quant_datasets):
        for j in range(1, 3):  # Com ou sem outliers
            for k in range(quant_amostragens + 1):  # Amostragens
                print(f"i{i+1} j{j} k{k}")

                # Seleciona dataset
                filtrado = datasets[i] if filtros else datasets
                grupo_atividade = i + 1
                outlier_status = "Com" if j == 1 else "Sem"

                # Realiza amostragem se necessário
                if k > 0:
                    filtrado = filtrado.sample(frac=fracao, random_state=k)

                # Nome do arquivo
                nome_arquivo = f"DURACAO-{grupo_atividade}-{outlier_status}Outlier-{k}"

                # Ajusta a coluna de análise
                # filtrado[coluna] = filtrado[coluna] / 1000
                filtrado2 = filtrado[coluna]

                # Detecção de outliers
                print("Detecção de outliers")
                q1 = np.percentile(filtrado2, 25)
                q3 = np.percentile(filtrado2, 75)
                iqr = q3 - q1
                lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                if j == 2:  # Remove outliers
                    filtrado2 = filtrado2[(filtrado2 >= lb) & (filtrado2 <= ub)]

                # Estima distribuições
                print("Estimando distribuições")
                x = filtrado2.to_numpy()

                def fit_distribution(data, dist, **params):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            result = dist.fit(data, **params)
                        return result
                    except Exception as e:
                        print(f"Falha ao ajustar {dist.name}: {e}")
                        return None

                distribs = {
                    "weibull": weibull_min,
                    "gamma": gamma,
                    "lognormal": lognorm,
                    "exponential": expon,
                    "normal": norm,
                }

                results = {name: fit_distribution(x, dist) for name, dist in distribs.items()}
                results = {k: v for k, v in results.items() if v is not None}

                # Definindo número de bins do histograma
                num_bins = int(np.sqrt(len(x)))

                # Testes de aderência
                print("Testes de aderência")
                ad_tests = {}
                chi2_tests = {}

                for name, dist in distribs.items():
                    if name in results:
                        params = results[name]
                        try:
                            # Kolmogorov-Smirnov Test
                            test_stat, p_value = kstest(x, dist.cdf, args=params)
                            ad_tests[name] = {"kst_stat": test_stat, "kst_p": p_value}
                        except Exception as e:
                            print(f"Erro no KST para {name}: {e}")
                            ad_tests[name] = None

                        # Monte Carlo Test
                        try:
                            num_simulations = 1000  # Número de simulações de Monte Carlo
                            simulated_stats = []
                            for _ in range(num_simulations):
                                simulated_sample = dist.rvs(*params, size=len(x))
                                simulated_stat, _ = kstest(simulated_sample, dist.cdf, args=params)
                                simulated_stats.append(simulated_stat)

                            observed_stat, _ = kstest(x, dist.cdf, args=params)
                            mc_p_value = (np.sum(np.array(simulated_stats) >= observed_stat) + 1) / (num_simulations + 1)
                            ad_tests[name]["mc_stat"] = observed_stat
                            ad_tests[name]["mc_p"] = mc_p_value
                        except Exception as e:
                            print(f"Erro no Monte Carlo para {name}: {e}")
                            ad_tests[name]["mc_stat"] = None
                            ad_tests[name]["mc_p"] = None

                        # Cramér-von Mises Test
                        try:
                            cvm_result = cramervonmises(x, lambda v: dist.cdf(v, *params))
                            ad_tests[name]["cvm_stat"] = cvm_result.statistic
                            ad_tests[name]["cvm_p"] = cvm_result.pvalue
                        except Exception as e:
                            print(f"Erro no Cramér-von Mises para {name}: {e}")
                            ad_tests[name]["cvm_stat"] = None
                            ad_tests[name]["cvm_p"] = None

                        # Qui-quadrado
                        try:
                            sturges_rule = int((math.log2(len(x)))+1)
                            observed, bins = np.histogram(x, sturges_rule)
                            midpoints = (bins[:-1] + bins[1:]) / 2
                            expected = len(x) * dist.pdf(midpoints, *params)
                            expected[expected < 1e-5] = 1e-5
                            chi2_stat, chi2_p = chi2_contingency([observed, expected])[:2]
                            chi2_tests[name] = {"chi2_stat": chi2_stat, "chi2_p": chi2_p}
                        except Exception as e:
                            print(f"Erro no Chi-Square para {name}: {e}")

                resultados_resumo.append({
                    "dataset": i + 1,
                    "outlier_status": outlier_status,
                    "sample": k,
                    "ad_tests": ad_tests,
                    "chi2_tests": chi2_tests,
                    "distribuicoes": distribs,
                    "data": x,  # Adicionando os dados usados
                })

    return resultados_resumo

