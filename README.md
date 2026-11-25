# Projeto CGMacros — Módulo 1: Pré-processamento (Integrante 1)

Lê o dataset CGMacros v1.0.0 e entrega as tabelas que a rede neural (Integrante 2) e o algoritmo genético (Integrante 3) usam.

## Como rodar
```bash
pip install -r requirements.txt
# O dataset fica FORA do projeto (não entra no zip). Aponte para a pasta onde você baixou/extraiu o PhysioNet;
# a busca por CGMacros-XXX.csv e bio.csv é recursiva, então pode apontar para a pasta raiz.
python preprocessing.py --raw "C:\caminho\para\cgmacros-1.0.0" --out data\processed
```
Nos notebooks (`notebooks/`), ajuste `RAW_DIR` na primeira célula de código (ou defina a variável de ambiente `CGMACROS_RAW`). As fotos NÃO são usadas.

Ao terminar, o terminal mostra (entre outras coisas) a tabela **sensores × ponta de dedo** e a **fonte dos METs**. Resultado conferido no dataset real: 45 participantes, 43 elegíveis (IDs 7 e 18 ficam de fora por não terem D1–D9), treino/validação/teste = 25/9/9. Na primeira execução, `wide` tinha 170 features e `day_level` 96; a versão atual descarta as constantes no treino (22 e 1), e a execução conferida no dataset real ficou com **148** e **95** features.

## Como conferir se está funcionando
1. **Código, sem o dataset (~1 min):** `python tests\test_preprocessing.py` deve terminar com `TODOS OS TESTES PASSARAM`. Ele gera dados sintéticos, roda o pipeline (inclusive pela linha de comando) e confere: glicose limpa, features recalculadas à mão, alvo = glicose do D9, ausência de vazamento treino/teste e `preprocessor.pkl` carregável.
2. **Dataset real:** rode o pipeline (comando acima). Valores esperados no terminal: 45 participantes; tabela sensores × ponta de dedo com `midpoint` e `pipeline_glucose` de viés ≈ 0 (Libre negativo, Dexcom positivo); `Fonte de METs: {'METs': 34, 'estimated_from_calories': 11}`; `Elegíveis: 43/45 | treino=25 valid=9 teste=9`; `[wide] 148 features (22 constantes no treino e 0 vazias descartadas)`; `[day_level] 95 features (1 constantes ...)`.
3. **Verificador automático das saídas:** `python tools\check_outputs.py` (lê `data/processed`, não precisa do dataset). Cada linha é `[OK]`, `[INFO]`, `[AVISO]` (vale olhar) ou `[ERRO]` (algo errado; o script sai com código 1). Confere série por minuto, D1–D9, alvo recalculado da série, divisão sem repetir participante, ausência de vazamento (nada de D7–D9 glicose nem alvo como entrada), `X` sem faltantes e se `preprocessor.pkl` reproduz `X_train`. Os `[AVISO]` esperados no dataset real: macros corrigidos nos IDs 16 e 19, poucas refeições nos IDs 32 e 42 e alguns valores de feature fora do padrão em validação/teste (a linha lista quais).
4. **Notebooks:** execute `01` e `02` com "Run All" (ajuste `RAW_DIR` na primeira célula). Se alguma célula falhar, o erro aparece na própria célula.

## Estrutura
| Arquivo | Para quê |
|---|---|
| `preprocessing.py` | Todo o pipeline (funções reutilizáveis + linha de comando) |
| `notebooks/01_exploracao_dados.ipynb` | Explora arquivos, colunas, missing, sensores (+ validação com ponta de dedo e A1c), refeições, grupos |
| `notebooks/02_preprocessamento.ipynb` | Pipeline passo a passo e geração da base em `data/processed/` |
| `tools/make_synthetic_data.py` | Gera dados SINTÉTICOS no formato do CGMacros (inclui as variações reais de colunas; só para teste) |
| `tests/test_preprocessing.py` | Testes de sanidade (sem vazamento, alvo correto, limpeza, sensores, METs) |
| `tools/check_outputs.py` | Verificador automático da base gerada em `data/processed` (rode depois do pipeline) |
| `model.py` | MLP em PyTorch (treino, avaliação, salvar/carregar, previsão) — Integrante 2 |
| `notebooks/03_rede_neural.ipynb` | Treina a rede em `day_level`, avalia e salva em `models/` |
| `tests/test_model.py` | Testes de sanidade da rede (treino aprende, avaliação correta, save/load) |
| `data/DataDictionary_*.csv` | Dicionários de dados oficiais |

## Arquivos gerados em `data/processed/`
| Arquivo | Conteúdo |
|---|---|
| `minute_clean.csv.gz` | Série limpa por minuto: `glucose` (consenso Libre+Dexcom), `libre_gl`, `dexcom_gl`, `hr`, `mets`, macros, `day` (1..n) |
| `daily_features.csv` | 1 linha por participante×dia: **comportamento** + **resposta glicêmica**; coluna `split` = `input` (D1–D6) / `eval` (D7–D9) / `extra` |
| `input_D1_D6.csv`, `eval_D7_D9.csv` | `daily_features` já separado em entrada e avaliação |
| `meals.csv` | 1 linha por refeição: macros, horário, glicose basal, pico, Δpico, iAUC 2 h |
| `participants.csv` | 1 linha por participante: bio, grupo (`group_a1c`), `eligible`, `reason_ineligible` |
| `dataset_wide.csv` | **Formulação do projeto**: resumo D1–D6 + bio + comportamento D7/D8/D9 → `target_d9_gl_mean` |
| `dataset_day_level.csv` | Versão com mais linhas (~8 por participante): dia *t* + dia *t−1* + bio → `target_gl_mean` |
| `splits.json` | IDs de treino/validação/teste (por participante, estratificado por grupo) |
| `model_ready/{wide,day_level}/` | `X_*.csv` (escalado), `y_*.csv`, `ids_*.csv`, `preprocessor.pkl`, `feature_columns.json`, `dropped_features.json` (colunas descartadas por serem constantes/vazias no treino) |
| `quality_report.csv` | Qualidade por participante: faltantes, valores fora de faixa, `dexcom_minus_libre_median`, `mets_source`, `kcal_per_met`, `capped_<nutriente>` |
| `fingerstick_validation.csv`, `fingerstick_summary.csv` | Cada sensor × glicemia de ponta de dedo do `bio.csv` (viés, MAE, RMSE) |
| `config.json` | Parâmetros usados na execução |

## Para o Integrante 2 (rede neural)
```python
import pandas as pd, preprocessing as P
X_train = pd.read_csv("data/processed/model_ready/wide/X_train.csv")
y_train = pd.read_csv("data/processed/model_ready/wide/y_train.csv")   # glicose média em D9 (mg/dL)
prep = P.load_preprocessor("data/processed/model_ready/wide/preprocessor.pkl")
X_new = prep.transform(df_novo)   # escala dados novos com o MESMO scaler do treino
```
- `wide` tem ~40 linhas (1 por participante): pouco para uma MLP. `day_level` tem ~8× mais linhas; sugestão: treinar nele e usar a mesma função para estimar D9. Alinhem qual usar.
- Não faça novo `train_test_split` por linha: use `splits.json` (por participante).
- `prep.transform(df, clip=5)` limita os valores escalados a ±5 desvios. Vale usar se algum participante de validação/teste for atípico (no dataset real, o ID 30 tem café da manhã ~60 % do padrão e gera z ≈ −16 em `hist_breakfast_calories/carbs`) e, no AG, para impedir que a busca explore valores muito fora do que a rede viu no treino.
- **Implementação (`model.py` + `notebooks/03_rede_neural.ipynb`):** MLP em PyTorch com early stopping na validação. O alvo é padronizado internamente (`MLPRegressor.y_mean/y_std`, fixados no treino) — sem isso o MSE inicial é dominado pela escala bruta da glicose (~120 mg/dL) e a rede não sai do chute "prever a média". `M.train_model(...)` devolve o modelo + histórico de perda; `M.evaluate(...)` dá MAE/RMSE/R²; `M.save_model`/`M.load_model` persistem pesos (`.pt`) e arquitetura (`.json`) em `models/`; `M.predict(model, df_novo, prep)` é a interface que o Integrante 3 usa para avaliar planos do AG. Treinado em `day_level` com `hidden_sizes=(32,)`, `dropout=0.3`, `weight_decay=1e-2` (escolhidos por busca em grade no MAE de validação — rede pequena + regularização forte generaliza melhor com ~212 linhas de treino). Resultado: MAE ≈ 9,6 mg/dL / R² ≈ 0,72 em validação, MAE ≈ 11,9 mg/dL / R² ≈ 0,80 em teste.

## Para o Integrante 3 (algoritmo genético)
Os genes naturais são as colunas de `P.BEHAVIOR_FEATURES` de cada dia (`d7_*`, `d8_*`, `d9_*` no `wide`): macros por refeição, horário das refeições, minutos ativos etc.
Limites realistas para cada gene: use `daily_features.csv` (mín/máx/percentis). Passe os planos por `prep.transform(...)` antes de chamar a rede.
**Atenção:** só as colunas de `model_ready/<tabela>/feature_columns.json` chegam à rede. No `wide`, café da manhã e parte do almoço de D7–D9 foram descartados por serem constantes no treino (ver "Refeições do protocolo" abaixo), então o AG só consegue "recomendar" o que o modelo enxerga: jantar, lanches, horários e atividade.

## Decisões de projeto (alteráveis por parâmetro)
- **Glicose (`--glucose consensus`, padrão).** Libre e Dexcom discordam: no dataset real o Dexcom lê, em média, **31,95 mg/dL acima** do Libre (correlação minuto a minuto 0,83). Comparando com as pontas de dedo do `bio.csv` (32 medições em que os dois sensores já estavam ativos), o Libre leu **15,9 mg/dL abaixo** e o Dexcom **16,8 acima**; a média dos dois ficou a +0,5 mg/dL (erro absoluto médio: Libre 21,6, Dexcom 20,0, média 9,6). Segunda checagem, que não depende do horário: a glicose média de cada participante contra a esperada pela A1c (`eAG = 28,7·A1c − 46,7`) deu viés de −20,5 (Libre), +11,5 (Dexcom) e −4,3 mg/dL (`consensus`), com correlação 0,81, 0,81 e 0,85. Por isso o `consensus` calcula, por participante, o deslocamento mediano `Dexcom − Libre` e usa `(Libre + Dexcom)/2`; nos minutos em que só um sensor existe, ele é deslocado para o mesmo ponto médio (sem "degraus" quando um sensor some). Alternativas: `libre_filled` (Libre + Dexcom corrigido onde falta), `mean` (média simples), `libre`, `dexcom`. Valores fora de 40–400 viram NaN; só buracos ≤ 30 min são interpolados; depois do alinhamento a glicose é limitada a 40–400 mg/dL. **Atenção:** TIR/TAR/TBR (cortes 70/180) dependem muito da fonte escolhida.
  Conferência com todos os participantes: `fingerstick_summary.csv` e seção 10 do notebook 01 (também compara com a glicose média esperada pela A1c, `eAG = 28,7·A1c − 46,7`).
- **Dias (`--day-mode first_full`, padrão).** O registro começa no meio do dia (~9–10 h), então o 1º dia de calendário é parcial. D1 = primeiro dia com glicose em ≥ 80 % do dia, o que deixa D1–D9 completos (cabem 9 dias inteiros em ~10 dias). Minutos anteriores a D1 ficam com `day=0` só em `minute_clean.csv.gz`. Alternativas: `calendar` (D1 = primeira data, parcial) e `from_start` (blocos de 24 h; quebra refeições, não recomendado).
- **Atividade: `METs` × `Intensity`.** 34 arquivos têm `METs` (guardado ×10, o código divide por 10). 11 arquivos (IDs 11, 26, 27, 31, 32, 33, 34, 35, 38, 42, 46) têm `Intensity` e **não** têm `METs`; para eles os METs são **estimados** a partir de `Calories (Activity)` (o gasto por minuto é proporcional aos METs; 1 MET = 1º percentil das calorias positivas do participante). Conferido em arquivos reais: no ID 1 (`METs`), `Calories (Activity)` = 1,0484 × METs em todos os minutos; no ID 27 (`Intensity`), as calorias são múltiplos exatos de 1,0884 (o gasto em repouso), como esperado de METs×10 inteiros; `Intensity` é um indicador por minuto (no ID 27 só 0/1, valendo 1 quando METs ≳ 2,4). Minutos com 0 kcal contam como ausência de dado. O ID 7 tem `METs`, mas não tem `Calories (Activity)` (tem `Steps` e `RecordIndex` extras) e é inelegível por ter ~7 dias. A origem de cada participante está em `quality_report.csv` (`mets_source` = `METs` / `estimated_from_calories` / `missing`). A validação do estimador (em quem tem os dois) está no notebook 01, seção 9.
- **Atividade (relógio).** Se o HR cobre < 50 % do dia (`activity_coverage`), as features de atividade do dia ficam NaN (imputadas depois pela mediana do treino); senão `active_min` e `cal_activity_sum` são normalizados para o dia inteiro. Alguns participantes têm 30–50 % de HR faltando.
- **Colunas por nome, não por posição.** Os CSVs não têm todos o mesmo cabeçalho (coluna de índice `Unnamed: 0`, ordem Libre/Dexcom trocada, `Amount Consumed` ausente, `Sugar` extra, `Intensity` no lugar de `METs`). O leitor padroniza tudo por nome e preenche com NaN o que não existe. Resumo por arquivo: notebook 01, seção 1.
- **Refeições do protocolo e features constantes.** Segundo a [página do dataset no PhysioNet](https://physionet.org/content/cgmacros/1.0.0), os cafés da manhã eram *shakes* proteicos com quantidades variadas de carboidrato, proteína, gordura e fibra, os almoços eram pedidos de uma mesma rede de restaurantes (Chipotle), e só os jantares eram livres. Isso aparece nos dados: no `wide`, 22 features eram **constantes no treino**: 21 de café da manhã e almoço (todos os macros de D7 e D9, parte de D8, o almoço de D7/D9 e as médias `hist_breakfast_protein/fiber`), ou seja, todos os participantes de treino tinham exatamente a mesma refeição nesses dias, e `eth_other` (ninguém no treino). O `StandardScaler` deixa colunas constantes em unidade bruta (escala 1), então qualquer valor diferente em validação/teste, ou num plano do AG, viraria algo como z = −14,7 ou +250. Por isso `fit_preprocessor` **descarta** colunas constantes (e vazias) no treino e as lista em `dropped_features.json` e em `prep.meta`. No `day_level` o café e o almoço **variam entre os dias do estudo** (igual para todos), então é ali que o efeito da composição dessas refeições pode ser aprendido.
- **Lanches:** `snack`, `snacks` e `snack 1` viram `snack`. Há 0..n por dia; entram nos totais e em `snack_*` e `n_snacks`. Café, almoço e jantar têm 1 linha cada.
- **Erros de digitação nos macros** (ex.: 2830 g de fibra): acima de `MEAL_VALUE_CAPS` (calorias 2500, carbo 300, proteína 200, gordura 200, fibra 100 por refeição) viram faltantes e são imputados pela mediana do mesmo tipo de refeição do participante. Contagem por participante e por nutriente em `quality_report.csv` (`meal_values_capped`, `capped_calories`, `capped_carbs`, …). Ajuste os tetos em `preprocessing.py` se o grupo preferir.
- **Elegível:** tem D1–D9 e D9 com ≥ 80 % de glicose.
- **Nutrientes:** vêm só na linha de início da refeição; nas demais linhas valem 0.
- **bio:** LDL=800, VLDL=400 e Cho/HDL=400 → NaN (erros documentados). A1c tratada como % (se > 20, converte de mmol/mol). `group_a1c`: <5.7 saudável, 5.7–6.5 pré-diabetes, ≥6.5 DM2 (reproduz 15/16/14 participantes; o dataset original tem seu próprio rótulo).

## Achados no dataset real (já tratados no código)
- Registro de ~10 dias começando no meio do dia; o ID 7 tem só ~7 dias e o ID 18 tem um buraco de ~9 dias: ambos ficam inelegíveis. Um participante tem ~19 dias (os dias além de D9 ficam como `extra`).
- `Meal Type` tem lanches com 3 grafias; macros com erros de digitação e `Amount Consumed` até 900 (limitado a 100).
- Datas deslocadas (2019–2025) por anonimização: só importa a ordem dos dias.
- Libre já vem interpolado em 1 min (leituras reais a cada 15 min); o Dexcom só começa ~75 min depois do arquivo e tem buracos.

## Pontos em aberto (nenhum impede o uso da base)
A escolha do `consensus` está confirmada pelas duas checagens (pontas de dedo e A1c).
1. IDs **16 e 19**: fibra de ~2200 g em um jantar (provável erro de unidade); o valor é imputado. Veja `capped_fiber`.
2. IDs **32 e 42**: só ~20–22 refeições em ~10 dias (possível sub-registro). Considere excluí-los se prejudicarem o modelo.
3. ID 30: café da manhã com ~60 % do padrão do estudo (339 kcal contra ~567 no treino). Confira se é porção menor ou `Amount Consumed` baixo; hoje os macros são usados como estão no arquivo, e o dicionário os descreve como "da refeição consumida". O ID 27 tem lanche com 13 g de fibra em D7 (valor raro, não erro).
4. METs estimados pelas calorias nos 11 participantes com `Intensity`: a proporcionalidade foi conferida nos IDs 1 e 27; o notebook 01 (seção 9) repete a checagem em todos os participantes que têm `METs` (esperado: correlação ≈ 1).
