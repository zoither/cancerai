# Disciplina de Inteligência Artificial , Professor Munif , Unicesumar 2026

# Classificação de Tumores de Mama com KNN e Redes Neurais Artificiais

## Integrantes
- Vinícius César Chaves Nunes - RA: 23000808-2

---

## Resumo do Projeto

### Contextualização
O câncer de mama é uma das doenças mais prevalentes no mundo, e o diagnóstico precoce é fator determinante para o sucesso do tratamento. Técnicas de Inteligência Artificial podem auxiliar médicos na análise de biópsias, classificando tumores com alta precisão.

### Problema
Classificar automaticamente tumores de mama como **malignos** ou **benignos** com base em características clínicas extraídas de exames de biópsia.

### Hipótese
Ambos os modelos conseguem classificar tumores com alta acurácia, mas esperamos que a RNA apresente desempenho superior ao KNN por conseguir capturar relações não-lineares nos dados.

### Métodos utilizados
- **Parte 1:** KNN — K-Nearest Neighbors (K=5)
- **Parte 2:** RNA — Rede Neural Artificial / MLP (64-32 neurônios)

---

## Dataset

| Atributo | Valor |
|---|---|
| Nome | Breast Cancer Wisconsin Dataset |
| Origem | UCI Machine Learning Repository / scikit-learn |
| Total de amostras | 569 registros |
| Número de features | 30 atributos numéricos |
| Variável alvo | 0 = Maligno, 1 = Benigno |
| Divisão | 426 treino / 143 teste (75% / 25%) |
| Pré-processamento | StandardScaler (média=0, desvio=1) |

O dataset descreve características dos núcleos celulares de biópsias: raio, textura, perímetro, área, suavidade, compacidade, concavidade, simetria e dimensão fractal — cada atributo em 3 versões (média, erro padrão e pior valor).

**Obtenção:** disponível diretamente via `sklearn.datasets.load_breast_cancer()` — sem necessidade de download externo.

---

## Modelos

### KNN (Parte 1)
- K = 5 vizinhos
- Distância Euclidiana
- Dados normalizados com StandardScaler

### RNA — MLP (Parte 2)
- Camadas ocultas: 64 e 32 neurônios
- Ativação: ReLU
- Máx. iterações: 300 (com early stopping)

---

## Avaliação dos Modelos

| Métrica | KNN (K=5) | RNA (MLP) |
|---|---|---|
| Acurácia | 0.9790 | 0.9371 |
| Precisão | 0.9677 | 0.9451 |
| Recall | 1.0000 | 0.9556 |
| F1-Score | 0.9836 | 0.9503 |

### Gráficos

**Comparação de Métricas:**
![Gráfico Métricas](graficos/grafico2_comparacao_metricas.png)

**Matrizes de Confusão:**
![Matrizes de Confusão](graficos/grafico1_matrizes_confusao.png)

**Curva de Perda — RNA:**
![Curva de Perda](graficos/grafico3_curva_perda_rna.png)

**Variação de K — KNN:**
![Variação K](graficos/grafico4_knn_variacao_k.png)

---

## Comparação e Conclusão

O **KNN com K=5 superou a RNA** em todas as métricas. O Recall perfeito (1.0000) do KNN é especialmente relevante em diagnóstico médico: nenhum tumor maligno foi classificado incorretamente como benigno.

A RNA demonstrou convergência estável, mas para este dataset (569 amostras), o KNN foi mais eficiente. A RNA tende a superar o KNN em datasets maiores e mais complexos.

**Conclusão:** a hipótese inicial foi refutada — o KNN foi o melhor modelo para este problema.

---

## Como Executar

```bash
pip install scikit-learn matplotlib pandas numpy
python main.py
```

## Modelo Treinado
O modelo é treinado ao executar `main.py`. Por ser leve (KNN e MLP do scikit-learn), o treinamento ocorre em segundos e não requer download externo.

---

## Estrutura do Repositório

```
trabalho-final-ia/
├── main.py
├── graficos/
│   ├── grafico1_matrizes_confusao.png
│   ├── grafico2_comparacao_metricas.png
│   ├── grafico3_curva_perda_rna.png
│   └── grafico4_knn_variacao_k.png
├── README.md
├── relatorio.pdf
└── requirements.txt
```
