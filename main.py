# =============================================================
# Trabalho Final — Disciplina de Inteligência Artificial
# Professor Munif Gebara Junior — Unicesumar 2026
# Integrante: Vinícius César Chaves Nunes — RA: 23000808-2
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')
import os
os.makedirs('graficos', exist_ok=True)

print("=" * 60)
print("  TRABALHO FINAL — INTELIGÊNCIA ARTIFICIAL — UNICESUMAR 2026")
print("  Vinícius César Chaves Nunes — RA: 23000808-2")
print("=" * 60)

# ---- 1. CARREGAMENTO DO DATASET ----
print("\n[1] Carregando dataset Breast Cancer Wisconsin...")
data = load_breast_cancer()
X, y = data.data, data.target
print(f"    Amostras: {X.shape[0]} | Features: {X.shape[1]}")
print(f"    Classes: {data.target_names}")

# ---- 2. DIVISÃO TREINO/TESTE ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
print(f"\n[2] Divisão: {len(X_train)} treino / {len(X_test)} teste")

# ---- 3. NORMALIZAÇÃO ----
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print("[3] Normalização aplicada (StandardScaler)")

# ---- 4. MODELO 1: KNN (Parte 1) ----
print("\n[4] Treinando KNN (K=5) — Parte 1...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s, y_train)
y_pred_knn = knn.predict(X_test_s)

print("\n--- Resultados KNN ---")
print(f"  Acurácia : {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"  Precisão : {precision_score(y_test, y_pred_knn):.4f}")
print(f"  Recall   : {recall_score(y_test, y_pred_knn):.4f}")
print(f"  F1-Score : {f1_score(y_test, y_pred_knn):.4f}")
print("\n  Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_knn))

# ---- 5. MODELO 2: RNA (Parte 2) ----
print("\n[5] Treinando RNA/MLP (64-32 neurônios) — Parte 2...")
rna = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                    random_state=42, early_stopping=True, validation_fraction=0.1)
rna.fit(X_train_s, y_train)
y_pred_rna = rna.predict(X_test_s)

print("\n--- Resultados RNA ---")
print(f"  Acurácia : {accuracy_score(y_test, y_pred_rna):.4f}")
print(f"  Precisão : {precision_score(y_test, y_pred_rna):.4f}")
print(f"  Recall   : {recall_score(y_test, y_pred_rna):.4f}")
print(f"  F1-Score : {f1_score(y_test, y_pred_rna):.4f}")
print(f"  Épocas   : {rna.n_iter_}")
print("\n  Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_rna))

# ---- 6. GRÁFICOS ----
print("\n[6] Gerando gráficos...")

C1, C2 = '#2563EB', '#DC2626'

# Gráfico 1: Matrizes de Confusão
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, cm, title in zip(axes,
    [confusion_matrix(y_test, y_pred_knn), confusion_matrix(y_test, y_pred_rna)],
    ['KNN (K=5)', 'RNA (MLP 64-32)']):
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predito'); ax.set_ylabel('Real')
    ax.set_xticks([0,1]); ax.set_xticklabels(['Maligno','Benigno'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['Maligno','Benigno'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.colorbar(im, ax=ax)
plt.suptitle('Matrizes de Confusão — KNN vs RNA', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('graficos/grafico1_matrizes_confusao.png', dpi=150, bbox_inches='tight')
plt.close()

# Gráfico 2: Comparação de métricas
metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
vals_knn = [accuracy_score(y_test,y_pred_knn), precision_score(y_test,y_pred_knn),
            recall_score(y_test,y_pred_knn), f1_score(y_test,y_pred_knn)]
vals_rna = [accuracy_score(y_test,y_pred_rna), precision_score(y_test,y_pred_rna),
            recall_score(y_test,y_pred_rna), f1_score(y_test,y_pred_rna)]
x = np.arange(len(metricas)); width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x-width/2, vals_knn, width, label='KNN', color=C1, alpha=0.85)
ax.bar(x+width/2, vals_rna, width, label='RNA (MLP)', color=C2, alpha=0.85)
for i, (v1, v2) in enumerate(zip(vals_knn, vals_rna)):
    ax.text(i-width/2, v1+0.005, f'{v1:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(i+width/2, v2+0.005, f'{v2:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_ylim(0.85, 1.05); ax.set_xticks(x); ax.set_xticklabels(metricas)
ax.set_title('Comparação de Métricas — KNN vs RNA', fontsize=14, fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('graficos/grafico2_comparacao_metricas.png', dpi=150, bbox_inches='tight')
plt.close()

# Gráfico 3: Curva de perda RNA
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(rna.loss_curve_, color=C2, linewidth=2.5, label='Treino')
ax.set_xlabel('Época'); ax.set_ylabel('Perda (Loss)')
ax.set_title('Curva de Perda — RNA (MLP)', fontsize=14, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('graficos/grafico3_curva_perda_rna.png', dpi=150, bbox_inches='tight')
plt.close()

# Gráfico 4: KNN variação K
ks = range(1, 21)
accs = [accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k).fit(X_train_s,y_train).predict(X_test_s)) for k in ks]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(ks), accs, marker='o', color=C1, linewidth=2.5, markersize=6)
ax.axvline(x=5, color='gray', linestyle='--', alpha=0.6, label='K=5 (escolhido)')
ax.set_xlabel('Valor de K'); ax.set_ylabel('Acurácia')
ax.set_title('KNN — Acurácia por Valor de K', fontsize=14, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('graficos/grafico4_knn_variacao_k.png', dpi=150, bbox_inches='tight')
plt.close()

print("    Gráficos salvos em graficos/")
print("\n[OK] Execução concluída com sucesso!")
