"""
Proyecto #3 - CC3085 Inteligencia Artificial
Clasificador SPAM/HAM con Redes Bayesianas
Estudiante: Mia Alejandra Fuentes Mérida - 23775
"""

import re
import string
import random
import warnings
from collections import Counter
from math import prod

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from wordcloud import WordCloud
from nltk.stem import PorterStemmer

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# Estilo
plt.rcParams.update({
    "figure.facecolor": "#F9F9F9",
    "axes.facecolor": "#F9F9F9",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

SPAM_COLOR = "#E74C3C"
HAM_COLOR  = "#2ECC71"

STOPWORDS_EN = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m",
    "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn",
    "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
    "weren", "won", "wouldn", "u", "ur", "r", "ok", "im", "got", "get", "go", "come",
    "know", "like", "said", "call", "later", "lol", "oh", "yeah", "hey", "hi", "hello",
    "yes", "no", "ok", "na", "da", "la", "n", "gt", "lt", "amp"
}

stemmer = PorterStemmer()
PUNCT   = set(string.punctuation)


# SECCION 1: PREPROCESAMIENTO 
def preprocess(text: str) -> list[str]:
    """Tokeniza, normaliza, elimina stopwords y aplica stemming. Devuelve lista."""
    tokens = re.findall(r"\b[a-zA-Z]+\b", str(text))
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if not all(c in PUNCT for c in t)]
    tokens = [t for t in tokens if t not in STOPWORDS_EN]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens


def top_words(texts, n=20, stopwords=None):
    all_words = []
    for text in texts:
        tokens = re.findall(r"\b[a-zA-Z]+\b", str(text).lower())
        if stopwords:
            tokens = [w for w in tokens if w not in stopwords]
        all_words.extend(tokens)
    return Counter(all_words).most_common(n)


# SECCION 2: CARGA Y EDA 
print("\n" + "=" * 72)
print("SECCIÓN 1 - CARGA Y EXPLORACIÓN DE DATOS (EDA)")
print("=" * 72)

df = pd.read_csv("spam_ham.csv", sep=";", encoding="latin-1")
df.columns = ["label", "text"]
df["text"]  = df["text"].fillna("").astype(str)
df["label"] = df["label"].str.strip().str.lower()
df["length"] = df["text"].apply(len)
df["tokens"] = df["text"].apply(preprocess)
df["text_clean"] = df["tokens"].apply(lambda t: " ".join(t))
df["length_clean"] = df["text_clean"].apply(len)

total   = len(df)
n_spam  = (df["label"] == "spam").sum()
n_ham   = (df["label"] == "ham").sum()
pct_spam = n_spam / total * 100
pct_ham  = n_ham  / total * 100

spam_texts = df[df["label"] == "spam"]["text"]
ham_texts  = df[df["label"] == "ham"]["text"]
spam_len   = df[df["label"] == "spam"]["length"]
ham_len    = df[df["label"] == "ham"]["length"]
spam_clean = df[df["label"] == "spam"]["text_clean"]
ham_clean  = df[df["label"] == "ham"]["text_clean"]
spam_len_c = df[df["label"] == "spam"]["length_clean"]
ham_len_c  = df[df["label"] == "ham"]["length_clean"]

print(f"Total de mensajes : {total}")
print(f"Spam              : {n_spam} ({pct_spam:.1f}%)")
print(f"Ham               : {n_ham}  ({pct_ham:.1f}%)")

# EDA — antes del preprocesamiento
fig = plt.figure(figsize=(18, 22))
fig.suptitle("EDA ANTES DEL PREPROCESAMIENTO", fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

ax0 = fig.add_subplot(gs[0, 0])
bars = ax0.bar(["Ham", "Spam"], [n_ham, n_spam],
               color=[HAM_COLOR, SPAM_COLOR], edgecolor="white", width=0.5)
for bar, val in zip(bars, [n_ham, n_spam]):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f"{val:,}", ha="center", fontsize=11, fontweight="bold")
ax0.set_title("Distribución de mensajes"); ax0.set_ylabel("Cantidad")

ax1 = fig.add_subplot(gs[0, 1])
ax1.pie([n_ham, n_spam], labels=["Ham", "Spam"],
        colors=[HAM_COLOR, SPAM_COLOR], autopct="%1.1f%%", startangle=140)
ax1.set_title("Proporción Spam vs Ham")

ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(spam_len, bins=40, color=SPAM_COLOR, alpha=0.75, edgecolor="white", density=True)
spam_len.plot.kde(ax=ax2, color=SPAM_COLOR, linewidth=2)
ax2.set_title("Densidad longitud SPAM"); ax2.set_xlabel("Longitud"); ax2.set_ylabel("Densidad")

ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(ham_len, bins=40, color=HAM_COLOR, alpha=0.75, edgecolor="white", density=True)
ham_len.plot.kde(ax=ax3, color=HAM_COLOR, linewidth=2)
ax3.set_title("Densidad longitud HAM"); ax3.set_xlabel("Longitud"); ax3.set_ylabel("Densidad")

words_spam = top_words(spam_texts, 20, stopwords=STOPWORDS_EN)
words_s, counts_s = zip(*words_spam)
ax4 = fig.add_subplot(gs[2, 0])
ax4.barh(list(reversed(words_s)), list(reversed(counts_s)),
         color=SPAM_COLOR, alpha=0.85, edgecolor="white")
ax4.set_title("Top 20 palabras SPAM"); ax4.set_xlabel("Frecuencia")

words_ham = top_words(ham_texts, 20, stopwords=STOPWORDS_EN)
words_h, counts_h = zip(*words_ham)
ax5 = fig.add_subplot(gs[2, 1])
ax5.barh(list(reversed(words_h)), list(reversed(counts_h)),
         color=HAM_COLOR, alpha=0.85, edgecolor="white")
ax5.set_title("Top 20 palabras HAM"); ax5.set_xlabel("Frecuencia")

ax6 = fig.add_subplot(gs[3, 0])
wc_spam = WordCloud(width=800, height=400, background_color="white",
                    colormap="Reds", stopwords=STOPWORDS_EN,
                    max_words=100, collocations=False).generate(" ".join(spam_texts))
ax6.imshow(wc_spam, interpolation="bilinear"); ax6.axis("off"); ax6.set_title("WordCloud SPAM")

ax7 = fig.add_subplot(gs[3, 1])
wc_ham = WordCloud(width=800, height=400, background_color="white",
                   colormap="Greens", stopwords=STOPWORDS_EN,
                   max_words=100, collocations=False).generate(" ".join(ham_texts))
ax7.imshow(wc_ham, interpolation="bilinear"); ax7.axis("off"); ax7.set_title("WordCloud HAM")

plt.savefig("eda_antes_preprocesamiento.png", dpi=150, bbox_inches="tight")
plt.close()
print("OK: eda_antes_preprocesamiento.png")

# EDA — despues del preprocesamiento
fig2 = plt.figure(figsize=(18, 22))
fig2.suptitle("EDA DESPUÉS DEL PREPROCESAMIENTO", fontsize=15, fontweight="bold", y=0.98)
gs2 = gridspec.GridSpec(4, 2, figure=fig2, hspace=0.55, wspace=0.35)

ax0 = fig2.add_subplot(gs2[0, 0])
bars = ax0.bar(["Ham", "Spam"], [n_ham, n_spam],
               color=[HAM_COLOR, SPAM_COLOR], edgecolor="white", width=0.5)
for bar, val in zip(bars, [n_ham, n_spam]):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f"{val:,}", ha="center", fontsize=11, fontweight="bold")
ax0.set_title("Distribución de mensajes (sin cambio)"); ax0.set_ylabel("Cantidad")

ax1 = fig2.add_subplot(gs2[0, 1])
ax1.pie([n_ham, n_spam], labels=["Ham", "Spam"],
        colors=[HAM_COLOR, SPAM_COLOR], autopct="%1.1f%%", startangle=140)
ax1.set_title("Proporción Spam vs Ham")

ax2 = fig2.add_subplot(gs2[1, 0])
ax2.hist(spam_len_c, bins=40, color=SPAM_COLOR, alpha=0.75, edgecolor="white", density=True)
spam_len_c.plot.kde(ax=ax2, color=SPAM_COLOR, linewidth=2)
ax2.set_title("Densidad longitud SPAM (preprocesado)"); ax2.set_xlabel("Longitud")

ax3 = fig2.add_subplot(gs2[1, 1])
ax3.hist(ham_len_c, bins=40, color=HAM_COLOR, alpha=0.75, edgecolor="white", density=True)
ham_len_c.plot.kde(ax=ax3, color=HAM_COLOR, linewidth=2)
ax3.set_title("Densidad longitud HAM (preprocesado)"); ax3.set_xlabel("Longitud")

words_spam_c = top_words(spam_clean, 20)
words_sc, counts_sc = zip(*words_spam_c)
ax4 = fig2.add_subplot(gs2[2, 0])
ax4.barh(list(reversed(words_sc)), list(reversed(counts_sc)),
         color=SPAM_COLOR, alpha=0.85, edgecolor="white")
ax4.set_title("Top 20 palabras SPAM (preprocesado)"); ax4.set_xlabel("Frecuencia")

words_ham_c = top_words(ham_clean, 20)
words_hc, counts_hc = zip(*words_ham_c)
ax5 = fig2.add_subplot(gs2[2, 1])
ax5.barh(list(reversed(words_hc)), list(reversed(counts_hc)),
         color=HAM_COLOR, alpha=0.85, edgecolor="white")
ax5.set_title("Top 20 palabras HAM (preprocesado)"); ax5.set_xlabel("Frecuencia")

ax6 = fig2.add_subplot(gs2[3, 0])
wc_s2 = WordCloud(width=800, height=400, background_color="white",
                  colormap="Reds", max_words=100, collocations=False).generate(" ".join(spam_clean))
ax6.imshow(wc_s2, interpolation="bilinear"); ax6.axis("off"); ax6.set_title("WordCloud SPAM (preprocesado)")

ax7 = fig2.add_subplot(gs2[3, 1])
wc_h2 = WordCloud(width=800, height=400, background_color="white",
                  colormap="Greens", max_words=100, collocations=False).generate(" ".join(ham_clean))
ax7.imshow(wc_h2, interpolation="bilinear"); ax7.axis("off"); ax7.set_title("WordCloud HAM (preprocesado)")

plt.savefig("eda_despues_preprocesamiento.png", dpi=150, bbox_inches="tight")
plt.close()
print("OK: eda_despues_preprocesamiento.png")

print("\nComparativa de longitud promedio antes/después:")
for label, before_col, after_col in [("SPAM", spam_len, spam_len_c), ("HAM", ham_len, ham_len_c)]:
    m_b, m_a = before_col.mean(), after_col.mean()
    print(f"  {label}: {m_b:.1f} → {m_a:.1f} chars ({(1-m_a/m_b)*100:.1f}% reducción)")


# SECCIoN 3: SPLIT 80/20 
print("\n" + "=" * 72)
print("SECCIÓN 2 - SPLIT ENTRENAMIENTO (80%) / PRUEBA (20%)")
print("=" * 72)

df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx   = int(len(df_shuffled) * 0.8)
train_df    = df_shuffled.iloc[:split_idx].copy()
test_df     = df_shuffled.iloc[split_idx:].copy()

print(f"Entrenamiento: {len(train_df)} mensajes")
print(f"Prueba       : {len(test_df)} mensajes")


# SECCION 4: MODELO BAYESIANO 
print("\n" + "=" * 72)
print("SECCIÓN 3 - ENTRENAMIENTO DEL MODELO BAYESIANO")
print("=" * 72)

class BayesianSpamClassifier:
    """
    Clasificador Naive-Bayes basado en las fórmulas del Proyecto #3.

    Para cada palabra W:
        P(S|W) = P(W|S)·P(S) / [P(W|S)·P(S) + P(W|H)·P(H)]

    Para un texto con palabras W1…Wn (método de Robinson combinado):
        p = p1·p2·…·pn
        q = (1-p1)·(1-p2)·…·(1-pn)
        P(SPAM|texto) = p / (p + q)
    """

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing          # suavizado de Laplace
        self.p_spam: float = 0.0
        self.p_ham:  float = 0.0
        self.word_prob_spam: dict = {}      # P(S|W) para cada palabra
        self.p_w_given_spam: dict = {}      # P(W|S)
        self.p_w_given_ham:  dict = {}      # P(W|H)
        self.vocab: set = set()

    # Entrenamiento 
    def fit(self, df_train: pd.DataFrame):
        spam_docs = df_train[df_train["label"] == "spam"]["tokens"]
        ham_docs  = df_train[df_train["label"] == "ham"]["tokens"]

        total = len(df_train)
        n_s   = len(spam_docs)
        n_h   = len(ham_docs)

        self.p_spam = n_s / total
        self.p_ham  = n_h / total

        # Contar apariciones de palabras por DOCUMENTO (presencia, no frecuencia)
        spam_word_counts: Counter = Counter()
        ham_word_counts:  Counter = Counter()

        for tokens in spam_docs:
            for w in set(tokens):          # set → presencia única por mensaje
                spam_word_counts[w] += 1

        for tokens in ham_docs:
            for w in set(tokens):
                ham_word_counts[w] += 1

        self.vocab = set(spam_word_counts.keys()) | set(ham_word_counts.keys())

        # P(W|S) y P(W|H) con suavizado de Laplace
        V = len(self.vocab)
        for word in self.vocab:
            self.p_w_given_spam[word] = (spam_word_counts[word] + self.smoothing) / (n_s + self.smoothing * V)
            self.p_w_given_ham[word]  = (ham_word_counts[word]  + self.smoothing) / (n_h + self.smoothing * V)

        # Calcular P(S|W) para cada palabra (fórmula del proyecto)
        for word in self.vocab:
            p_ws = self.p_w_given_spam[word]
            p_wh = self.p_w_given_ham[word]
            numerator   = p_ws * self.p_spam
            denominator = numerator + p_wh * self.p_ham
            self.word_prob_spam[word] = numerator / denominator if denominator > 0 else 0.5

        print(f"  P(SPAM)        = {self.p_spam:.4f}")
        print(f"  P(HAM)         = {self.p_ham:.4f}")
        print(f"  Vocabulario    = {len(self.vocab):,} palabras únicas")
        print(f"  Suavizado      = Laplace (k={self.smoothing})")

    #  Probabilidad de SPAM para un texto 
    def predict_proba(self, text: str, unknown_prob: float = 0.5,
                      max_tokens: int = 15) -> tuple[float, list]:
        """
        Devuelve (P_spam, top3_palabras_predictorias).
        Fórmula combinada de Robinson:
            P = p1*p2*...*pn / (p1*p2*...*pn + (1-p1)*(1-p2)*...(1-pn))

        Se limita a las max_tokens palabras MÁS informativas (más alejadas de 0.5)
        para evitar que palabras ligeramente HAM colapsen la probabilidad a 0.
        """
        tokens = list(set(preprocess(text)))       

        # Asignar probabilidad a TODOS los tokens del mensaje
        # (los desconocidos reciben unknown_prob=0.5, neutro)
        all_token_probs = [(w, self.word_prob_spam.get(w, unknown_prob)) for w in tokens]

        # Filtrar palabras informativas (|p-0.5| > 0.05) y
        # ordenar por DISTANCIA a 0.5 — las más extremas primero
        # Limitar a max_tokens para evitar que palabras débilmente HAM
        # anulen palabras fuertemente SPAM en la multiplicación de Robinson
        probs_calc = sorted(
            [(w, p) for w, p in all_token_probs if abs(p - 0.5) > 0.05],
            key=lambda x: abs(x[1] - 0.5),
            reverse=True
        )[:max_tokens]

        # Top-3 siempre desde TODOS los tokens, ordenados por P(S|W) descendente
        top3 = sorted(all_token_probs, key=lambda x: x[1], reverse=True)[:3]

        if not probs_calc:
            return self.p_spam, top3

        p_vals = [p for _, p in probs_calc]
        # Producto numerador y denominador (log-space para evitar underflow)
        log_num = sum(np.log(p) for p in p_vals)
        log_den = sum(np.log(1 - p) for p in p_vals)

        # p / (p + q)  ->  1 / (1 + q/p)  ->  1 / (1 + exp(log_den - log_num))
        ratio = np.exp(log_den - log_num)
        spam_prob = 1.0 / (1.0 + ratio)

        return float(spam_prob), top3

    def predict(self, text: str, threshold: float = 0.5) -> str:
        prob, _ = self.predict_proba(text)
        return "spam" if prob >= threshold else "ham"


# Entrenar
clf = BayesianSpamClassifier(smoothing=1.0)
clf.fit(train_df)


# SECCION 5: EVALUACION
print("\n" + "=" * 72)
print("SECCIÓN 4 - PRUEBAS DE RENDIMIENTO")
print("=" * 72)

# Predecir probabilidades sobre el set de prueba
test_df["spam_prob"] = test_df["text"].apply(lambda x: clf.predict_proba(x)[0])
y_true = (test_df["label"] == "spam").astype(int).values

def evaluate_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (TP + TN) / len(y_true)
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

y_proba = test_df["spam_prob"].values

# Evaluar múltiples thresholds
thresholds = np.arange(0.1, 1.0, 0.05)
results    = [evaluate_threshold(y_true, y_proba, t) for t in thresholds]

print("\nRendimiento por threshold:")
print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 55)
best_f1, best_t, best_r = 0, 0.5, None
for t, r in zip(thresholds, results):
    marker = " ← mejor F1" if r["f1"] > best_f1 else ""
    if r["f1"] > best_f1:
        best_f1, best_t, best_r = r["f1"], t, r
    print(f"{t:>10.2f} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
          f"{r['recall']:>10.4f} {r['f1']:>10.4f}{marker}")

print(f"\nMejor threshold: {best_t:.2f}  →  F1={best_f1:.4f}")

# Matriz de confusion con el mejor threshold
r = best_r
print(f"\nMatriz de confusion (threshold={best_t:.2f}):")
print(f"                 Pred SPAM   Pred HAM")
print(f"  Real SPAM        {r['TP']:>5}       {r['FN']:>5}")
print(f"  Real HAM         {r['FP']:>5}       {r['TN']:>5}")
print(f"\n  Precision : {r['precision']:.4f}")
print(f"  Recall    : {r['recall']:.4f}")
print(f"  F1-score  : {r['f1']:.4f}")
print(f"  Accuracy  : {r['accuracy']:.4f}")


# SECCIO 6: GRAFICAS DE RENDIMIENTO 
fig3, axes = plt.subplots(1, 3, figsize=(18, 5))
fig3.suptitle("PRUEBAS DE RENDIMIENTO", fontsize=14, fontweight="bold")

# Curva Precision / Recall / F1 vs Threshold
precisions = [r["precision"] for r in results]
recalls    = [r["recall"]    for r in results]
f1s        = [r["f1"]        for r in results]
ax = axes[0]
ax.plot(thresholds, precisions, label="Precision", color="#3498DB", linewidth=2)
ax.plot(thresholds, recalls,    label="Recall",    color="#E67E22", linewidth=2)
ax.plot(thresholds, f1s,        label="F1-score",  color="#9B59B6", linewidth=2)
ax.axvline(best_t, color="gray", linestyle="--", alpha=0.7, label=f"Mejor (t={best_t:.2f})")
ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
ax.set_title("Métricas vs Threshold"); ax.legend()

# Matriz de confusion visual
cm_data = np.array([[r["TP"], r["FN"]], [r["FP"], r["TN"]]])
ax2 = axes[1]
im = ax2.imshow(cm_data, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm_data[i, j]), ha="center", va="center",
                 fontsize=16, fontweight="bold",
                 color="white" if cm_data[i, j] > cm_data.max()/2 else "black")
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
ax2.set_xticklabels(["Pred SPAM", "Pred HAM"])
ax2.set_yticklabels(["Real SPAM", "Real HAM"])
ax2.set_title(f"Matriz de Confusión\n(threshold={best_t:.2f})")

# Distribucion de probabilidades por clase
ax3 = axes[2]
spam_mask = test_df["label"] == "spam"
ax3.hist(test_df[spam_mask]["spam_prob"],  bins=30, alpha=0.7,
         color=SPAM_COLOR, label="SPAM", density=True)
ax3.hist(test_df[~spam_mask]["spam_prob"], bins=30, alpha=0.7,
         color=HAM_COLOR, label="HAM", density=True)
ax3.axvline(best_t, color="gray", linestyle="--", label=f"Threshold={best_t:.2f}")
ax3.set_xlabel("P(SPAM|texto)"); ax3.set_ylabel("Densidad")
ax3.set_title("Distribución de probabilidades"); ax3.legend()

plt.tight_layout()
plt.savefig("rendimiento_modelo.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nOK: rendimiento_modelo.png")

# Top palabras mas predictivas de SPAM
print("\nTop 15 palabras con mayor poder predictivo de SPAM:")
top_spam_words = sorted(clf.word_prob_spam.items(), key=lambda x: x[1], reverse=True)[:15]
for i, (w, p) in enumerate(top_spam_words, 1):
    print(f"  {i:>2}. '{w}'  →  P(S|W)={p:.4f}")


# ─── SECCION 7: MODULO INTERACTIVO 
print("\n" + "=" * 72)
print("SECCIÓN 5 - MODULO INTERACTIVO")
print("=" * 72)
print("Escribe un mensaje para clasificarlo. Escribe 'salir' para terminar.\n")

THRESHOLD = best_t   # usar el mejor threshold encontrado

while True:
    texto = input("Ingresa un mensaje: ").strip()
    if texto.lower() in {"salir", "exit", "quit"}:
        print("¡Hasta luego!")
        break
    if not texto:
        continue

    prob, top3 = clf.predict_proba(texto)
    clase = "SPAM" if prob >= THRESHOLD else "HAM"

    print(f"\n  Clasificación : {clase}")
    print(f"  P(SPAM)       : {prob:.4f}  ({prob*100:.1f}%)")
    print("  Top 3 palabras con mayor poder predictivo de SPAM:")
    for w, p in top3:
        print(f"    • '{w}'  ->  P(S|W) = {p:.4f}")
    print()


print("\n" + "=" * 72)
print("Archivos generados:")
print("  - eda_antes_preprocesamiento.png")
print("  - eda_despues_preprocesamiento.png")
print("  - rendimiento_modelo.png")
print("=" * 72)