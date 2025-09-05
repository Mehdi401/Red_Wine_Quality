# Red_Wine_Quality[README (1).md](https://github.com/user-attachments/files/22181287/README.1.md)

# Red Wine Quality — Multi‑Task Learning (Keras)

A clean, copy‑paste‑ready README describing the end‑to‑end notebook you provided (`Red_Wine_Quality.ipynb`).  
This project explores the classic **Red Wine Quality** dataset and builds a **multi‑task neural network** to predict:

1) **Quality label** (High vs Low)  
2) **Alcohol content** (Low / Medium / High)  
3) **Citric acid level** (Low / High)

---

## 📦 Dataset

- **Source:** UCI Machine Learning Repository — _Wine Quality (Red)_  
- **Shape in notebook:** `1599 rows × 12 columns`
- **Columns (as used/seen in the notebook):**
  - `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`,  
    `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`, `quality`

> A short feature glossary (from the notebook):
>
> - **fixed acidity:** most acids involved with wine are non‑volatile  
> - **volatile acidity:** amount of acetic acid  
> - **citric acid:** amount of citric acid  
> - **residual sugar:** sugar left after fermentation  
> - **chlorides:** amount of salt  
> - **free sulfur dioxide / total sulfur dioxide:** preservative with antimicrobial & antioxidant properties  
> - **density:** how tightly matter is packed  
> - **pH:** acidity/basicity (most wines are ~3–4)  
> - **alcohol:** % alcohol content  
> - **quality:** target score (integer)

---

## 🎯 Problem Setup

### Targets engineered in the notebook
- **`quality_label`** (binary): High vs Low  
  ```python
  df['quality_label'] = df['quality'].apply(lambda x: 'Low' if x in [3, 4, 5] else 'High')
  ```
- **`alcohol_content`** (3 classes): Low / Medium / High  
  ```python
  bins   = [df['alcohol'].min()-1, 9.75, 11, df['alcohol'].max()+1]
  labels = ['Low', 'Medium', 'High']
  df['alcohol_content'] = pd.cut(df['alcohol'], bins=bins, labels=labels)
  ```
- **`citric_acid_binary`** (binary): Low (≤ 0.26) vs High (> 0.26)  
  ```python
  df['citric_acid_binary'] = (df['citric acid'] > 0.26).astype(int)  # 0=Low, 1=High
  ```

### Features used for modeling
To avoid leakage from target‑like columns, the model uses **9 numeric features** (excludes `citric acid`, `alcohol`, and `quality`):  
`['fixed acidity', 'volatile acidity', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates']`

---

## 🔍 EDA (as performed in the notebook)

- **Correlation heatmap** across numeric features.
- **Pairplot** colored by `quality`.
- **Class balance plots** for `quality`, `quality_label`, `alcohol_content`, and `citric_acid_binary`.

*(The notebook renders these plots with Seaborn/Matplotlib.)*

---

## 🧹 Preprocessing

- **Label encoding** of non‑numeric targets (`quality_label`, `alcohol_content`) via `LabelEncoder`.
  - Note: `LabelEncoder` maps classes alphabetically (e.g., `'High'→0`, `'Low'→1`, `'Medium'→2`).
- **Standardization** with `StandardScaler` on the feature matrix.
  - In the notebook the scaler is fit on the **entire dataset** before the train/test split (this is a mild **data‑leakage** risk).  
    *Future improvement:* fit the scaler **only on the training split** via a `Pipeline`/`ColumnTransformer`.

### Outlier exploration (IQR method)
The notebook includes functions to **count** and **replace** outliers (IQR=Q3−Q1; bounds=Q1−1.5·IQR, Q3+1.5·IQR) and prints the counts **before vs after** mean‑replacement.

**Outliers (before → after) — key columns:**  
- `residual sugar`: **126 → 11**  
- `chlorides`: **87 → 33**  
- `sulphates`: **55 → 23**  
- `fixed acidity`: **41 → 24**  
- `density`: **35 → 6**  
- `pH`: **28 → 6**  
- `total sulfur dioxide`: **45 → 26**  
- `volatile acidity`: **19 → 4**  
- `alcohol`: **12 → 0**  
- `quality`: **27 → 0**  
- `citric acid`: **1 → 0**  
- `free sulfur dioxide`: **26 → 0**

> **Important:** The modeling section continues with the original `df` (the outlier‑replaced copy `df_replaced` is **not** used downstream).

---

## 🧠 Model — Multi‑Task Neural Network (Keras Functional API)

Shared trunk on 9 standardized features → **three output heads**:

```python
input_layer = Input(shape=(X_train.shape[1],), name='Input')
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
x = BatchNormalization(); x = Dropout(0.2)
x = Dense(64,  activation='relu', kernel_regularizer=l2(0.001))
x = BatchNormalization(); x = Dropout(0.2)
x = Dense(32,  activation='leaky_relu', kernel_regularizer=l2(0.001))
x = BatchNormalization(); x = Dropout(0.2)
x = Dense(16,  activation='relu', kernel_regularizer=l2(0.001))
x = BatchNormalization(); x = Dropout(0.5)

# Heads
quality_head      = Dense(1, activation='sigmoid', name='quality')
alcohol_head      = Dense(3, activation='softmax', name='alcohol')
citric_acid_head  = Dense(1, activation='sigmoid', name='citric_acid')
```

**Compile / Train**
```python
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss={
        'quality': 'binary_crossentropy',
        'alcohol': 'sparse_categorical_crossentropy',
        'citric_acid': 'binary_crossentropy'
    },
    metrics={'quality':'accuracy','alcohol':'accuracy','citric_acid':'accuracy'}
)

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train,
    {'quality': y_quality_train, 'alcohol': y_alcohol_train, 'citric_acid': y_citric_acid_train},
    validation_split=0.2,
    epochs=150,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=0
)
```

---

## 📊 Results (on the held‑out test set)

**Overall test accuracy by task**
- **Quality (High vs Low):** **0.74**
- **Alcohol content (3‑class):** **0.74**
- **Citric acid (Low vs High):** **0.87**

**Detailed reports (excerpted from notebook):**

**Quality (binary) — Confusion Matrix**
```
[[102  35]
 [ 35 100]]
```
*(balanced performance; precision/recall/f1 ≈ 0.74 for both classes)*

**Alcohol content (3‑class) — Classification report**
```
precision  recall  f1-score   support
0           0.86    0.84      0.85         68
1           0.70    0.77      0.73         95
2           0.69    0.64      0.66        109
accuracy                         0.74        272
```
> Using `LabelEncoder`, class indices follow alphabetical order:  
> `0='High'`, `1='Low'`, `2='Medium'`.

**Citric acid (binary) — Classification report & Confusion Matrix**
```
accuracy: 0.87, 0-class recall: 0.91, 1-class recall: 0.83
Confusion Matrix:
[[124  12]
 [ 23 113]]
```

---

## 🛠️ Environment & Dependencies

- Python, NumPy, Pandas, Scikit‑learn
- TensorFlow / Keras
- Matplotlib, Seaborn

**Install (example):**
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

---

## 🚀 How to Reproduce

1. Clone/download this repo and open the notebook `Red_Wine_Quality.ipynb`.
2. Make sure `winequality-red.csv` is available (path used in the notebook: `/content/drive/MyDrive/DL/winequality-red.csv`).  
   - Or update the `pd.read_csv(...)` path to where the CSV is stored on your machine.
3. Run the cells top‑to‑bottom. The notebook:
   - Performs EDA and class engineering (`quality_label`, `alcohol_content`, `citric_acid_binary`)
   - Standardizes features
   - Splits data into train/test (`test_size=0.2`, `random_state=42`)
   - Trains the multi‑task model with early stopping
   - Prints classification reports and confusion matrices

---

## ✅ Design Choices & Notes

- **Multi‑task setup** leverages shared structure to learn from related targets, potentially improving generalization.
- **No leakage features**: `alcohol`, `citric acid`, and `quality` are excluded from the input features.
- **Regularization**: L2 weight decay (`0.001`) + BatchNorm + Dropout to reduce overfitting.
- **Caveat**: StandardScaler fit on full data; prefer fitting on training split only.
- **Outliers**: IQR‑based analysis included; mean‑replacement was **exploratory** and not used in final training.





