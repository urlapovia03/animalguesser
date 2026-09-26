# 🐾 Классификация изображений животных

## 📝 Описание проекта

Проект сравнивает несколько ранее обученных моделей классификации изображений (Практические работы №2–5), выбирает лучшую по метрикам качества, разворачивает её в виде REST API на **FastAPI** и предоставляет веб-интерфейс на **Streamlit** для загрузки или рисования изображения и получения предсказания с распределением вероятностей по классам.

## 📊 Датасет

- 5 классов животных: **butterfly, cow, elephant, sheep, squirrel**.
- Структура:
  - `train/<class>_train/*.jpg` — обучающая выборка, разложенная по папкам классов;
  - `test/*.jpg` + `Testing_set_animals.csv` — тестовые изображения. Колонка меток в CSV пустая (характерно для Kaggle-style датасетов, где тестовая часть предназначена для отдельного сабмишена).
- Поскольку истинные метки тестовой выборки недоступны, все метрики ниже посчитаны на отложенных 20% из `train` (`train_test_split`, `stratify` по классу, `random_state=42`).

## 🤖 Сравниваемые модели

| Модель | Практическая работа | Вход | Архитектура |
|---|---|---|---|
| Dense | ПР-2 | 128×128×3 (flatten) | Полносвязная сеть без свёрточных слоёв |
| CNN | ПР-3 | 128×128×3 | Свёрточная сеть |
| CNN+BN+Dropout | ПР-4 | 32×32×3 | Свёрточная сеть с BatchNormalization и Dropout |
| EfficientNetB4 | ПР-5 | 224×224×3 | Transfer learning на базе EfficientNetB4 (ImageNet) |

## 📈 Результаты сравнения

| Модель | Accuracy | Precision | Recall | F1 | Инференс (мс/изобр.) |
|---|---|---|---|---|---|
| **EfficientNetB4 (ПР-5)** | 0.9951 | 0.9951 | 0.9951 | **0.9951** | 1.030 |
| CNN+BN+Dropout (ПР-4) | 0.7759 | 0.7746 | 0.7759 | 0.7727 | 0.124 |
| CNN (ПР-3) | 0.7729 | 0.7838 | 0.7729 | 0.7700 | 0.390 |
| Dense (ПР-2) | 0.4335 | 0.4965 | 0.4335 | 0.4128 | 0.339 |

**Лучшая модель по F1-мере — EfficientNetB4 (ПР-5)**, сохранена как `best_classification_model.keras`.

> ⚠️ **Методологическое замечание.** Т.к. `Testing_set_animals.csv` не содержит реальных меток, метрики получены на отложенной части `train` (20%, `random_state=42`), а не на официальном тестовом наборе. Проверка на другом сиде (`random_state=123`) показала стабильность результата EfficientNetB4 (F1 = 0.9957), но заметный разброс у CNN+BN+Dropout (F1 от 0.7727 до 0.9203) — вероятный признак того, что при исходном обучении этой модели не было честного отложенного holdout. Метрики CNN+BN+Dropout стоит интерпретировать с этой оговоркой.

### Визуализации результатов

![Сравнение метрик качества](images/metrics_comparison.png)
![Матрицы ошибок](images/confusion_matrices.png)

> Как добавить эти картинки: в ноутбуке перед `plt.show()` для соответствующих графиков добавь `plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')` (и аналогично для матриц ошибок), скачай файлы и положи в папку `images/` этого репозитория.

## 🗂️ Структура проекта

Проект разделён на два репозитория:

**Backend (API):**
```
main.py                          — FastAPI-приложение
requirements.txt                 — зависимости backend
best_classification_model.keras  — сохранённая лучшая модель (EfficientNetB4)
```

**Frontend (интерфейс):**
```
app.py            — Streamlit-приложение
requirements.txt  — зависимости frontend
```

## 🚀 Развёрнутые сервисы

- **Backend / публичный API:** https://render-pr10.onrender.com
  - Swagger-документация: https://render-pr10.onrender.com/docs
- **Frontend / Streamlit-интерфейс:** https://frontend-pr10-ag9wbx58dtrz7zhbz3cj6n.streamlit.app/
- **GitHub — backend:** https://github.com/urlapovia03/backend-pr10
- **GitHub — frontend:** https://github.com/urlapovia03/frontend-pr10

## 🛠️ Локальное развёртывание

### 1. Backend (FastAPI)

```bash
git clone <ССЫЛКА_НА_BACKEND_РЕПОЗИТОРИЙ>
cd <папка-backend>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Проверка: открой http://localhost:8000 — должен вернуться `{"status": "ok", "model_loaded": true}`.

### 2. Frontend (Streamlit)

```bash
git clone <ССЫЛКА_НА_FRONTEND_РЕПОЗИТОРИЙ>
cd <папка-frontend>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Перед локальным запуском поправь в `app.py` адрес API на локальный:

```python
API_URL = "http://localhost:8000/predict"
```

Запуск:

```bash
streamlit run app.py
```

Приложение откроется на http://localhost:8501.

## 📡 Примеры использования API

**curl:**

```bash
curl -X POST "https://render-pr10.onrender.com/predict" \
  -F "file=@elephant.jpg"
```

**Python (requests):**

```python
import requests

with open("elephant.jpg", "rb") as f:
    response = requests.post(
        "https://render-pr10.onrender.com/predict",
        files={"file": f},
    )
print(response.json())
```

**Пример ответа:**

```json
{
  "predicted_class": "elephant",
  "confidence": 0.9987,
  "probabilities": {
    "butterfly": 0.0002,
    "cow": 0.0001,
    "elephant": 0.9987,
    "sheep": 0.0007,
    "squirrel": 0.0003
  }
}
```

## 🧰 Используемые технологии

- **Python**, **TensorFlow / Keras** — обучение и инференс моделей.
- **FastAPI**, **Uvicorn** — REST API.
- **Streamlit**, **streamlit-drawable-canvas** — веб-интерфейс.
- **scikit-learn** — метрики качества (accuracy, precision, recall, F1, confusion matrix).
- **OpenCV**, **Pillow** — обработка изображений.
- **Render.com** — хостинг API, **Streamlit Cloud** — хостинг интерфейса.
