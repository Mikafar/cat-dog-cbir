# Cat vs Dog CBIR + KNN

Content-based image retrieval: query a cat/dog image, get the top-5 most similar /
least similar images from a 320-image database, and a KNN (k=5, cosine) classification.
Also evaluates on 10 unseen pets.

## Run locally

Requires only Python 3.9+ — one command sets up everything:

```bash
git clone https://github.com/Mikafar/cat-dog-cbir.git
cd cat-dog-cbir
python start_site.py                 # creates .venv, installs deps, starts site on :8501
python start_site.py 8503            # optionally pick a different port
```

(On Windows, use `py start_site.py`.) First run downloads torch + model weights.

## Website features

- **Tab 1 — Query**: 🎲 sample a database image or upload your own cat/dog photo;
  shows top-5 similar / top-5 dissimilar images + KNN prediction with confidence.
- **Tab 2 — Unseen test**: randomly draws 10 unseen images and shows the KNN
  predictions with failed-case visualization.

## Developer

- `prepare_data_and_train.py`: download images, encode with ResNet50, train KNN,
  save `cbir_knn_data.pkl`.
- `app.py`: Streamlit website.