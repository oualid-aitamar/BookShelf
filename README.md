# 📚 Shelf — Book Recommendation System

> A content-based book recommender built with Python and NLP, with a standalone interactive interface.

---

## 🎯 Overview

**Shelf** is a book recommendation system that suggests similar books based on their title, subtitle, author, category, and description. It uses **TF-IDF vectorization** and **Cosine Similarity** to measure how close two books are in a high-dimensional text space.

The interface runs entirely in the browser — no server, no installation required.

---


## ⚙️ How It Works

1. Each book is converted into a numerical vector using **TF-IDF** on a combined text feature:
   ```
   features = title + subtitle + authors + categories + description
   ```

2. **Cosine Similarity** is computed between all book pairs (6,810 × 6,810 matrix)

3. The **top 8 most similar books** are pre-computed for every book and stored in a JSON index

4. The interface loads these pre-computed results → **instant recommendations**, no real-time computation

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn (TF-IDF, Cosine Similarity) |
| Frontend | HTML5, CSS3, JavaScript ES6+ |
| Dataset | 6,810 real books (title, author, category, description, ratings) |

---

## 📁 Project Structure

```
shelf/
├── recommender.py        # ML pipeline: preprocessing + TF-IDF + similarity
├── books_clean.csv       # Cleaned dataset with 6,810 books
├── book_app.html         # Standalone web interface
├── bdata.js              # Pre-computed recommendations (JSON)
└── README.md
```

---

## 🚀 Getting Started

### Run the ML pipeline

```bash
# Install dependencies
pip install pandas numpy scikit-learn

# Run the recommender script
python appr.py
```

This will:
- Clean and preprocess the dataset
- Build the TF-IDF matrix
- Compute similarities between all books
- Export `bdata.js` with pre-computed recommendations

### Launch the interface

No server needed. Simply open `book_app.html` in your browser.

> ⚠️ Make sure `book_app.html` and `bdata.js` are in the **same folder**.

---

## ✨ Features

- 🔍 **Live search** — results appear as you type (debounced at 180ms)
- 🏷️ **Genre filters** — dynamic category chips built from the dataset
- 📖 **Book detail page** — cover, description, metadata
- 🤝 **8 similar books** per title, recommended via cosine similarity
- 📱 **Responsive design** — works on all screen sizes
- ⚡ **No backend** — 100% standalone, runs directly in the browser

---

## 📊 Dataset

The dataset contains **6,810 books** with the following fields:

| Field | Description |
|-------|-------------|
| `title` | Book title |
| `subtitle` | Subtitle (if any) |
| `authors` | Author(s) |
| `categories` | Literary genre |
| `description` | Book description |
| `average_rating` | Average reader rating |
| `ratings_count` | Number of ratings |
| `thumbnail` | Cover image URL |
| `published_year` | Year of publication |
| `num_pages` | Number of pages |

---

## 📈 What I Learned

- **Feature engineering matters more than algorithm complexity** — combining multiple text fields (title + description + category) significantly improved recommendation quality compared to using a single field
- Pre-computing similarity results trades storage for speed — a practical optimization for static datasets
- Building a standalone app without a backend forces clean separation between data processing and presentation

---

## 🔮 Possible Improvements

- [ ] Add **Collaborative Filtering** based on user ratings
- [ ] Integrate a **Hybrid Recommender** (content + collaborative)
- [ ] Deploy online with a lightweight Python backend (FastAPI / Flask)
- [ ] Add user accounts and personal bookshelves

---


