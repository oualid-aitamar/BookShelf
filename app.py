from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle, os, warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ── Chargement des données ──────────────────────────────────────
print("📚 Chargement du dataset...")
BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE, 'books_clean.csv'))
df = df.fillna('')
df = df.reset_index(drop=True)

print("🔧 Construction TF-IDF...")
tfidf = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['features'])
print(f"✅ Prêt — {len(df)} livres indexés.")


def book_to_dict(row, score=None):
    d = {
        'isbn13': str(row.get('isbn13', '')),
        'title': row.get('title', ''),
        'subtitle': row.get('subtitle', ''),
        'authors': row.get('authors', ''),
        'categories': row.get('categories', ''),
        'thumbnail': row.get('thumbnail', ''),
        'description': row.get('description', '')[:400] + ('...' if len(str(row.get('description', ''))) > 400 else ''),
        'published_year': int(row.get('published_year', 0)) if str(row.get('published_year', '')).isdigit() else None,
        'average_rating': float(row.get('average_rating', 0)) if row.get('average_rating', '') != '' else None,
        'num_pages': int(row.get('num_pages', 0)) if str(row.get('num_pages', '')).replace('.0','').isdigit() else None,
        'ratings_count': int(float(row.get('ratings_count', 0))) if row.get('ratings_count', '') != '' else None,
    }
    if score is not None:
        d['similarity_score'] = round(float(score), 4)
    return d


# ── Routes ──────────────────────────────────────────────────────

@app.route('/api/search')
def search():
    """Recherche de livres par texte libre (title, author, category)."""
    q = request.args.get('q', '').strip().lower()
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 12))
    if not q:
        return jsonify({'results': [], 'total': 0})

    mask = (
        df['title'].str.lower().str.contains(q, na=False) |
        df['authors'].str.lower().str.contains(q, na=False) |
        df['categories'].str.lower().str.contains(q, na=False) |
        df['description'].str.lower().str.contains(q, na=False)
    )
    results = df[mask].copy()
    total = len(results)
    results = results.iloc[(page-1)*per_page : page*per_page]
    return jsonify({
        'results': [book_to_dict(row) for _, row in results.iterrows()],
        'total': total,
        'page': page,
        'per_page': per_page
    })


@app.route('/api/recommend/<isbn13>')
def recommend(isbn13):
    """Recommande des livres similaires basé sur TF-IDF + Cosine Similarity."""
    n = int(request.args.get('n', 8))
    matches = df[df['isbn13'].astype(str) == str(isbn13)]
    if matches.empty:
        return jsonify({'error': 'Livre non trouvé', 'recommendations': []})

    idx = matches.index[0]
    query_vec = tfidf_matrix[idx]
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    scores[idx] = 0  # exclure le livre lui-même

    top_indices = scores.argsort()[::-1][:n]
    recs = []
    for i in top_indices:
        recs.append(book_to_dict(df.iloc[i], score=scores[i]))

    return jsonify({
        'book': book_to_dict(df.iloc[idx]),
        'recommendations': recs
    })


@app.route('/api/recommend_by_text')
def recommend_by_text():
    """Recommande basé sur un texte libre (ex: description d'un livre imaginaire)."""
    text = request.args.get('text', '').strip()
    n = int(request.args.get('n', 8))
    if not text:
        return jsonify({'recommendations': []})

    query_vec = tfidf.transform([text])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:n]

    recs = [book_to_dict(df.iloc[i], score=scores[i]) for i in top_indices]
    return jsonify({'recommendations': recs})


@app.route('/api/book/<isbn13>')
def get_book(isbn13):
    matches = df[df['isbn13'].astype(str) == str(isbn13)]
    if matches.empty:
        return jsonify({'error': 'Non trouvé'}), 404
    return jsonify(book_to_dict(matches.iloc[0]))


@app.route('/api/stats')
def stats():
    return jsonify({
        'total_books': len(df),
        'total_authors': df['authors'].nunique(),
        'total_categories': df['categories'].nunique(),
        'avg_rating': round(pd.to_numeric(df['average_rating'], errors='coerce').mean(), 2),
        'top_categories': df['categories'].value_counts().head(10).to_dict()
    })


@app.route('/api/featured')
def featured():
    """Retourne des livres populaires."""
    n = int(request.args.get('n', 12))
    top = df[df['ratings_count'] != ''].copy()
    top['ratings_count_n'] = pd.to_numeric(top['ratings_count'], errors='coerce')
    top = top.dropna(subset=['ratings_count_n']).nlargest(n, 'ratings_count_n')
    return jsonify({'results': [book_to_dict(row) for _, row in top.iterrows()]})


if __name__ == '__main__':
    app.run(debug=True, port=5050)
