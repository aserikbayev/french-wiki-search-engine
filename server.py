import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from sentence_transformers import CrossEncoder
from retriv import set_base_path, SparseRetriever
from flask import Flask, request, jsonify, render_template

set_base_path("./retriv_wd") # I prefer to have caches in project dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    title: str = ""

class SearchEngine:
    def __init__(self, 
                 index_name: str,
                 model_path: str,
                 batch_size: int = 32):
        self.batch_size = batch_size
        
        # Initialize reranker
        logging.info("Loading reranker model...")
        self.reranker = CrossEncoder(model_path, num_labels=1)

        # Initialize BM25 retriever
        logging.info("Loading BM25 index...")
        try:
            self.retriever = SparseRetriever.load(index_name)
        except FileNotFoundError:
            logging.info("Index not found, will create one now..")
            self.retriver = create_index(index_name)
        
    def _batch_rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[SearchResult]:
        """Rerank a batch of documents"""
        # Prepare input pairs for reranking
        pairs = [[query, doc['text']] for doc in documents]
        
        # Get reranker scores
        scores = self.reranker.predict(pairs)
        
        # Combine with document info and sort by score
        results = []
        for doc, score in zip(documents, scores):
            result = SearchResult(
                id=doc['id'],
                text=doc['text'],
                score=float(score),
                title=doc.get('title', '')
            )
            results.append(result)
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def search(self, query: str, top_k: int = 1000) -> List[SearchResult]:
        """
        Search for documents using BM25 and rerank results
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve and rerank
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        # Get initial BM25 results
        bm25_results = self.retriever.search(
            query=query,
            return_docs=True,
            cutoff=top_k
        )
        
        # Rerank top_k results with a cross encoder
        reranked_results = self._batch_rerank(query, bm25_results)
        
        return reranked_results

# Initialize Flask app
app = Flask(__name__)

# Need this to be a global variable
search_engine = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])
    
    results = search_engine.search(query)
    
    results_json = [
        {
            'id': r.id,
            'text': r.text[:300] + '...' if len(r.text) > 300 else r.text,
            'score': round(r.score, 3),
            'title': r.title
        }
        for r in results[:10] # get top 10
    ]
    
    return jsonify(results_json)

def create_index(index_name: str = "miracl-fr-bm25-index"):
    """Create BM25 index from collection"""
    logging.info(f"Creating index {index_name}...")

    import datasets
    collection = datasets.load_dataset("miracl/miracl-corpus", "fr", cache_dir="hf_datasets_cache")["train"]
    
    retriever = SparseRetriever(
        index_name=index_name,
        model="bm25",
        tokenizer="whitespace",
        stemmer="french",
        stopwords="french",
        do_lowercasing=True,
        do_ampersand_normalization=True,
        do_special_chars_normalization=True,
        do_acronyms_normalization=True,
        do_punctuation_removal=True,
    )
    
    retriever = retriever.index(
        collection,
        show_progress=True,
        callback=lambda doc: {
            "id": doc["docid"],
            "text": (doc.get("title", "") + ". " + doc["text"]).strip(),
            "title": doc.get("title", "")
        }
    )
    
    logging.info(f"Index {index_name} created successfully")
    return retriever

def init_app(index_name: str,
             model_path): # local path or huggingface id
    """Initialize the application"""
    global search_engine
    
    # # create index if missing
    # if not SparseRetriever.exists(index_name):
    #     import datasets
    #     corpus = datasets.load_dataset("miracl/miracl-corpus", "fr", cache_dir="hf_datasets_cache")["train"]
    #     create_index(corpus, index_name)

    # Initialize    
    search_engine = SearchEngine(index_name, model_path)
    
    return app


if __name__ == '__main__':
    app = init_app(
        index_name="miracl-fr-bm25-index",
        model_path="azat-serikbayev/crossencoder-camembert-base-mmarcoFR-miracl-fr")
    
    app.run(debug=True)