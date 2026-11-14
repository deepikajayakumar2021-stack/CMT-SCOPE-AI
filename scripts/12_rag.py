import os
import pickle
import re

import numpy as np

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(
    r"D:\CMT_PROJECT"
)

LITERATURE_DIR = (
    PROJECT_ROOT /
    "literature"
)

RAG_DIR = (
    PROJECT_ROOT /
    "rag"
)

RAG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CHUNKS_FILE = (
    RAG_DIR /
    "chunks.pkl"
)

VECTORIZER_FILE = (
    RAG_DIR /
    "vectorizer.pkl"
)

MATRIX_FILE = (
    RAG_DIR /
    "tfidf_matrix.pkl"
)


#pdf
def extract_pdf_text(pdf_path):

    from pypdf import PdfReader

    reader = PdfReader(
        str(pdf_path)
    )

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)



#  TEXT CLEANING


def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()



#  CHUNKING


def create_chunks(
    text,
    chunk_size=1200,
    overlap=200
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk = " ".join(
            words[start:end]
        )

        if len(chunk) > 100:

            chunks.append(
                chunk
            )

        start += (
            chunk_size -
            overlap
        )

    return chunks



#  BUILD KNOWLEDGE BASE


def build_rag_database():

    pdf_files = sorted(
        LITERATURE_DIR.glob(
            "*.pdf"
        )
    )

    if not pdf_files:

        raise FileNotFoundError(
            "\nNo PDF files found.\n"
            f"Expected literature folder:\n"
            f"{LITERATURE_DIR}\n"
        )

    all_chunks = []

    successful_pdfs = 0

    for pdf_file in pdf_files:

        print(
            "Reading:",
            pdf_file.name
        )

        try:

            text = extract_pdf_text(
                pdf_file
            )

            text = clean_text(
                text
            )

            if not text:

                print(
                    "WARNING: No extractable text:",
                    pdf_file.name
                )

                continue

            chunks = create_chunks(
                text
            )

            if not chunks:

                print(
                    "WARNING: No usable chunks:",
                    pdf_file.name
                )

                continue

            successful_pdfs += 1

            for index, chunk in enumerate(
                chunks
            ):

                all_chunks.append({

                    "source":
                        pdf_file.name,

                    "chunk_id":
                        index,

                    "text":
                        chunk
                })

            print(
                "Chunks created:",
                len(chunks)
            )

        except Exception as error:

            print(
                "ERROR reading:",
                pdf_file.name
            )

            print(
                error
            )


    if not all_chunks:

        raise RuntimeError(
            "No usable text was extracted from the PDFs."
        )


    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    texts = [
        item["text"]
        for item in all_chunks
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000
    )

    matrix = vectorizer.fit_transform(
        texts
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    with open(
        CHUNKS_FILE,
        "wb"
    ) as f:

        pickle.dump(
            all_chunks,
            f
        )

    with open(
        VECTORIZER_FILE,
        "wb"
    ) as f:

        pickle.dump(
            vectorizer,
            f
        )

    with open(
        MATRIX_FILE,
        "wb"
    ) as f:

        pickle.dump(
            matrix,
            f
        )


    print()
    print("=" * 70)
    print("RAG DATABASE")
    print("=" * 70)

    print()

    print(
        "PDF files found:",
        len(pdf_files)
    )

    print(
        "PDF files processed:",
        successful_pdfs
    )

    print(
        "Total chunks:",
        len(all_chunks)
    )

    print(
        "Vocabulary size:",
        len(
            vectorizer.vocabulary_
        )
    )

    print()

    print(
        "Saved:",
        CHUNKS_FILE
    )

    print(
        "Saved:",
        VECTORIZER_FILE
    )

    print(
        "Saved:",
        MATRIX_FILE
    )


# ============================================================
# 6. RETRIEVAL
# ============================================================

def retrieve(
    query,
    top_k=5
):

    with open(
        CHUNKS_FILE,
        "rb"
    ) as f:

        chunks = pickle.load(
            f
        )

    with open(
        VECTORIZER_FILE,
        "rb"
    ) as f:

        vectorizer = pickle.load(
            f
        )

    with open(
        MATRIX_FILE,
        "rb"
    ) as f:

        matrix = pickle.load(
            f
        )


    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        matrix
    )[0]


    indices = np.argsort(
        similarities
    )[::-1][:top_k]


    results = []

    for index in indices:

        results.append({

            "source":
                chunks[index]["source"],

            "chunk_id":
                chunks[index]["chunk_id"],

            "score":
                float(
                    similarities[index]
                ),

            "text":
                chunks[index]["text"]
        })


    return results


# ============================================================
# 7. RETRIEVAL TEST
# ============================================================

def test_retrieval():

    query = (
        "Charcot-Marie-Tooth disease "
        "CMT1A gait pattern disease severity "
        "and clinical assessment"
    )

    results = retrieve(
        query,
        top_k=3
    )

    print()
    print("=" * 70)
    print("RAG RETRIEVAL TEST")
    print("=" * 70)

    print()

    for number, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{number}.",
            result["source"]
        )

        print(
            "   Chunk:",
            result["chunk_id"]
        )

        print(
            "   Similarity:",
            round(
                result["score"],
                4
            )
        )

        preview = result["text"][:300]

        print(
            "   Preview:",
            preview
        )

        print()


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    build_rag_database()

    test_retrieval()

    print("=" * 70)
    print("STAGE 12 COMPLETE")
    print("=" * 70)