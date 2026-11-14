import sys
from pathlib import Path
from importlib.util import spec_from_file_location
from importlib.util import module_from_spec

PROJECT_ROOT = Path(
    r"D:\CMT_PROJECT"
)

SCRIPTS_DIR = (
    PROJECT_ROOT /
    "scripts"
)

RAG_SCRIPT = (
    SCRIPTS_DIR /
    "12_rag.py"
)



def load_rag_module():

    if not RAG_SCRIPT.exists():

        raise FileNotFoundError(
            "\nStage 12 RAG file was not found.\n\n"
            f"Expected:\n{RAG_SCRIPT}\n"
        )

    spec = spec_from_file_location(
        "cmt_stage12_rag",
        str(RAG_SCRIPT)
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            "Unable to load Stage 12 RAG module."
        )

    module = module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module



def retrieve_evidence(
    query,
    top_k=5
):
    """
    Retrieve literature evidence using Stage 12 TF-IDF RAG.

    Parameters
    ----------
    query : str
        User research query.

    top_k : int
        Number of literature chunks to retrieve.

    Returns
    -------
    list
        Retrieved literature results.
    """

    query = str(
        query
    ).strip()

    if not query:

        return []


    rag_module = load_rag_module()


   
    chunks_file = getattr(
        rag_module,
        "CHUNKS_FILE",
        None
    )

    vectorizer_file = getattr(
        rag_module,
        "VECTORIZER_FILE",
        None
    )

    matrix_file = getattr(
        rag_module,
        "MATRIX_FILE",
        None
    )


    if (
        chunks_file is not None
        and
        vectorizer_file is not None
        and
        matrix_file is not None
    ):

        if not (
            Path(chunks_file).exists()
            and
            Path(vectorizer_file).exists()
            and
            Path(matrix_file).exists()
        ):

            raise FileNotFoundError(
                "\nRAG database has not been created yet.\n\n"
                "Run Stage 12 first:\n"
                "python 12_rag.py\n"
            )


   
    if not hasattr(
        rag_module,
        "retrieve"
    ):

        raise AttributeError(
            "Stage 12 RAG module does not contain "
            "the expected retrieve() function."
        )


    results = rag_module.retrieve(
        query,
        top_k=top_k
    )


    return results



def format_evidence(
    results
):
    """
    Convert RAG results into a clean text format.
    """

    if not results:

        return "No relevant literature was retrieved."


    output = []


    for number, result in enumerate(
        results,
        start=1
    ):

        source = result.get(
            "source",
            "Unknown source"
        )

        chunk_id = result.get(
            "chunk_id",
            "N/A"
        )

        score = result.get(
            "score",
            0.0
        )

        text = result.get(
            "text",
            ""
        )


        output.append(
            f"""
Evidence {number}

Source:
{source}

Chunk:
{chunk_id}

Similarity:
{float(score):.4f}

Text:
{text}
""".strip()
        )


    return "\n\n" + "\n\n".join(
        output
    )



if __name__ == "__main__":

    print()
    print("=" * 70)
    print("CMT-SCOPE AI — RAG ENGINE TEST")
    print("=" * 70)
    print()

    query = (
        "Charcot-Marie-Tooth disease "
        "CMT1A gait pattern disease severity "
        "and clinical assessment"
    )

    try:

        results = retrieve_evidence(
            query,
            top_k=5
        )

        print(
            "Retrieved:",
            len(results),
            "evidence chunks"
        )

        print()

        for number, result in enumerate(
            results,
            start=1
        ):

            print(
                f"{number}. "
                f"{result.get('source', 'Unknown')}"
            )

            print(
                "   Chunk:",
                result.get(
                    "chunk_id",
                    "N/A"
                )
            )

            print(
                "   Similarity:",
                round(
                    float(
                        result.get(
                            "score",
                            0
                        )
                    ),
                    4
                )
            )

            print(
                "   Preview:",
                result.get(
                    "text",
                    ""
                )[:250]
            )

            print()


        print("=" * 70)
        print("RAG ENGINE TEST COMPLETE")
        print("=" * 70)

    except Exception as error:

        print(
            "ERROR:",
            error
        )