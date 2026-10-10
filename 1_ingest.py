import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import re
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PDF_PATH      = "medical_book.pdf"
CHROMA_DIR    = "./chroma_db"
COLLECTION    = "medical_book"
CHUNK_SIZE    = 700  # characters
CHUNK_OVERLAP = 150
EMBED_MODEL   = "all-MiniLM-L6-v2"
# ──────────────────────────────────────────────────────────────────────────────


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text page-by-page, skipping TOC/Index/Bibliography pages."""
    doc = fitz.open(pdf_path)
    pages = []
    skip_keywords = ["table of contents", "bibliography", "references", "index"]

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        if not text or len(text) < 100:
            continue  # skip blank/near-blank pages

        # Skip pages that are clearly TOC/Index/Bibliography
        first_line = text[:80].lower()
        if any(kw in first_line for kw in skip_keywords):
            print(f"  Skipping page {page_num + 1} (TOC/Index/Bibliography)")
            continue

        pages.append({
            "text": text,
            "page_number": page_num + 1
        })

    print(f"✅ Extracted text from {len(pages)} pages (out of {len(doc)} total)")
    return pages


def recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks using sentence-aware splitting."""
    separators = ["\n\n", "\n", ". ", " "]
    chunks = []

    def split_with(text, seps):
        if len(text) <= chunk_size:
            return [text]
        sep = seps[0] if seps else " "
        parts = text.split(sep)
        current, result = "", []
        for part in parts:
            if len(current) + len(part) + len(sep) <= chunk_size:
                current += part + sep
            else:
                if current.strip():
                    result.append(current.strip())
                current = part + sep
        if current.strip():
            result.append(current.strip())
        return result

    raw_chunks = split_with(text, separators)
    # Apply overlap
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            chunks.append(chunk)
        else:
            # prepend tail of previous chunk
            prev_tail = chunks[-1][-overlap:] if len(chunks[-1]) > overlap else chunks[-1]
            chunks.append(prev_tail + " " + chunk)

    return chunks


def main():
    if not os.path.exists(PDF_PATH):
        print(f"❌ ERROR: '{PDF_PATH}' not found. Place your PDF in the same folder.")
        return

    print(f"\n📖 Reading: {PDF_PATH}")
    pages = extract_text_from_pdf(PDF_PATH)

    print(f"\n✂️  Chunking text (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    all_chunks, all_ids, all_metadata = [], [], []
    chunk_index = 0

    for page in pages:
        chunks = recursive_split(page["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < 50:
                continue
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_index}")
            all_metadata.append({"page_number": page["page_number"]})
            chunk_index += 1

    print(f"✅ Total chunks created: {len(all_chunks)}")

    print(f"\n🧠 Loading embedding model: {EMBED_MODEL} ...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("⚙️  Generating embeddings (this may take a few minutes)...")
    embeddings = embedder.encode(
        all_chunks, batch_size=64, show_progress_bar=True)
    embeddings = embeddings.tolist()


    print(f"\n💾 Storing in ChromaDB at: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete old collection if re-running
    try:
        client.delete_collection(COLLECTION)
        print("  (Old collection deleted — rebuilding fresh)")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    # Batch insert (ChromaDB has a batch limit)
    BATCH = 500
    for i in range(0, len(all_chunks), BATCH):
        collection.add(
            documents=all_chunks[i:i+BATCH],
            embeddings=embeddings[i:i+BATCH],
            ids=all_ids[i:i+BATCH],
            metadatas=all_metadata[i:i+BATCH]
        )
        print(f"  Inserted chunks {i} – {min(i+BATCH, len(all_chunks))}")

    print(f"\n🎉 INGESTION COMPLETE!")
    print(f"   Total chunks stored: {collection.count()}")
    print(f"   Database saved to  : {CHROMA_DIR}/")
    print(f"\n👉 Now run: python 2_app.py")


if __name__ == "__main__":
    main()
