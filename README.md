# Scholar - RAG Research Assistant

Scholar is a highly accurate, locally-indexed Research Paper QA Chatbot. It is designed specifically to extract and synthesize information exclusively from uploaded PDFs, ensuring completely grounded answers with no hallucinations. 

## Features
- **Hybrid Retrieval System**: Fuses Dense Vector Search (ChromaDB) and Sparse Keyword Search (BM25) using Reciprocal Rank Fusion (RRF) for extreme accuracy on academic papers.
- **Precision Reranking**: Utilizes FlashRank to rerank the top retrieved chunks, ensuring only the most relevant context is fed into the LLM.
- **Premium User Interface**: An aesthetically pleasing, fully responsive Vanilla HTML/CSS/JS frontend featuring instant Light/Dark mode toggling and elegant micro-animations.
- **Persistent Conversations**: Uses an embedded SQLite database to save your chat sessions so you can easily pick up where you left off.
- **Strict Citations**: Every generated response includes the exact paper name, page number, and source snippet.

## Results & Example Interaction

Because the system uses **Reciprocal Rank Fusion** alongside **FlashRank reranking**, it can handle highly complex and specific academic queries by fetching the most relevant passages. 

**Example Query:** `"what is content moderation?"`

**Example Generated Response (with formatting & citations):**
> Content moderation refers to the process of analyzing and regulating content, such as text, images, and audio, to ensure it meets certain standards or guidelines. Based on the provided context, here is a comprehensive overview of content moderation:
> 
> * Traditional content moderation paradigms are becoming obsolete due to the evolution of modern enterprise workflows, which now heavily rely on a mix of audio, video, and image-based exchanges [1].
> * Effective content moderation requires a proactive approach, intercepting and validating content pre-delivery, rather than relying on reactive systems that flag content only after it appears in the chat stream [1].
> * Multimodal content moderation involves analyzing text, images, and audio in tandem to facilitate secure communication within a company or organization [2].
> 
> **Sources Cited:**
> 1. *Beyond_Text_Review_Paper.pdf* (Page 1)
> 2. *ConvoEase_camera-ready paper.pdf* (Page 1)

*The frontend dynamically renders this markdown with clear syntax highlighting, bullet points, and clickable source citation cards.*

## Screenshots

<div align="center">
  <img src="assets/frontend.png" alt="Scholar Frontend UI" width="800"/>
  <p><em>Premium Dark Mode Interface with conversational history and local indexed library.</em></p>
</div>

<br/>

<div align="center">
  <img src="assets/langfuse_trace.png" alt="Langfuse Trace" width="800"/>
  <p><em>Detailed Langfuse trace showing the Hybrid Retrieval, Reranking, and LLM Generation pipeline.</em></p>
</div>

## Architecture

```mermaid
flowchart TD
    %% Styling
    classDef frontend fill:#1E1E28,stroke:#C4FF3D,stroke-width:2px,color:#ECECF1
    classDef api fill:#2D7F3A,stroke:#1A1A2E,stroke-width:2px,color:#FFFFFF
    classDef storage fill:#5B3FD4,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    classDef process fill:#2C2C38,stroke:#8A8A99,stroke-width:1px,color:#ECECF1
    classDef llm fill:#FF4D4D,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF

    subgraph User Interaction
        UI[Premium Web Interface]:::frontend
    end

    subgraph FastAPI Backend
        API[REST API Endpoints]:::api
        
        %% Ingestion Flow
        subgraph Document Ingestion
            Parse[PDF Parsing & Cleaning]:::process
            Chunk[Sentence-aware Chunking]:::process
            Embed[MiniLM Embeddings]:::process
        end

        %% Storage Layer
        subgraph Storage Layer
            DB[(ChromaDB Vector Store)]:::storage
            BM[(BM25 Sparse Index)]:::storage
            SQL[(SQLite Conversations)]:::storage
        end
        
        %% Query Flow
        subgraph Hybrid Retrieval Pipeline
            Dense[Dense Semantic Search]:::process
            Sparse[Sparse Keyword Search]:::process
            RRF[Reciprocal Rank Fusion]:::process
            Rerank[FlashRank Reranking]:::process
        end

        %% Generation
        subgraph Response Generation
            Gen[Groq LLM Generation]:::llm
        end
    end

    %% Upload Flow
    UI -- "1. Upload PDFs" --> API
    API --> Parse
    Parse --> Chunk
    Chunk --> Embed
    Embed --> DB
    DB -. "Auto-syncs" .-> BM

    %% Query Flow
    UI -- "2. Ask Question" --> API
    API --> SQL
    API --> Dense
    API --> Sparse
    
    Dense -- "Vector Matches" --> RRF
    Sparse -- "Keyword Matches" --> RRF
    RRF --> Rerank
    Rerank -- "Top K Context" --> Gen
    
    Gen -- "Answer & Citations" --> API
    API -- "3. Display Results" --> UI
```

## Tech Stack
- **Frontend**: Vanilla HTML / CSS / JavaScript (No build steps required)
- **Backend**: FastAPI (Python)
- **Vector Database**: ChromaDB
- **Retrieval Engine**: `rank-bm25` (Sparse), `SentenceTransformers` (Dense), `FlashRank` (Reranking)
- **Language Model**: Groq API (`llama-3.3-70b-versatile`)
- **Conversation Storage**: SQLite

## Getting Started

1. **Clone the repository** and navigate to the project directory.
2. **Activate your environment** and install requirements:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install fastapi uvicorn python-multipart rank-bm25 chromadb sentence-transformers flashrank groq pymupdf pydantic
   ```
3. **Configure Environment Variables**: Create a `.env` file in the root of the project:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
4. **Start the Backend Server**:
   ```bash
   python -m uvicorn server:app --reload --port 8000
   ```
5. **Access the Application**: Open your web browser and navigate to `http://localhost:8000`
