# Mini IDP Project Tasks

## High-Level Objective (2024-07-26)

- [ ] Build a self-serve internal platform that enables developers and data scientists to upload documents or datasets, select a processing pipeline (e.g. RAG chatbot, summarizer), and run scalable, observable workflows using modular backend components.

---

## Level 4 Implementation Plan

### 1. Requirements Analysis

**Overall Project Requirements (derived from PRD):**
- User interface for document/data upload (PDF, CSV, JSON).
- Metadata logging for uploaded files.
- Selection of processing pipelines: RAG chatbot, PDF summarizer, Text classifier.
- Optional dynamic parameter configuration for pipelines.
- Secure REST APIs (upload, trigger pipeline, check status).
- Database storage for pipeline run metadata.
- UI portal to view uploads, launch/monitor jobs, display outputs.
- Workflow orchestration layer (n8n, Prefect, or Dagster).
- Asynchronous job queue (Celery + Redis or RQ).
- Basic logging and observability dashboard (job status, durations, logs).
- Dockerized pipelines with shared volume/workspace.
- JWT-based user authentication and basic RBAC.

**Stretch Goals:**
- Multi-user tenant architecture.
- Usage/resource tracking.
- Custom embedding/model selector.
- Document + pipeline versioning.

### 2. Components Affected (High-Level Mapping to Proposed Structure)

- **`app/` (FastAPI Backend):** Handles all API logic, database interactions, job queuing requests, authentication.
    - `routers/`: Endpoints for upload, pipeline management, status.
    - `services/`: Business logic for file handling, pipeline initiation, metadata management.
    - `models/`: Pydantic models for API requests/responses, DB schemas.
    - `core/config.py`: Application settings.
    - `main.py`: FastAPI app setup.
- **`workflows/`:** Contains logic for individual pipelines and orchestrator configurations.
    - `pipelines/`: Scripts/modules for RAG, summarizer, classifier.
    - `orchestrator_config/`: Configuration files for Prefect/n8n/Dagster.
- **`frontend/`:** User interface for all interactions.
- **`tests/`:** Unit and integration tests for backend and workflows.
- **Infrastructure Components:**
    - Database (SQLite/PostgreSQL)
    - Message Queue (Redis)
    - Workflow Orchestrator (Prefect/n8n/Dagster)
    - Docker & Docker Compose

### 3. Architecture Considerations

- **Modularity:** Each pipeline will be a distinct module, ideally containerized. The backend, frontend, and workflow orchestrator will be separate services.
- **Scalability:** Asynchronous processing via job queues is critical. The chosen orchestrator should support scaling workers.
- **Observability:** Structured logging and a clear way to track job status and history.
- **Security:** JWT for API security, proper input validation, and secure handling of data.
- **Technology Choices (to be finalized per phase):**
    - Workflow Orchestrator: Evaluate n8n, Prefect, Dagster based on ease of use, Python integration, and community support. *Initial Leaning: Prefect for Python-centric approach.*
    - Frontend: Streamlit for rapid prototyping, React for a more custom/complex UI. *Initial Leaning: Streamlit for MVP, then evaluate React.*

### 4. Implementation Strategy (Phased Approach)

**Phase 1: Core Backend & First Pipeline (MVP)**
  - **Goal:** Establish foundational backend, file upload, and one end-to-end processing pipeline with a simple interface.
  - **Tasks:**
    - [x] **P1.1: Project Setup & Basic FastAPI App** (2024-07-26)
        - [x] Create project directories based on agreed structure.
        - [x] Initialize `requirements.txt` (FastAPI, Uvicorn, Pydantic, SQLModel, pydantic-settings).
        - [x] Setup basic FastAPI app (`app/main.py`) with a health check endpoint.
        - [x] Configure `app/core/config.py` for basic settings.
    - [x] **P1.2: Document Upload API & Storage** (2024-07-26)
        - [x] Create `app/routers/upload.py` with a `/upload` endpoint.
        - [x] Implement `app/services/file_service.py` for handling file uploads (store locally for now).
        - [x] Add Pydantic models for upload request/response.
        - [x] Log basic metadata (filename, size, type, timestamp). (2024-07-26)
    - [x] **P1.3: Database Setup & Metadata Model** (2024-07-26)
        - [x] Choose DB (SQLite for MVP). Setup connection (`app/db/session.py`).
        - [x] Define `UploadedFileLog` (`app/models/file_models.py`) and `PipelineRun` (`app/models/pipeline_models.py`) SQLModels.
        - [x] Basic DB table creation on app startup (`app/main.py` calls `create_db_and_tables`).
    - [x] **P1.4: PDF Summarizer Pipeline - Logic** (2024-07-26)
        - [x] Create `workflows/pipelines/summarizer.py` (with `pypdf` and `sumy`).
        - [x] Implement PDF parsing (using `pypdf`).
        - [x] Implement summarization logic (using `sumy` with LSA).
        - [x] Define input (PDF path) and output (summary dict with status).
    - [x] **P1.5: Job Queue Setup (Celery + Redis or RQ)** (2024-07-26, Completed 2024-07-29)
        - [x] Integrate Celery with Redis. Configured `celery_app.py`, `config.py`, `requirements.txt`.
        - [x] Create a Celery task for PDF summarizer (`summarization_tasks.py`).
        - [x] Task updates `PipelineRun` status (QUEUED, PROCESSING, COMPLETED, FAILED) and stores Celery task ID.
        - [x] Debugging Celery task execution and argument passing. (2024-07-29)
    - [x] **P1.6: Pipeline Trigger & Status API** (Completed 2024-07-29)
        - [x] Create `app/routers/pipelines.py` with `/pipelines/trigger` and `/pipelines/{run_id}/status` endpoints.
        - [x] `app/services/pipeline_service.py` to create `pipeline_run` record, dispatch job.
        - [x] Correcting Celery task dispatch from `pipeline_service.py`. (2024-07-29)
    - [ ] **P1.7: Basic UI (Streamlit)**
        - [ ] Create `frontend/app.py` (Streamlit).
        - [ ] UI to upload a PDF.
        - [ ] UI to trigger summarizer pipeline for uploaded PDF.
        - [ ] UI to display job status and summary result.
    - [ ] **P1.8: Unit Tests for Core Components**
        - [ ] Tests for file upload service, pipeline trigger.
        - [ ] Test for summarizer (mocking external calls).

**Phase 2: Workflow Orchestration & Additional Pipelines**
  - **Goal:** Integrate a workflow orchestrator, add RAG and Text Classifier pipelines.
  - **Tasks:**
    - [ ] **P2.1: Select & Integrate Workflow Orchestrator** (e.g., Prefect)
        - [ ] Research and finalize choice (Prefect, n8n, Dagster).
        - [ ] Install and configure the orchestrator.
        - [ ] Refactor PDF Summarizer task to be managed by the orchestrator.
    - [ ] **P2.2: RAG Chatbot Pipeline**
        - [ ] `workflows/pipelines/rag_chatbot.py`
        - [ ] Document loading and chunking (LangChain).
        - [ ] Embedding generation (e.g., Sentence Transformers, OpenAI embeddings). *Creative Component: Choice of embedding model and vector store.*
        - [ ] Vector store setup (FAISS for local, consider alternatives).
        - [ ] Retrieval and generation logic (LangChain).
        - [ ] Integrate with orchestrator.
    - [ ] **P2.3: Text Classifier Pipeline**
        - [ ] `workflows/pipelines/text_classifier.py`
        - [ ] Define classification schema (rule-based or model-based). *Creative Component: Design of classification logic/model choice.*
        - [ ] Implement classification logic.
        - [ ] Integrate with orchestrator.
    - [ ] **P2.4: Update UI for New Pipelines**
        - [ ] Allow selection of RAG or Classifier.
        - [ ] UI for interacting with RAG (chat interface).
        - [ ] UI for displaying classification results.
    - [ ] **P2.5: Dynamic Pipeline Parameters (Optional)**
        - [ ] Allow users to set basic parameters (e.g., chunk size for RAG) via UI.
        - [ ] Pass parameters through API to orchestrator/pipelines.
    - [ ] **P2.6: Unit Tests for New Pipelines**

**Phase 3: UI/UX, Authentication & Basic Observability**
  - **Goal:** Enhance user experience, secure the platform, and add monitoring.
  - **Tasks:**
    - [ ] **P3.1: UI Enhancements** (If moving to React, this is a larger sub-project)
        - [ ] Improve layout, navigation.
        - [ ] Better job monitoring table/view.
        - [ ] Consistent styling.
    - [ ] **P3.2: JWT Authentication**
        - [ ] Implement token-based auth (e.g., `FastAPI-Users` or custom).

### Discovered During Work (2024-07-29)
- **Celery Task `AttributeError: 'str' object has no attribute 'hex'` (P1.5):** 
    - Problem: `run_uuid` (string) was used directly in SQLModel query within Celery task, expecting a `UUID` object.
    - Fix: Converted `run_uuid_str` to `uuid.UUID` object in `summarization_tasks.py` before querying.
- **Uvicorn Reload Issues (P1.5):**
    - Problem: Uvicorn failed to reload `summarization_tasks.py` correctly, citing `IndentationError` and `AttributeError` even after fixes were applied.
    - Fix: Full stop/start of Uvicorn server (without `--reload` initially) seemed to resolve. Potentially an issue with `WatchFiles` or caching.
- **Celery Task Dispatch Argument Mismatch (P1.5 & P1.6):**
    - Problem: `pipeline_service.py` called `summarize_pdf_task.delay()` with an unexpected keyword argument `pipeline_run_uuid` and was missing other required arguments.
    - Current Fix (Applied 2024-07-29): Corrected the arguments in `pipeline_service.py` to match the task signature (`run_uuid_str`, `uploaded_file_log_id`, `file_path`, `original_filename`).
- **Celery Prerun Signal Argument Handling (P1.5):**
    - Problem: `task_prerun` signal handler in `summarization_tasks.py` was not correctly extracting `run_uuid` when task was called with positional vs. keyword arguments.
    - Fix (Applied 2024-07-29): Updated signal handler to robustly check `args` and `kwargs` from the signal for the `run_uuid`.
- **Missing Table Metadata in Celery Worker (P1.5):**
    - Problem: `summarization_tasks.py` was missing an import for `UploadedFileLog`, causing `sqlalchemy.exc.NoReferencedTableError` in Celery worker context.
    - Fix (Applied 2024-07-29): Added `from app.models.file_models import UploadedFileLog` to `summarization_tasks.py`.
- **Missing NumPy Dependency for Sumy LSA (P1.4 & P1.5):**
    - Problem: `sumy.summarizers.lsa` requires `numpy`, which was not in `requirements.txt`.
    - Fix (Applied 2024-07-29): Added `numpy` to `requirements.txt`.

---