# Mini IDP Project Tasks

## High-Level Objectives

### Primary Objective (2024-07-26)
- [x] Build a self-serve internal platform that enables developers and data scientists to upload documents or datasets, select a processing pipeline (e.g. RAG chatbot, summarizer), and run scalable, observable workflows using modular backend components.

### Data Science Extension Objective (2024-12-19)
- [ ] Extend the platform with comprehensive machine learning capabilities, enabling data enthusiasts to upload CSV/Excel datasets, configure ML problems (classification/regression), select algorithms with hyperparameters, and receive trained model results through an intuitive interface.

**Note:** See `TASK_DS.md` for detailed breakdown of data science feature tasks and implementation phases.

---

## Level 4 Implementation Plan

### 1. Requirements Analysis

**Overall Project Requirements (derived from PRD):**
- User interface for document/data upload (PDF, CSV, JSON, Excel).
- Metadata logging for uploaded files.
- Selection of processing pipelines: RAG chatbot, PDF summarizer, Text classifier, ML training.
- Optional dynamic parameter configuration for pipelines.
- Secure REST APIs (upload, trigger pipeline, check status).
- Database storage for pipeline run metadata.
- UI portal to view uploads, launch/monitor jobs, display outputs.
- Workflow orchestration layer (Prefect).
- Basic logging and observability dashboard (job status, durations, logs).
- Dockerized pipelines with shared volume/workspace.
- JWT-based user authentication and basic RBAC.

**Data Science Extension Requirements:**
- CSV/Excel dataset upload with data profiling and validation.
- ML problem type selection (classification, regression).
- Algorithm selection with configurable hyperparameters.
- Automated data preprocessing and feature engineering.
- Model training, evaluation, and comparison capabilities.
- Results visualization and model performance metrics.
- Model persistence and experiment tracking.

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
    - `pipelines/`: Scripts/modules for RAG, summarizer, classifier, ML training.
    - `ml/`: Machine learning specific modules (algorithm registry, preprocessing, evaluation).
    - `orchestrator_config/`: Configuration files for Prefect.
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
    - [x] **P1.7: Basic UI (React)** (Completed 2024-07-30)
        - [x] Initialize React app using Vite (`frontend/`). (2024-07-30)
        - [x] Scaffold folder structure (`components`, `pages`, `services`). (2024-07-30)
        - [x] Implement `FileUpload.jsx` component. (2024-07-30)
        - [x] Implement `UploadPage.jsx` page. (2024-07-30)
        - [x] Implement `api.js` service for upload. (2024-07-30)
        - [x] Setup basic routing with React Router for UploadPage. (2024-07-30)
        - [x] Debug initial rendering of the React app. (2024-07-30)
        - [x] Test PDF upload via React UI to backend successfully. (2024-07-30)
        - [x] UI to trigger summarizer pipeline for uploaded PDF. (2024-07-31)
        - [x] UI to display job status and summary result. (2024-07-31)
    - [x] **P1.8: Unit Tests for Core Components** (Completed 2024-07-31)
        - [x] Tests for file upload service (`app/services/file_service.py`).
        - [x] Tests for pipeline service (`app/services/pipeline_service.py`).
        - [x] Tests for upload router (`app/routers/upload.py`).
        - [x] Tests for pipelines router (`app/routers/pipelines.py`).
        - [x] Tests for PDF summarizer workflow logic (`workflows/pipelines/summarizer.py`).
        - [x] Tests for Celery summarization task and signals (`app/tasks/summarization_tasks.py`).
        - [x] All tests passing after debugging (2024-07-31).

**Phase 2: Workflow Orchestration with Prefect**
  - **Goal:** Integrate a workflow orchestrator, add RAG and Text Classifier pipelines.
  - **Tasks:**
    - [X] **P2.1: Integrate Prefect & Refactor Summarizer Pipeline**
        - [X] Install Prefect
        - [X] Refactor `workflows/pipelines/summarizer.py` to use Prefect `@task` and `@flow` decorators.
        - [X] Remove Celery task (`app/tasks/summarization_tasks.py`, `app/core/celery_app.py`) and dependencies (`celery`, `redis`).
        - [X] Update `app/services/pipeline_service.py` to call the Prefect flow synchronously.
        - [X] Update `app/models/pipeline_models.py` (rename `celery_task_id`, add `result`, `error_message`).
        - [X] Update `app/routers/pipelines.py` to use the synchronous service call and updated models.
        - [X] Update unit tests for `app/services/pipeline_service.py` (`test_pipeline_service.py`) to cover `trigger_pipeline_flow` for all pipeline types (Summarizer, RAG Ingestion, Text Classifier), including success, failure, and edge cases. (2024-08-01) - Completed 2024-08-01
    - [x] **P2.2: RAG Chatbot Pipeline**
        - [x] `workflows/pipelines/rag_chatbot.py` (2024-08-01)
        - [x] Document loading and chunking (LangChain). (2024-08-01)
        - [x] Embedding generation (e.g., Sentence Transformers, OpenAI embeddings). *Creative Component: Choice of embedding model and vector store.* (2024-08-01)
        - [x] Vector store setup (FAISS for local, consider alternatives). (2024-08-01)
        - [x] Retrieval and generation logic (LangChain). (2024-08-01)
        - [x] Integrate with orchestrator. (2024-08-01)
    - [x] **P2.3: Text Classifier Pipeline**
        - [x] `workflows/pipelines/text_classifier.py` (2024-08-01)
        - [x] Define classification schema (rule-based or model-based). (2024-08-01) - *Initial rule-based schema defined*
        - [x] Implement classification logic. (2024-08-01) - *Initial rule-based logic implemented*
        - [x] Integrate with orchestrator. (2024-08-01) - *Basic Prefect flow created*
        - [x] Fix test failures and ensure all tests are passing. (2024-08-02)
    - [x] **P2.4: Update UI for New Pipelines**
        - [x] Allow selection of RAG or Classifier. (2024-08-01) - *Added pipeline selector (Summarizer, RAG, Classifier) to UploadPage; API already supports type.*
        - [x] UI for interacting with RAG (chat interface). (2024-08-02) - *Created ChatPage with interactive chat interface for RAG documents.*
        - [x] UI for displaying classification results. (2024-08-02) - *Enhanced StatusPage to show formatted results for all pipeline types including Text Classification.*
    - [x] **P2.7: RAG Chatbot API Enhancements** (Completed 2024-08-03)
        - [x] Create dedicated router `app/routers/rag.py` for RAG-specific endpoints.
        - [x] Implement `app/services/rag_service.py` with question answering functionality.
        - [x] Add vector store management with list, check status, and deletion capabilities.
        - [x] Enhance `workflows/pipelines/vector_store_manager.py` with utilities for working with document-specific vector stores.
        - [x] Hook up endpoints to the main FastAPI application.
    - [ ] **P2.5: Dynamic Pipeline Parameters (Optional)**
        - [ ] Allow users to set basic parameters (e.g., chunk size for RAG) via UI.
        - [ ] Pass parameters through API to orchestrator/pipelines.
    - [x] **P2.6: Unit Tests for New Pipelines**
        - [x] Fix test failures in pipeline router tests (checking for file log existence). (2024-08-02)
        - [x] Fix test failures in pipeline service tests (unsupported pipeline type error message). (2024-08-02)
        - [x] Fix test failures in RAG core tests (LangChain chain invocation mocking). (2024-08-02)
        - [x] All 8 test failures resolved and tests passing. (2024-08-02)

**Phase 3: UI/UX, Authentication & Basic Observability**
  - **Goal:** Enhance user experience, secure the platform, and add monitoring.
  - **Tasks:**
    - [ ] **P3.1: UI Enhancements** (If moving to React, this is a larger sub-project)
        - [ ] Improve layout, navigation.
        - [ ] Better job monitoring table/view.
        - [ ] Consistent styling.
    - [ ] **P3.2: JWT Authentication**
        - [ ] Implement token-based auth (e.g., `FastAPI-Users` or custom).

**Phase 4: Asynchronous Operations & Enhanced UX (Date: <YYYY-MM-DD>)**
  - **Goal:** Improve platform responsiveness and user experience through asynchronous pipeline execution and UI modernization.
  - **Tasks:**
    - [ ] **P4.1: Asynchronous Pipeline Execution with Prefect**
        - [ ] Research and decide on Prefect asynchronous execution model (e.g., deployments with agent/work-queue, or simpler async flow invocation).
        - [ ] Refactor `app/services/pipeline_service.py` to trigger Prefect flows asynchronously. The service should return a job/run ID immediately.
        - [ ] Ensure `PipelineRun` status (QUEUED, PROCESSING, COMPLETED, FAILED) is correctly updated by the asynchronous Prefect flows.
        - [ ] Verify database interactions and logging within async flows.
        - [ ] Update relevant unit tests for asynchronous behavior.
    - [ ] **P4.2: Frontend UI - Modernization & Styling**
        - [ ] Review existing CSS in `frontend/src/App.css` and component-specific styles.
        - [ ] Refactor and enhance CSS for a more sleek, modern, and consistent look and feel across all pages (`UploadPage.jsx`, `ChatPage.jsx`, `StatusPage.jsx`, etc.).
        - [ ] Improve layout, typography, color palette, and use of white space.
        - [ ] (Optional/Discuss) Evaluate and potentially integrate a lightweight, modern UI component library if custom CSS becomes too cumbersome, keeping `PLANNING.MD` constraints in mind.
    - [ ] **P4.3: Frontend UI - Multi-Pipeline Selection & Batch Operations**
        - [ ] Design UI elements on `UploadPage.jsx` (or a new dedicated page/modal) to allow users to select an uploaded document and then choose multiple pipelines (RAG, Summarizer, Classifier) to run against it.
        - [ ] Implement frontend logic to gather these selections.
        - [ ] Design and implement a new backend API endpoint (e.g., `/pipelines/trigger_batch`) to accept requests for running multiple pipelines on a single document.
        - [ ] Update `app/services/pipeline_service.py` (or a new service method) to handle batch requests, creating multiple `PipelineRun` entries and dispatching multiple asynchronous Prefect flows.
        - [ ] Enhance `StatusPage.jsx` or create a new view to display the status of batch jobs and their individual sub-pipeline runs.
        - [ ] Update relevant unit tests for batch operations.

**Phase DS: Data Science & Machine Learning Platform (Date: 2024-12-19)**
  - **Goal:** Implement comprehensive machine learning capabilities for dataset upload, algorithm selection, model training, and results analysis.
  - **Reference:** See `TASK_DS.md` for detailed breakdown of all data science tasks.
  - **Tasks:**
    - [ ] **Phase DS1: Core Data Science Infrastructure**
        - [ ] Enhanced file upload support for CSV/Excel datasets
        - [ ] Data profiling and preview services
        - [ ] ML pipeline foundation with algorithm registry
        - [ ] Data preprocessing automation
        - [ ] Core ML training workflow with model evaluation
    - [ ] **Phase DS2: User Interface & Experience**
        - [ ] Data upload and configuration UI components
        - [ ] Algorithm and hyperparameter configuration interface
        - [ ] Results dashboard and model comparison interface
    - [ ] **Phase DS3: Advanced Features & Optimization**
        - [ ] Model management and persistence system
        - [ ] Enhanced ML capabilities and visualizations
    - [ ] **Phase DS4: Integration & Production Readiness**
        - [ ] Integration with existing platform infrastructure
        - [ ] Performance optimization and comprehensive testing

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

### Discovered During Work (2024-08-02)
- **Test Failures in Pipeline Router (P2.6):**
    - Problem: `mock_db_session_override.get` was expected to be called in tests but wasn't being called because the router wasn't checking for file log existence.
    - Fix: Added file log existence check in the router before calling the service.
- **Inconsistent Error Messages (P2.6):**
    - Problem: Error message format in tests didn't match the actual format returned by the code.
    - Fix: Updated the error message format in the router and test assertions.
- **LangChain Mocking Issues (P2.6):**
    - Problem: Tests were trying to mock individual components (llm.invoke, parser.invoke) but LangChain now uses a chain pattern with the `|` operator.
    - Fix: Updated tests to properly mock the LangChain chain pattern by patching the `__or__` operator and chain invocation.

### Discovered During Work (2024-08-03)
- **Vector Store Management:**
    - Added new API endpoints for managing vector stores to support better RAG chatbot functionality.
    - Implemented listing of all vector stores with metadata (size, path, document ID).
    - Added ability to check if a specific vector store exists.
    - Added ability to delete vector stores no longer needed.