# MultiHop Agent

A compliant multi-hop reasoning agent system that does not use AI search tools (RAG, large model retrieval, etc.) as required by the Tianchi competition.

## Architecture

The system follows a seven-layer architecture:

```
+-------------------+
|     Output Layer  |  -> Structured answers
+-------------------+
|   Execution Layer |  -> WebWatcher, WebAgent
+-------------------+
|   Validation Layer|  -> Code execution, OFAC API, cross-validation
+-------------------+
|   Reasoning Layer |  -> Planner Agent, Reasoner
+-------------------+
|    Knowledge Layer|  -> Neo4j, Wikidata, LangChain Graph RAG (build-only)
+-------------------+
|    Retrieval Layer|  -> BM25, Contriever, Brave API
+-------------------+
|     Input Layer   |  -> Receives questions from question.json
+-------------------+
```

## Key Components

- **Planner Agent**: Parses complex questions and generates multi-hop sub-task sequences
- **Traditional Retriever**: Implements BM25 (keyword) and Contriever (dense) retrieval
- **Graph Builder**: Converts unstructured text to structured entity-relation triples in Neo4j
- **Reasoner**: Performs path reasoning and Cypher queries on the knowledge graph
- **Validator**: Implements multi-dimensional validation (mathematical, external API, cross-validation)
- **Executor**: Handles web interactions and external tool execution
- **Answer Generator**: Produces competition-compliant structured answers

## Project Structure

- `src/`: Core agent logic and utilities
- `apps/`: API / Web / Console entry points
- `scripts/`: Helper scripts and test utilities
- `configs/`: Configuration files (`config.yaml`, `mcp_config.json`)
- `data/qa/`: Question and answer datasets
- `docs/`: Design and strategy documents
- `references/`: Reference projects and external code

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Neo4j database (local or remote)
4. Copy `configs/config.yaml.example` to `configs/config.yaml` and update API keys

## Usage

The system can process questions from `data/qa/question.json` and generate structured answers in the required format.

Common entry points:

- API server: `apps/api/api_server.py` or `scripts/start_server.py`
- Console: `apps/console/console_interface.py`
- Web UI: `apps/web/web_interface.py`

## Compliance

This system is fully compliant with the Tianchi competition requirements:
- ✅ No RAG or large model retrieval used
- ✅ Only traditional retrieval methods (BM25, Contriever)
- ✅ Knowledge graph-based multi-hop reasoning
- ✅ Multi-dimensional validation mechanisms
- ✅ Structured output format

## Configuration

Edit `configs/config.yaml` to configure database connections and API keys.

## Dependencies

See `requirements.txt` for complete dependency list.

## License

MIT License
