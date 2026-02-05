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
|    Retrieval Layer|  -> BM25, Contriever, Bing API
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

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Neo4j database (local or remote)
4. Configure API keys in `configs/config.yaml`

## Usage

The system can process questions from `question.json` and generate structured answers in the required format.

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
