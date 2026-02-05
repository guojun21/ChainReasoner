"""
Main MultiHop Agent system that orchestrates all components.
Implements the complete multi-hop reasoning pipeline.
"""

import time
from typing import Any, Dict, List, Optional


class MultiHopAgent:
    """Main agent system for multi-hop reasoning without AI search tools."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        # Components will be initialized in initialize_system
        self.planner = None
        self.retriever = None
        self.graph_builder = None
        self.reasoner = None
        self.validator = None
        self.executor = None
        self.answer_generator = None
        self.system_initialized = False
        
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j", 
            "neo4j_password": "password",
            "contriever_model_path": None,
            "api_keys": {},
            "browser_config": {"headless": True, "timeout": 30},
            "max_retries": 3,
            "top_k_retrieval": 5
        }
    
    def initialize_system(self, documents: Optional[List[str]] = None):
        """Initialize all system components."""
        print("Initializing MultiHop Agent system...")
        
        # Import components here to avoid circular imports
        from src.agents.planner_agent import PlannerAgent
        from src.tools.retriever import TraditionalRetriever
        from src.tools.graph_builder import GraphBuilder
        from src.agents.reasoner import Reasoner
        from src.agents.validator import Validator
        from src.agents.executor import Executor
        from src.agents.answer_generator import AnswerGenerator
        
        self.planner = PlannerAgent()
        self.retriever = TraditionalRetriever()
        self.graph_builder = GraphBuilder(
            neo4j_uri=self.config.get("neo4j_uri", "bolt://localhost:7687"),
            neo4j_user=self.config.get("neo4j_user", "neo4j"),
            neo4j_password=self.config.get("neo4j_password", "password")
        )
        self.reasoner = Reasoner(
            neo4j_uri=self.config.get("neo4j_uri", "bolt://localhost:7687"),
            neo4j_user=self.config.get("neo4j_user", "neo4j"),
            neo4j_password=self.config.get("neo4j_password", "password")
        )
        self.validator = Validator(api_keys=self.config.get("api_keys", {}))
        self.executor = Executor(browser_config=self.config.get("browser_config", {}))
        self.answer_generator = AnswerGenerator()
        
        # Initialize retriever with documents if provided
        if documents:
            self.retriever.initialize(
                documents=documents,
                contriever_model_path=self.config.get("contriever_model_path")
            )
        
        # Initialize other components
        self.graph_builder.initialize_graph()
        self.reasoner.connect_to_graph()
        self.executor.initialize_executor()
        
        self.system_initialized = True
        print("MultiHop Agent system initialized successfully!")
    
    def process_question(self, question: str, question_id: str = "unknown") -> Dict[str, Any]:
        """Process a single multi-hop reasoning question."""
        if not self.system_initialized:
            raise ValueError("System not initialized. Call initialize_system() first.")
        
        start_time = time.time()
        print(f"Processing question: {question}")
        
        try:
            # Step 1: Task decomposition
            parsed_question = self.planner.parse_question(question)
            sub_tasks = self.planner.decompose_task(parsed_question)
            print(f"Generated {len(sub_tasks)} sub-tasks")
            
            # Step 2: Multi-hop retrieval
            all_retrieved_docs = []
            for task in sub_tasks:
                query = task["query"]
                retrieved_results = self.retriever.hybrid_search(
                    query, 
                    top_k=self.config.get("top_k_retrieval", 5)
                )
                all_retrieved_docs.extend([r["document"] for r in retrieved_results])
            
            # Remove duplicates while preserving order
            unique_docs = []
            seen = set()
            for doc in all_retrieved_docs:
                if doc not in seen:
                    unique_docs.append(doc)
                    seen.add(doc)
            
            print(f"Retrieved {len(unique_docs)} unique documents")
            
            # Step 3: Knowledge graph construction
            triples = self.graph_builder.build_graph_from_documents(unique_docs)
            print(f"Built knowledge graph with {len(triples)} triples")
            
            # Step 4: Path reasoning
            entities = parsed_question.get("entities", [])
            relations = parsed_question.get("relations", [])
            
            if entities:
                reasoning_result = self.reasoner.perform_multi_hop_reasoning(entities, relations)
            else:
                # Fallback: use the original question as query
                reasoning_result = {
                    "reasoning_type": "fallback",
                    "answer": "No entities found, using direct retrieval",
                    "confidence": 0.6
                }
            
            print(f"Reasoning completed with confidence: {reasoning_result.get('confidence', 0.0)}")
            
            # Step 5: Fact validation
            validation_result = self.validator.validate_reasoning_chain([reasoning_result])
            print(f"Validation result: {validation_result['is_valid']}")
            
            # Step 6: Execute complex tasks if needed
            execution_result = None
            for task in sub_tasks:
                if task.get("task_type") in ["web_browsing", "form_filling", "ocr"]:
                    execution_result = self.executor.execute_complex_task(
                        task_description=task["description"],
                        required_tools=["web_browser"]
                    )
                    break
            
            # Step 7: Generate final answer
            final_answer = self.answer_generator.generate_answer(
                question_id=question_id,
                answer=str(reasoning_result.get("result", reasoning_result)),
                question_type=parsed_question.get("question_type", "general"),
                confidence=reasoning_result.get("confidence", 0.7) * validation_result.get("confidence", 0.8),
                evidence=unique_docs[:3],  # Top 3 evidence documents
                reasoning_steps=[reasoning_result, validation_result]
            )
            
            processing_time = time.time() - start_time
            final_answer["processing_time"] = round(processing_time, 2)
            
            print(f"Question processed in {processing_time:.2f} seconds")
            return final_answer
            
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            # Return error answer
            return {
                "question_id": question_id,
                "answer": f"Error: {str(e)}",
                "answer_type": "error",
                "confidence": 0.0,
                "evidence": [],
                "reasoning_steps": [],
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    def process_questions_batch(self, questions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process a batch of questions."""
        results = []
        for question_data in questions:
            question = question_data.get("question", "")
            question_id = question_data.get("id", "unknown")
            result = self.process_question(question, question_id)
            results.append(result)
        return results
