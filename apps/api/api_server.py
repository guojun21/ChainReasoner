#!/usr/bin/env python3
"""
Enhanced MultiHop Agent API Server with Multi-hop Reasoning and MCP Integration
Provides HTTP/HTTPS endpoints for question answering with full multi-hop reasoning.
"""



import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import requests
import yaml
from flask import Flask, Response, jsonify, request, stream_with_context

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
DEFAULT_MCP_CONFIG_PATH = BASE_DIR / "configs" / "mcp_config.json"
INDEX_HTML_PATH = BASE_DIR / "apps" / "web" / "index.html"
sys.path.insert(0, str(BASE_DIR))

from src.utils.logger_config import get_logger, MultiHopLogger
from src.agents.constrained_search import ConstrainedSearchAgent


class EnhancedMultiHopAPIServer:
    """Enhanced API Server with Multi-hop Reasoning and MCP Integration."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_logger("api", "api.log")
        self.logger.info("="*70)
        self.logger.info("Enhanced MultiHop API Server - Starting")
        self.logger.info("="*70)
        
        self.config = self._load_config(config_path)
        self.base_model = self.config.get("base_model", {})
        self.api_token = self.config.get("api_token", "default_token_123456")
        self.mcp_config = self._load_mcp_config()
        search_cfg = self.config.get("search_agent", {}) if isinstance(self.config, dict) else {}
        self._brave_min_interval = max(search_cfg.get("brave_min_interval_seconds", 1.2), 1.5)
        self._brave_last_call_ts = 0.0
        # [THREAD-SAFE] Lock for Brave rate limiting — needed for concurrent eval
        import threading
        self._brave_lock = threading.Lock()
        search_cfg = self.config.get("search_agent", {}) if isinstance(self.config, dict) else {}
        enable_rewrite = search_cfg.get("enable_rewrite", True)
        enable_llm_answer = search_cfg.get("enable_llm_answer", True)
        enable_llm_verify = search_cfg.get("enable_llm_verify", True)
        self.search_agent = ConstrainedSearchAgent(
            self._call_brave_search,
            max_queries=max(search_cfg.get("max_queries", 3), 3),  # 3 queries for hop-1, saving budget for hop-2
            per_query_delay=search_cfg.get("per_query_delay", 0.2),
            max_results_per_query=max(search_cfg.get("max_results_per_query", 8), 8),
            max_evidence=max(search_cfg.get("max_evidence", 8), 8),
            adaptive_threshold_n=search_cfg.get("adaptive_threshold_n", 0.5),
            rewrite_fn=None,  # Disabled: rewriting strips critical context
            llm_answer_fn=self._extract_answer_llm if enable_llm_answer else None,
            llm_verify_fn=self._verify_answer_llm if enable_llm_verify else None,
            llm_decompose_fn=self._decompose_question_llm if enable_rewrite else None,
            llm_knowledge_fn=self._knowledge_answer_llm if enable_llm_answer else None
        )
        self.app = Flask(__name__)
        self._setup_routes()
        
        self.logger.info(f"Model: {self.base_model.get('model_id', 'unknown')}")
        self.logger.info(f"MCP Services: {len(self.mcp_config.get('mcpServers', {}))} available")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not config_file.is_absolute():
            config_file = BASE_DIR / config_file
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_mcp_config(self) -> Dict[str, Any]:
        """Load MCP configuration."""
        if DEFAULT_MCP_CONFIG_PATH.exists():
            with open(DEFAULT_MCP_CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"mcpServers": {}}
    
    def _call_llm(self, question: str) -> Dict[str, Any]:
        """Call LLM API with reasoning process."""
        start_time = time.time()
        
        self.logger.info("LLM API Call - Starting")
        self.logger.debug(f"Question: {question[:100]}...")
        
        api_url = self.base_model.get("api_url")
        api_key = self.base_model.get("api_key")
        model_id = self.base_model.get("model_id")
        temperature = self.base_model.get("temperature", 0.7)
        max_tokens = self.base_model.get("max_tokens", 2048)
        
        self.logger.debug(f"Model: {model_id}")
        self.logger.debug(f"API URL: {api_url}")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        system_prompt = """You are a helpful AI assistant that answers questions accurately and concisely.
Please provide your reasoning process step by step before giving the final answer.
Format your response as:
REASONING PROCESS:
[Step 1: ...]
[Step 2: ...]
...

FINAL ANSWER:
[Your final answer here]"""
        
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            reasoning_steps = []
            final_answer = ""
            
            if "REASONING PROCESS:" in content:
                parts = content.split("REASONING PROCESS:")[1].split("FINAL ANSWER:")
                reasoning_text = parts[0].strip()
                final_answer = parts[1].strip() if len(parts) > 1 else ""
                
                for line in reasoning_text.split('\n'):
                    if line.strip():
                        reasoning_steps.append(line.strip())
            else:
                final_answer = content
            
            duration = time.time() - start_time
            self.logger.info(f"LLM API Call - Success (Duration: {duration:.2f}s)")
            self.logger.debug(f"Reasoning steps: {len(reasoning_steps)}")
            self.logger.debug(f"Answer length: {len(final_answer)} characters")
            
            return {
                "reasoning_steps": reasoning_steps,
                "answer": final_answer
            }
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"LLM API Call - Failed (Duration: {duration:.2f}s)")
            MultiHopLogger.log_error(self.logger, e, "LLM API call")
            
            return {
                "reasoning_steps": [f"Error: {str(e)}"],
                "answer": f"Error: {str(e)}"
            }
    
    def _call_mcp_service(self, service_name: str, query: str) -> Dict[str, Any]:
        """Call MCP service for additional information."""
        start_time = time.time()
        
        self.logger.info(f"MCP Service Call - {service_name}")
        self.logger.debug(f"Query: {query[:100]}...")
        
        mcp_servers = self.mcp_config.get("mcpServers", {})
        
        if service_name not in mcp_servers:
            self.logger.error(f"MCP Service - {service_name} not found in configuration")
            return {
                "error": f"MCP service '{service_name}' not found",
                "available_services": list(mcp_servers.keys())
            }
        
        service_config = mcp_servers[service_name]
        self.logger.debug(f"Service config: {service_config}")
        
        try:
            if service_name == "searxng":
                result = self._call_searxng(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "web-search":
                result = self._call_web_search(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "brave-search":
                result = self._call_brave_search(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "mcp-deepwiki":
                result = self._call_mcp_deepwiki(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "trends-hub":
                result = self._call_trends_hub(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "arxiv-mcp-server":
                result = self._call_arxiv_mcp(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "pozansky-stock-server":
                result = self._call_pozansky_stock(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "worldbank-mcp":
                result = self._call_worldbank_mcp(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "mcp-server-hotnews":
                result = self._call_hotnews(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "biomcp":
                result = self._call_biomcp(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            else:
                self.logger.warning(f"MCP Service - {service_name} not implemented")
                return {
                    "error": f"MCP service '{service_name}' not yet implemented",
                    "note": "This service is configured but not yet integrated"
                }
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"MCP Service - {service_name} exception (Duration: {duration:.2f}s)")
            MultiHopLogger.log_error(self.logger, e, f"MCP service call: {service_name}")
            return {
                "error": f"MCP service error: {str(e)}"
            }
    
    def _call_searxng(self, query: str) -> Dict[str, Any]:
        """Call SearXNG search service."""
        self.logger.debug("Calling SearXNG search service")
        
        searxng_url = "https://searx.stream"
        search_url = f"{searxng_url}/search"
        
        try:
            response = requests.get(
                search_url,
                params={"q": query, "format": "json"},
                timeout=30
            )
            response.raise_for_status()
            results = response.json()
            
            if results.get("results"):
                count = len(results.get("results", []))
                self.logger.debug(f"SearXNG returned {count} results")
                return {
                    "service": "searxng",
                    "query": query,
                    "results": results["results"][:5],
                    "count": count
                }
            else:
                self.logger.debug("SearXNG returned no results")
                return {
                    "service": "searxng",
                    "query": query,
                    "results": [],
                    "count": 0
                }
        except Exception as e:
            self.logger.error(f"SearXNG error: {str(e)}")
            return {
                "error": f"SearXNG error: {str(e)}"
            }
    
    def _call_web_search(self, query: str) -> Dict[str, Any]:
        """Call web search service."""
        self.logger.debug("Calling web search service")
        
        try:
            response = requests.get(
                "https://duckduckgo.com/html/",
                params={"q": query},
                timeout=30
            )
            response.raise_for_status()
            
            self.logger.debug("Web search completed successfully")
            return {
                "service": "web-search",
                "query": query,
                "status": "success",
                "note": "Web search completed"
            }
        except Exception as e:
            self.logger.error(f"Web search error: {str(e)}")
            return {
                "error": f"Web search error: {str(e)}"
            }
    
    def _get_brave_api_key(self) -> Optional[str]:
        """Resolve Brave Search API key from config or environment."""
        if os.getenv("BRAVE_API_KEY"):
            return os.getenv("BRAVE_API_KEY")
        api_keys = self.config.get("api_keys", {}) if isinstance(self.config, dict) else {}
        if api_keys.get("brave_api_key"):
            return api_keys.get("brave_api_key")
        brave_config = self.mcp_config.get("mcpServers", {}).get("brave-search", {})
        return brave_config.get("env", {}).get("BRAVE_API_KEY")

    def _call_llm_generic(self, system_prompt: str, user_prompt: str, temperature: float = 0.0, max_tokens: int = 256, purpose: str = "") -> str:
        """Call LLM with custom prompts and return content."""
        api_url = self.base_model.get("api_url")
        api_key = self.base_model.get("api_key")
        model_id = self.base_model.get("model_id")
        if not api_url or not api_key or not model_id:
            return ""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.info(
                    f"LLM generic call: purpose={purpose or 'generic'} model={model_id} "
                    f"temp={temperature} max_tokens={max_tokens} "
                    f"system_len={len(system_prompt)} user_len={len(user_prompt)}"
                )
                response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                if "choices" in result and result["choices"]:
                    return result["choices"][0]["message"]["content"].strip()
                # Rate limit or empty response - retry after delay
                if result.get("status") in ("449", "429") or result.get("msg", "").lower().find("rate limit") >= 0:
                    self.logger.warning(f"LLM rate limited (attempt {attempt+1}), retrying in {5*(attempt+1)}s...")
                    time.sleep(5 * (attempt + 1))
                    continue
                self.logger.error(f"LLM generic call missing choices: {str(result)[:400]}")
                return ""
            except Exception as e:
                self.logger.error(f"LLM generic call error (attempt {attempt+1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                return ""
        return ""

    def _knowledge_answer_llm(self, question: str) -> str:
        """Use LLM's own knowledge to reason through a multi-hop question and produce a candidate answer."""
        system_prompt = (
            "You are an expert at answering complex multi-hop questions using your own knowledge.\n\n"
            "TASK: Reason step by step through the question, then give ONLY the final answer.\n\n"
            "RULES:\n"
            "1. Think through each clue in the question systematically\n"
            "2. Use your encyclopedic knowledge to identify entities, dates, events\n"
            "3. For Chinese questions, answer in Chinese; for English questions, answer in English\n"
            "4. Your final answer must be a specific name, number, year, or short phrase (1-15 words)\n"
            "5. NEVER say 'Unknown' or 'I don't know' — always give your best answer\n"
            "6. NEVER include explanations in your answer — just the answer itself\n\n"
            "FORMAT:\n"
            "Reasoning: [your step-by-step reasoning]\n"
            "Answer: [your concise answer]\n"
        )
        user_prompt = f"Question: {question}"
        raw = self._call_llm_generic(system_prompt, user_prompt, temperature=0.0, max_tokens=500, purpose="knowledge_answer")
        if not raw:
            return ""
        # Extract the answer from the response
        if "Answer:" in raw:
            answer_part = raw.split("Answer:")[-1].strip()
            # Take just the first line of the answer
            answer_part = answer_part.split("\n")[0].strip()
            return answer_part
        # If no "Answer:" marker, try to use the last meaningful line
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        if lines:
            return lines[-1]
        return raw.strip()

    def _decompose_question_llm(self, question: str) -> str:
        """Use LLM to decompose a complex multi-hop question into effective web search queries."""
        system_prompt = (
            "You are an expert at formulating web search queries for multi-hop questions. "
            "Given a complex question, identify the KEY entities/facts described, then generate "
            "3-5 highly targeted search queries that will find the specific answer.\n\n"
            "CRITICAL RULES:\n"
            "1. Each query must be 5-20 words with SPECIFIC details from the question\n"
            "2. NEVER use single generic words like 'author', 'spiritual', 'thesis'\n"
            "3. Include proper nouns, dates, technical terms, and distinctive phrases\n"
            "4. For multi-hop questions, create queries for EACH hop in the reasoning chain\n"
            "5. For Chinese questions, write Chinese queries; for English, write English queries\n"
            "6. Think about what Wikipedia article or authoritative source would answer this\n\n"
            "EXAMPLES:\n\n"
            "Question: 一位欧洲学者的某项开源硬件项目...该实体在21世纪10年代中期停止了在其欧洲本土的主要交易...\n"
            "Queries:\n"
            "RepRap 开源3D打印项目 创始人 Adrian Bowyer 商业实体\n"
            "RepRapPro Ltd 停止交易 2015年\n"
            "Adrian Bowyer Bath大学 退休 开源硬件\n\n"
            "Question: Who is the author of the article...prosopography...encomienda...1972 journal...\n"
            "Queries:\n"
            "prosopography colonial Spanish America encomienda hacienda 1972 article\n"
            "Latin American Research Review 1972 import substitution industrialization\n"
            "James Lockhart social history colonial Spanish America\n\n"
            "Question: There is a spiritual teacher...thirty letters...controversial guru...FBI file...\n"
            "Queries:\n"
            "Osho Rajneesh thirty letters book spiritual teacher\n"
            "Bhagwan Shree Rajneesh FBI file total pages Scribd\n"
            "Rajneesh FBI file public document pages count\n\n"
            "Output ONLY the search queries, one per line. No numbering, no bullets, no explanation.\n"
        )
        user_prompt = f"Question: {question}"
        return self._call_llm_generic(system_prompt, user_prompt, temperature=0.0, max_tokens=400, purpose="decompose_question")

    def _rewrite_query_llm(self, query: str) -> str:
        """Rewrite query to be retrieval-friendly."""
        system_prompt = "You are a search query optimizer. Rewrite the query to be concise and retrieval-friendly. Return only the rewritten query."
        rewritten = self._call_llm_generic(system_prompt, f"Query: {query}", temperature=0.0, max_tokens=64, purpose="rewrite_query")
        return rewritten or query

    def _extract_answer_llm(self, question: str, evidence: str) -> str:
        """Extract short answer from evidence."""
        system_prompt = (
            "You are a precise question answering system. You MUST use the provided evidence AND your own knowledge to answer.\n\n"
            "ABSOLUTE RULES:\n"
            "1. Return ONLY the answer — a specific name, number, year, title, or short phrase\n"
            "2. NEVER return 'Unknown' or 'Cannot be determined' — ALWAYS give your best answer\n"
            "3. NEVER return generic role words like 'Author', 'Director' — return the ACTUAL name\n"
            "4. NEVER include explanations, reasoning, preamble, or sentences\n"
            "5. Your answer should typically be 1-10 words maximum\n\n"
            "FORMAT RULES by question type:\n"
            "- Person name: Return full name (e.g. 'James Lockhart', '雷佳音和易烊千玺')\n"
            "- Company/org name: Return official name (e.g. 'RepRapPro Ltd', 'Japan Broadcasting Corporation')\n"
            "- Number/count: Return ONLY the number (e.g. '591', '4', '2.40')\n"
            "- Year: Return ONLY 4-digit year (e.g. '1979', '1953')\n"
            "- Device/thing: Return simplest common name (e.g. 'radio', 'BBC Micro')\n"
            "- Chinese answer: Respond in Chinese matching the question's expected format\n"
            "- English answer: Respond in English matching the question's expected format\n\n"
            "REASONING STRATEGY:\n"
            "- If the evidence includes a 'Preliminary answer from reasoning', treat it as a strong candidate\n"
            "- Verify or refine that preliminary answer using the web search evidence\n"
            "- If the search evidence confirms the preliminary answer, return it\n"
            "- If the search evidence suggests a BETTER answer, return the better one\n"
            "- If the evidence is irrelevant, use your knowledge alone — do NOT say Unknown\n"
            "- For multi-hop questions, follow the chain of reasoning step by step\n"
        )
        user_prompt = f"Question: {question}\n\nEvidence:\n{evidence}\n\nAnswer (just the answer, nothing else):"
        return self._call_llm_generic(system_prompt, user_prompt, temperature=0.0, max_tokens=150, purpose="extract_answer")

    def _verify_answer_llm(self, question: str, answer: str, evidence: str):
        """Verify answer with evidence. Returns (label, confidence)."""
        system_prompt = (
            "You are a verifier. Given question, answer, and evidence, respond with "
            "one label SUPPORTS, REFUTES, or INSUFFICIENT and a confidence 0-1. "
            "Format: LABEL|CONFIDENCE"
        )
        user_prompt = f"Question: {question}\nAnswer: {answer}\nEvidence:\n{evidence}\nVerdict:"
        raw = self._call_llm_generic(system_prompt, user_prompt, temperature=0.0, max_tokens=32, purpose="verify_answer")
        label = "INSUFFICIENT"
        confidence = 0.0
        if raw:
            parts = raw.strip().split("|")
            if parts:
                label = parts[0].strip().upper()
            if len(parts) > 1:
                try:
                    confidence = float(parts[1].strip())
                except ValueError:
                    confidence = 0.0
        return label, confidence

    def _mask_api_key(self, value: str) -> str:
        if not value:
            return ""
        if len(value) <= 8:
            return "*" * len(value)
        return f"{value[:4]}***{value[-4:]}"

    def _log_search_trace(self, record: Dict[str, Any]):
        """Append search trace record to JSONL file."""
        try:
            trace_path = MultiHopLogger._log_dir / "search_trace.jsonl"
            trace_path.parent.mkdir(exist_ok=True)
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write search trace: {e}")

    def _call_brave_search(self, query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call Brave Search API for web results."""
        self.logger.info("Calling brave-search service with real data")

        api_key = self._get_brave_api_key()
        if not api_key:
            return {
                "service": "brave-search",
                "query": query,
                "error": "Missing Brave API key (set BRAVE_API_KEY or api_keys.brave_api_key)"
            }

        try:
            # [THREAD-SAFE] Brave rate limiting with lock for concurrent eval
            with self._brave_lock:
                now = time.time()
                if now - self._brave_last_call_ts < self._brave_min_interval:
                    time.sleep(self._brave_min_interval - (now - self._brave_last_call_ts))
                self._brave_last_call_ts = time.time()
            endpoint = "https://api.search.brave.com/res/v1/web/search"
            params = {
                "q": query,
                "count": 10
            }
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": api_key
            }
            start_time = time.time()
            response = None
            for _retry in range(4):
                response = requests.get(endpoint, headers=headers, params=params, timeout=30)
                if response.status_code == 429:
                    # Aggressive exponential backoff: 15, 30, 60, 120 seconds
                    wait_time = 15 * (2 ** _retry)
                    self.logger.warning(f"Brave 429 rate limited, retrying in {wait_time}s (attempt {_retry+1}/4)...")
                    time.sleep(wait_time)
                    with self._brave_lock:
                        self._brave_last_call_ts = time.time()
                    continue
                break
            response.raise_for_status()
            data = response.json()
            results = data.get("web", {}).get("results", [])
            formatted = []
            for item in results[:10]:
                formatted.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("description", "")
                })

            elapsed_ms = int((time.time() - start_time) * 1000)
            top_titles = [item.get("title", "") for item in formatted[:3]]
            self.logger.info(
                f"Brave search ok: status={response.status_code} elapsed_ms={elapsed_ms} "
                f"query={query[:80]} results={len(formatted)} top_titles={top_titles}"
            )
            self._log_search_trace({
                "type": "brave_search",
                "endpoint": endpoint,
                "params": params,
                "headers": {"X-Subscription-Token": self._mask_api_key(api_key)},
                "original_query": (meta or {}).get("original_query"),
                "rewritten_query": (meta or {}).get("rewritten_query"),
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
                "result_count": len(formatted),
                "top_titles": top_titles
            })

            return {
                "service": "brave-search",
                "query": query,
                "results": formatted,
                "count": len(formatted)
            }
        except Exception as e:
            self.logger.error(f"Brave search error: {str(e)}")
            self._log_search_trace({
                "type": "brave_search_error",
                "params": {"q": query, "count": 5},
                "original_query": (meta or {}).get("original_query"),
                "rewritten_query": (meta or {}).get("rewritten_query"),
                "error": str(e)
            })
            return {
                "service": "brave-search",
                "query": query,
                "error": f"Brave search error: {str(e)}"
            }
    
    def _call_mcp_service_generic(self, service_name: str, command: list, function_name: str, query: str) -> Dict[str, Any]:
        """Generic MCP service caller using subprocess."""
        self.logger.debug(f"Calling {service_name} service with real data")
        
        try:
            import subprocess
            import json
            import time
            
            # 构建MCP请求
            mcp_request = {
                "id": f"test-{service_name}",
                "function": function_name,
                "arguments": {
                    "query": query,
                    "count": 10
                }
            }
            
            # 启动MCP服务并发送请求
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            # 发送请求
            request_json = json.dumps(mcp_request) + '\n'
            
            # 读取响应
            stdout_lines = []
            
            # 发送输入
            process.stdin.write(request_json)
            process.stdin.flush()
            
            # 设置超时
            start_time = time.time()
            timeout = 30
            
            # 读取输出
            while time.time() - start_time < timeout:
                # 读取标准输出
                if process.stdout.closed:
                    break
                
                line = process.stdout.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                
                stdout_lines.append(line)
                self.logger.debug(f"{service_name} output: {line.strip()}")
                
                # 检查是否包含JSON响应
                try:
                    if line.strip():
                        response = json.loads(line.strip())
                        if "result" in response:
                            # 找到有效响应，处理结果
                            result = response["result"]
                            # 提取搜索结果
                            search_results = []
                            if "content" in result:
                                for item in result["content"]:
                                    if item["type"] == "text":
                                        try:
                                            content_json = json.loads(item["text"])
                                            if "results" in content_json:
                                                for search_item in content_json["results"]:
                                                    search_results.append({
                                                        "title": search_item.get("title", ""),
                                                        "url": search_item.get("url", ""),
                                                        "content": search_item.get("content", "")
                                                    })
                                        except json.JSONDecodeError:
                                            # 尝试直接解析文本内容
                                            search_results.append({
                                                "title": f"{service_name} Result",
                                                "url": "",
                                                "content": item["text"]
                                            })
                            
                            # 终止进程
                            process.terminate()
                            try:
                                process.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                process.kill()
                            
                            self.logger.debug(f"{service_name} completed successfully, found {len(search_results)} results")
                            return {
                                "service": service_name,
                                "query": query,
                                "results": search_results,
                                "count": len(search_results)
                            }
                except json.JSONDecodeError:
                    # 不是JSON，继续读取
                    continue
            
            # 超时或没有找到有效响应
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
            
            # 读取剩余的错误输出
            if process.stderr:
                stderr_output = process.stderr.read()
                if stderr_output:
                    self.logger.error(f"{service_name} error: {stderr_output}")
            
            # 合并所有输出
            all_output = ''.join(stdout_lines)
            self.logger.debug(f"{service_name} complete output: {all_output}")
            
            # 尝试从完整输出中解析结果
            try:
                # 寻找JSON响应部分
                import re
                json_matches = re.findall(r'\{[^}]*"result"[^}]*\}', all_output)
                for match in json_matches:
                    try:
                        response = json.loads(match)
                        if "result" in response:
                            result = response["result"]
                            # 提取搜索结果
                            search_results = []
                            if "content" in result:
                                for item in result["content"]:
                                    if item["type"] == "text":
                                        try:
                                            content_json = json.loads(item["text"])
                                            if "results" in content_json:
                                                for search_item in content_json["results"]:
                                                    search_results.append({
                                                        "title": search_item.get("title", ""),
                                                        "url": search_item.get("url", ""),
                                                        "content": search_item.get("content", "")
                                                    })
                                        except json.JSONDecodeError:
                                            # 尝试直接解析文本内容
                                            search_results.append({
                                                "title": f"{service_name} Result",
                                                "url": "",
                                                "content": item["text"]
                                            })
                            
                            self.logger.debug(f"{service_name} completed successfully, found {len(search_results)} results")
                            return {
                                "service": service_name,
                                "query": query,
                                "results": search_results,
                                "count": len(search_results)
                            }
                    except json.JSONDecodeError:
                        continue
            except Exception as parse_error:
                self.logger.error(f"Error parsing {service_name} response: {parse_error}")
            
            # 如果解析失败，返回默认结果
            self.logger.warning(f"Failed to parse {service_name} response, returning default result")
            return {
                "service": service_name,
                "query": query,
                "results": [],
                "count": 0
            }
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"{service_name} timeout")
            if 'process' in locals():
                process.kill()
            return {
                "error": f"{service_name} timeout"
            }
        except Exception as e:
            self.logger.error(f"{service_name} error: {str(e)}")
            if 'process' in locals():
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except:
                    pass
            return {
                "error": f"{service_name} error: {str(e)}"
            }
    
    def _call_mcp_deepwiki(self, query: str) -> Dict[str, Any]:
        """Call mcp-deepwiki service."""
        return self._call_mcp_service_generic(
            "mcp-deepwiki",
            ["npx.cmd", "mcp-deepwiki"],
            "deepwiki_search",
            query
        )
    
    def _call_trends_hub(self, query: str) -> Dict[str, Any]:
        """Call trends-hub service."""
        return self._call_mcp_service_generic(
            "trends-hub",
            ["npx.cmd", "trends-hub"],
            "trends_search",
            query
        )
    
    def _call_arxiv_mcp(self, query: str) -> Dict[str, Any]:
        """Call arxiv-mcp-server service."""
        return self._call_mcp_service_generic(
            "arxiv-mcp-server",
            ["npx.cmd", "arxiv-mcp-server"],
            "arxiv_search",
            query
        )
    
    def _call_pozansky_stock(self, query: str) -> Dict[str, Any]:
        """Call pozansky-stock-server service."""
        return self._call_mcp_service_generic(
            "pozansky-stock-server",
            ["npx.cmd", "pozansky-stock-server"],
            "stock_search",
            query
        )
    
    def _call_worldbank_mcp(self, query: str) -> Dict[str, Any]:
        """Call worldbank-mcp service."""
        return self._call_mcp_service_generic(
            "worldbank-mcp",
            ["npx.cmd", "worldbank-mcp"],
            "worldbank_search",
            query
        )
    
    def _call_hotnews(self, query: str) -> Dict[str, Any]:
        """Call mcp-server-hotnews service."""
        return self._call_mcp_service_generic(
            "mcp-server-hotnews",
            ["npx.cmd", "mcp-server-hotnews"],
            "hotnews_search",
            query
        )
    
    def _call_biomcp(self, query: str) -> Dict[str, Any]:
        """Call biomcp service."""
        return self._call_mcp_service_generic(
            "biomcp",
            ["npx.cmd", "biomcp"],
            "bio_search",
            query
        )
    
    def _multi_hop_reasoning(self, question: str, use_mcp: bool = False) -> Dict[str, Any]:
        """Perform multi-hop reasoning with optional MCP integration."""
        start_time = time.time()
        
        self.logger.info("="*70)
        self.logger.info("Multi-Hop Reasoning - Starting")
        self.logger.info(f"Question: {question[:100]}...")
        self.logger.info(f"MCP Enabled: {use_mcp}")
        self.logger.info("="*70)
        
        reasoning_steps = []
        mcp_results = []
        
        if use_mcp:
            self.logger.info("Multi-Hop Step 1: Constrained search pipeline")
            reasoning_steps.append("Step 1: Constrained search pipeline")

            search_result = self.search_agent.answer(question)
            reasoning_steps.extend(search_result.get("reasoning_steps", []))
            mcp_results = search_result.get("search_traces", [])
            final_answer = search_result.get("answer", "Unknown")
        else:
            self.logger.info("Multi-Hop Step 1: LLM analysis")
            reasoning_steps.append("Step 1: LLM analysis")

            llm_result = self._call_llm(question)
            reasoning_steps.extend(llm_result.get("reasoning_steps", []))
            final_answer = llm_result.get("answer", "")
        
        self.logger.info("Multi-Hop Step 2: Synthesizing final answer")
        reasoning_steps.append("Step 2: Synthesizing final answer")
        
        duration = time.time() - start_time
        self.logger.info(f"Multi-Hop Reasoning - Completed (Duration: {duration:.2f}s)")
        self.logger.info(f"Total reasoning steps: {len(reasoning_steps)}")
        self.logger.info(f"MCP results: {len(mcp_results)}")
        
        return {
            "question": question,
            "answer": final_answer,
            "reasoning_steps": reasoning_steps,
            "mcp_results": mcp_results if use_mcp else [],
            "use_mcp": use_mcp,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_event_stream(self, question: str, use_mcp: bool = False) -> Generator[str, None, None]:
        """Generate SSE event stream with multi-hop reasoning."""
        self.logger.info("="*70)
        self.logger.info("SSE Stream - Starting")
        self.logger.info(f"Question: {question[:100]}...")
        self.logger.info(f"MCP Enabled: {use_mcp}")
        self.logger.info("="*70)
        
        result = self._multi_hop_reasoning(question, use_mcp)
        
        reasoning_steps = result.get("reasoning_steps", [])
        answer = result.get("answer", "")
        mcp_results = result.get("mcp_results", [])
        
        self.logger.info(f"Stream: Reasoning steps: {len(reasoning_steps)}")
        self.logger.info(f"Stream: MCP results: {len(mcp_results)}")
        self.logger.info(f"Stream: Final answer: {answer[:100]}...")
        
        for i, step in enumerate(reasoning_steps, 1):
            event = {
                "type": "reasoning",
                "step": i,
                "content": step
            }
            self.logger.debug(f"Stream: Sending step {i}")
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            time.sleep(0.3)
        
        if mcp_results:
            mcp_event = {
                "type": "mcp_results",
                "results": mcp_results
            }
            self.logger.debug("Stream: Sending MCP results")
            yield f"data: {json.dumps(mcp_event, ensure_ascii=False)}\n\n"
        
        final_event = {
            "type": "answer",
            "answer": answer,
            "use_mcp": use_mcp,
            "timestamp": datetime.now().isoformat()
        }
        self.logger.debug("Stream: Sending final answer")
        yield f"data: {json.dumps(final_event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        
        self.logger.info("SSE Stream - Completed")
        self.logger.info("="*70)
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        # 添加CORS中间件
        @self.app.after_request
        def after_request(response):
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
            response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
            return response
        
        @self.app.route('/', methods=['GET'])
        def index():
            """Main web interface endpoint."""
            self.logger.info("Web interface - Request received")
            try:
                with open(INDEX_HTML_PATH, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content
            except Exception as e:
                self.logger.error(f"Error serving index.html: {str(e)}")
                return jsonify({
                    "error": "Internal Server Error",
                    "message": "Failed to load web interface"
                }), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            self.logger.info("Health check - Request received")
            return jsonify({
                "status": "healthy",
                "service": "Enhanced MultiHop Agent API",
                "version": "2.0.0",
                "features": {
                    "multi_hop_reasoning": True,
                    "mcp_integration": True,
                    "sse_support": True
                },
                "mcp_services": list(self.mcp_config.get("mcpServers", {}).keys()),
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/api/v1/answer', methods=['POST'])
        def answer_endpoint():
            """Main answer endpoint with multi-hop reasoning and MCP support."""
            self.logger.info("="*70)
            self.logger.info("API Request - /api/v1/answer")
            
            auth_header = request.headers.get('Authorization', '')
            
            if not auth_header.startswith('Bearer '):
                self.logger.warning("API Request - Unauthorized: Missing or invalid Authorization header")
                return jsonify({
                    "error": "Unauthorized",
                    "message": "Missing or invalid Authorization header"
                }), 401
            
            token = auth_header.replace('Bearer ', '')
            
            if token != self.api_token:
                self.logger.warning("API Request - Forbidden: Invalid API token")
                return jsonify({
                    "error": "Forbidden",
                    "message": "Invalid API token"
                }), 403
            
            data = request.get_json()
            
            if not data or 'question' not in data:
                self.logger.warning("API Request - Bad Request: Missing 'question' field")
                return jsonify({
                    "error": "Bad Request",
                    "message": "Missing 'question' field in request body"
                }), 400
            
            question = data['question']
            use_mcp = data.get('use_mcp', False)
            accept_header = request.headers.get('Accept', '')
            
            self.logger.info(f"API Request - Question: {question[:100]}...")
            self.logger.info(f"API Request - MCP: {use_mcp}")
            self.logger.debug(f"API Request - Accept header: {accept_header}")
            
            if 'text/event-stream' in accept_header:
                self.logger.info("API Request - Using SSE stream")
                return Response(
                    stream_with_context(
                        self._generate_event_stream(question, use_mcp),
                        mimetype='text/event-stream'
                    ),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'X-Accel-Buffering': 'no',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Headers': 'Content-Type,Authorization,Accept',
                        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
                    }
                )
            else:
                self.logger.info("API Request - Using standard JSON response")
                result = self._multi_hop_reasoning(question, use_mcp)
                self.logger.info(f"API Response - Status: 200")
                return jsonify(result)
        
        @self.app.route('/api/v1/mcp/list', methods=['GET'])
        def mcp_list():
            """List available MCP services."""
            self.logger.info("API Request - /api/v1/mcp/list")
            mcp_services = self.mcp_config.get("mcpServers", {})
            self.logger.info(f"API Response - MCP services: {len(mcp_services)}")
            return jsonify({
                "mcp_services": mcp_services,
                "count": len(mcp_services)
            })
        
        @self.app.route('/api/v1/mcp/call', methods=['POST'])
        def mcp_call():
            """Call specific MCP service."""
            self.logger.info("="*70)
            self.logger.info("API Request - /api/v1/mcp/call")
            
            auth_header = request.headers.get('Authorization', '')
            
            if not auth_header.startswith('Bearer '):
                self.logger.warning("API Request - Unauthorized: Missing or invalid Authorization header")
                return jsonify({"error": "Unauthorized"}), 401
            
            token = auth_header.replace('Bearer ', '')
            if token != self.api_token:
                self.logger.warning("API Request - Forbidden: Invalid API token")
                return jsonify({"error": "Forbidden"}), 403
            
            data = request.get_json()
            if not data or 'service' not in data or 'query' not in data:
                self.logger.warning("API Request - Bad Request: Missing 'service' or 'query' field")
                return jsonify({"error": "Bad Request"}), 400
            
            service_name = data['service']
            query = data['query']
            
            # 确保查询参数编码正确
            self.logger.info(f"API Request - Service: {service_name}")
            self.logger.info(f"API Request - Query: {query}")
            self.logger.info(f"Query type: {type(query)}")
            self.logger.info(f"Query length: {len(query)}")
            
            # 尝试使用utf-8编码确保中文字符正确
            if isinstance(query, str):
                try:
                    # 检查字符串是否包含非ASCII字符
                    has_non_ascii = any(ord(c) > 127 for c in query)
                    self.logger.info(f"Query has non-ASCII characters: {has_non_ascii}")
                except Exception as e:
                    self.logger.error(f"Error checking query: {e}")
            
            result = self._call_mcp_service(service_name, query)
            self.logger.info(f"API Response - Status: 200")
            return jsonify(result)
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, ssl_context=None):
        """Run Flask server."""
        self.logger.info("="*70)
        self.logger.info("Starting Flask Server")
        self.logger.info(f"Host: {host}")
        self.logger.info(f"Port: {port}")
        self.logger.info("="*70)
        
        print("\n" + "="*70)
        print("Enhanced MultiHop Agent API Server")
        print("="*70)
        print(f"\nServer starting on {host}:{port}")
        print(f"API Token: {self.api_token}")
        print(f"Model: {self.base_model.get('model_id', 'unknown')}")
        print(f"\nFeatures:")
        print(f"  ✅ Multi-hop Reasoning")
        print(f"  ✅ MCP Integration")
        print(f"  ✅ SSE Support")
        print(f"\nAvailable MCP Services: {len(self.mcp_config.get('mcpServers', {}))}")
        for service in self.mcp_config.get('mcpServers', {}).keys():
            print(f"  - {service}")
        print(f"\nEndpoints:")
        print(f"  - GET  /health")
        print(f"  - POST /api/v1/answer")
        print(f"  - GET  /api/v1/mcp/list")
        print(f"  - POST /api/v1/mcp/call")
        print(f"\nExample curl command:")
        print(f'  curl -X POST \\')
        print(f'    -H "Authorization: Bearer {self.api_token}" \\')
        print(f'    -H "Content-Type: application/json" \\')
        print(f'    -H "Accept: text/event-stream" \\')
        print(f'    -d \'{{"question": "Where is the capital of France?", "use_mcp": true}}\' \\')
        print(f'    "http://{host}:{port}/api/v1/answer"')
        print("="*70 + "\n")
        
        self.app.run(host=host, port=port, ssl_context=ssl_context, threaded=True)


def main():
    """Main function."""
    server = EnhancedMultiHopAPIServer()
    
    try:
        import ssl
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain('server.crt', 'server.key')
        use_ssl = True
    except:
        use_ssl = False
    
    host = '0.0.0.0'
    port = 5000
    
    server.run(host=host, port=port, ssl_context=ssl_context if use_ssl else None)


if __name__ == "__main__":
    main()
