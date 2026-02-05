#!/usr/bin/env python3
"""
Unified Logger Configuration Module
Provides centralized logging configuration for all MultiHop Agent interfaces.
"""



import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


class MultiHopLogger:
    """Unified logger for MultiHop Agent system."""
    
    _loggers = {}
    _log_dir = Path(__file__).resolve().parents[2].parent / "autorunAtnight" / "logs"
    
    @classmethod
    def _ensure_log_directory(cls):
        """Ensure log directory exists."""
        cls._log_dir.mkdir(exist_ok=True)
    
    @classmethod
    def get_logger(cls, name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
        """
        Get or create a logger with the specified name.
        
        Args:
            name: Logger name (e.g., 'console', 'api', 'web')
            log_file: Optional log file name (default: {name}.log)
            level: Logging level (default: INFO)
        
        Returns:
            Logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        cls._ensure_log_directory()
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        if logger.handlers:
            return logger
        
        log_filename = log_file or f"{name}.log"
        log_path = cls._log_dir / log_filename
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.flush()
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.propagate = False
        
        cls._loggers[name] = logger
        
        return logger
    
    @classmethod
    def log_request(cls, logger: logging.Logger, endpoint: str, method: str, data: dict = None):
        """Log incoming request."""
        logger.info(f"Request: {method} {endpoint}")
        if data:
            logger.debug(f"Request data: {data}")
    
    @classmethod
    def log_response(cls, logger: logging.Logger, status_code: int, data: dict = None):
        """Log outgoing response."""
        logger.info(f"Response: Status {status_code}")
        if data:
            logger.debug(f"Response data: {data}")
    
    @classmethod
    def log_mcp_call(cls, logger: logging.Logger, service: str, query: str, success: bool, result: dict = None):
        """Log MCP service call."""
        if success:
            logger.info(f"MCP Call: {service} - Success")
            if result:
                count = result.get('count', 0)
                logger.debug(f"MCP Result: {service} returned {count} results")
        else:
            logger.error(f"MCP Call: {service} - Failed")
            if result:
                error = result.get('error', 'Unknown error')
                logger.error(f"MCP Error: {error}")
    
    @classmethod
    def log_llm_call(cls, logger: logging.Logger, model: str, question: str, success: bool, result: dict = None):
        """Log LLM API call."""
        logger.info(f"LLM Call: Model {model}")
        logger.debug(f"LLM Question: {question[:100]}...")
        if success:
            logger.info("LLM Call: Success")
            if result:
                answer = result.get('answer', '')
                logger.debug(f"LLM Answer: {answer[:100]}...")
        else:
            logger.error("LLM Call: Failed")
            if result:
                error = result.get('error', 'Unknown error')
                logger.error(f"LLM Error: {error}")
    
    @classmethod
    def log_multi_hop_step(cls, logger: logging.Logger, step: str, details: str = None):
        """Log multi-hop reasoning step."""
        logger.info(f"MultiHop Step: {step}")
        if details:
            logger.debug(f"Step Details: {details}")
    
    @classmethod
    def log_error(cls, logger: logging.Logger, error: Exception, context: str = None):
        """Log error with context."""
        logger.error(f"Error: {type(error).__name__}: {str(error)}")
        if context:
            logger.error(f"Error Context: {context}")
    
    @classmethod
    def log_performance(cls, logger: logging.Logger, operation: str, duration: float):
        """Log performance metrics."""
        logger.info(f"Performance: {operation} completed in {duration:.2f}s")


def get_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Convenience function to get a logger.
    
    Args:
        name: Logger name
        log_file: Optional log file name
        level: Logging level
    
    Returns:
        Logger instance
    """
    return MultiHopLogger.get_logger(name, log_file, level)


if __name__ == "__main__":
    logger = get_logger("test")
    logger.info("Logger test started")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.info("Logger test completed")
