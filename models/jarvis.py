from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, cast, Type, Protocol, runtime_checkable, TypeVar, TypedDict
from datetime import datetime
import logging
import json
from pathlib import Path
import uuid
from pathlib import Path
from typing_extensions import Protocol as TypingProtocol

# Type definitions
JSONType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]
LLMResponse = Dict[str, Any]
NLPInsights = Dict[str, Any]
ConversationHistory = List[Dict[str, Any]]

# Define protocols for LLM and NLP components
@runtime_checkable
class LLMProtocol(TypingProtocol):
    """Protocol for LLM implementations."""
    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        ...
    def complete(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        ...
    def predict(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        ...
    def summarize(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        ...

@runtime_checkable
class NLPProtocol(TypingProtocol):
    """Protocol for NLP analyzer implementations."""
    def analyze(self, text: str) -> Dict[str, Any]:
        ...
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        ...
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        ...

# Define type variables for LLM and NLP implementations
TLLM = TypeVar('TLLM', bound=LLMProtocol)
TNLPAnalyzer = TypeVar('TNLPAnalyzer', bound=NLPProtocol)

# Abstract base classes for LLM and NLP components
class LLMBase(LLMProtocol, ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    def complete(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    def predict(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        pass

class NLPBase(NLPProtocol, ABC):
    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        pass

# Adapter for LLM implementations
class LLMAdapter(LLMBase):
    """Adapter for LLM implementations to match the base class."""
    def __init__(self, llm: Any):
        self.llm = llm
    
    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        try:
            if hasattr(self.llm, 'generate'):
                return self.llm.generate(prompt, **kwargs)
            return {"response": "[AI niet beschikbaar]", "success": False}
        except Exception as e:
            return {"response": f"LLM error: {str(e)}", "success": False}
    
    def complete(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        try:
            if hasattr(self.llm, 'complete'):
                return self.llm.complete(text, **kwargs)
            return {"response": f"Completed: {text}", "success": True}
        except Exception as e:
            return {"response": f"Completion error: {str(e)}", "success": False}
    
    def predict(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        """Predict using the wrapped LLM."""
        if hasattr(self.llm, 'predict'):
            return self.llm.predict(text, **kwargs)
        return {"prediction": "Prediction not implemented", "success": False}
        
    def summarize(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        """Summarize text using the wrapped LLM."""
        if hasattr(self.llm, 'summarize'):
            return self.llm.summarize(text, **kwargs)
        # Fallback implementation that generates a simple summary
        words = text.split()
        summary = ' '.join(words[:50]) + ('...' if len(words) > 50 else '')
        return {"summary": summary, "success": True}

# Adapter for NLP implementations
class NLPAdapter(NLPBase):
    """Adapter for NLP implementations to match the base class."""
    def __init__(self, nlp: Any):
        self.nlp = nlp
    
    def analyze(self, text: str) -> Dict[str, Any]:
        try:
            return self.nlp.analyze(text)
        except Exception:
            return {
                "entities": [{"text": text, "type": "UNKNOWN"}],
                "sentiment": {"polarity": 0.0, "label": "neutral"},
                "success": False
            }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        try:
            return self.nlp.analyze_sentiment(text)
        except Exception:
            return {"sentiment": "neutral", "score": 0.0, "success": False}
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        try:
            return self.nlp.extract_entities(text)
        except Exception:
            return [{"text": text, "type": "UNKNOWN", "start": 0, "end": len(text)}]

# Default implementations
class DefaultLLM(LLMBase):
    """Default LLM implementation when core is not available."""
    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {"response": "LLM not available", "success": False}
        
    def complete(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        return {"response": text + " [completed]", "success": True}
        
    def predict(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "prediction": "This is a prediction from the default LLM implementation.",
            "confidence": 0.9,
            "success": True
        }
        
    def summarize(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        words = text.split()
        summary = ' '.join(words[:50]) + ('...' if len(words) > 50 else '')
        return {
            "summary": summary,
            "success": True,
            "model": "default-llm"
        }

class DefaultNLP(NLPBase):
    """Default NLP analyzer implementation when core is not available."""
    def analyze(self, text: str) -> Dict[str, Any]:
        return {
            "entities": [{"text": text, "type": "UNKNOWN"}],
            "sentiment": {"polarity": 0.0, "label": "neutral"},
            "success": True
        }
        
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        return {"sentiment": "neutral", "score": 0.0, "success": True}
        
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        return [{"text": text, "type": "UNKNOWN", "start": 0, "end": len(text)}]
    
    def __call__(self, text: str) -> Dict[str, Any]:
        return self.analyze(text)

class JarvisModel:
    """
    Main model for processing user input pipelines.
    Handles text processing, LLM interactions, and NLP analysis.
    
    This class uses the Protocol pattern to be compatible with different LLM and NLP implementations.
    """
    
    def __init__(
        self,
        llm: Optional[Union[LLMProtocol, Any]] = None,
        nlp_analyzer: Optional[Union[NLPProtocol, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        db: Optional[Any] = None,
    ) -> None:
        """
        Initialize the Jarvis model with required components.
        
        Args:
            llm: An instance of LLM implementation (defaults to LLMCore if None)
            nlp_analyzer: Optional NLP analyzer (defaults to NLPProcessor if None)
            config: Configuration dictionary
            db: Optional database session
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db = db
        
        # Initialize LLM with default implementation if not provided
        if llm is None:
            try:
                from llm.core.llm_core import LLMCore
                self.llm = LLMAdapter(LLMCore())
            except Exception as e:
                self.logger.warning(f"Falling back to DefaultLLM: {e}")
                self.llm = DefaultLLM()
        elif not isinstance(llm, LLMProtocol):
            self.llm = LLMAdapter(llm)
        else:
            self.llm = llm
            
        # Initialize NLP analyzer with default implementation if not provided
        if nlp_analyzer is None:
            try:
                from nlp.processor import NLPProcessor
                self.nlp_analyzer = NLPAdapter(NLPProcessor())
            except Exception as e:
                self.logger.warning(f"Falling back to DefaultNLP: {e}")
                self.nlp_analyzer = DefaultNLP()
        elif not isinstance(nlp_analyzer, NLPProtocol):
            self.nlp_analyzer = NLPAdapter(nlp_analyzer)
        else:
            self.nlp_analyzer = nlp_analyzer
        
        self._conversation_history: List[Dict[str, Any]] = []
        self.initialized = True
        self.logger.info("JarvisModel initialized with LLM and NLP components")

    def process_input(
        self, 
        text: str, 
        user_id: str = "default", 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process user input and generate a response using the configured LLM and NLP components.
        Returns a dict with at least a 'response' key (string) and 'success' boolean.
        """
        start_time = datetime.utcnow()
        request_id = str(uuid.uuid4())
        metadata = kwargs.pop('metadata', {})
        
        try:
            # Clean input text by removing any known prefixes
            cleaned_text = text.strip()
            for prefix in ["Ik heb je bericht ontvangen:", "Ik heb je bericht ontvangen", "Bericht ontvangen:"]:
                if cleaned_text.startswith(prefix):
                    cleaned_text = cleaned_text[len(prefix):].strip(" :\"'")
            
            self.logger.info(f"Processing input from user {user_id}: {cleaned_text[:100]}...")
            
            # Get NLP analysis
            nlp_result = self.nlp_analyzer.analyze(cleaned_text)
            
            # Create a clean prompt without any prefixes
            prompt = f"{cleaned_text}\n\nAntwoord direct en beknopt:"
            
            # Generate response using LLM
            response = self.llm.generate(
                prompt=prompt,
                context={
                    "user_id": user_id,
                    "nlp_insights": nlp_result,
                    **kwargs
                }
            )
            
            # Extract and clean the response text
            if isinstance(response, dict):
                response_str = response.get("text") or response.get("response") or str(response)
            else:
                response_str = str(response)
            
            # Clean common prefixes and suffixes
            response_str = response_str.strip()
            for prefix in ["Antwoord:", "Reactie:", "Bericht:", "-", "•"]:
                if response_str.startswith(prefix):
                    response_str = response_str[len(prefix):].strip()
            
            # Remove any surrounding quotes and whitespace
            response_str = response_str.strip(" \t\n'\"")
            
            # Log the interaction
            self._log_interaction(
                user_id=user_id,
                input_text=cleaned_text,
                response=response_str,
                confidence=nlp_result.get("confidence", 0.9),
                metadata={
                    **metadata,
                    "nlp_result": nlp_result,
                    "processing_time": (datetime.utcnow() - start_time).total_seconds()
                }
            )
            
            # Update conversation history with cleaned text
            self._update_conversation_history(
                user_id=user_id,
                user_input=cleaned_text,
                assistant_response=response_str,
                metadata={
                    "request_id": request_id,
                    **metadata
                }
            )
            
            return {
                "response": response_str,
                "success": True,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time": (datetime.utcnow() - start_time).total_seconds()
            }
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error processing input: {error_msg}", exc_info=True)
            self._log_interaction(
                user_id=user_id,
                input_text=text,
                response=error_msg,
                confidence=0.0,
                metadata={
                    "error": error_msg,
                    "error_type": type(e).__name__,
                    **kwargs.get('metadata', {})
                }
            )
            return {
                "response": "An error occurred while processing your request.",
                "success": False,
                "error": error_msg,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "user_id": user_id,
                    "error_type": type(e).__name__
                }
            }

    def _update_conversation_history(
        self,
        user_id: str,
        user_input: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update the conversation history with the latest interaction.
        
        Args:
            user_id: ID of the user
            user_input: The input text from the user
            assistant_response: The response from the assistant
            metadata: Additional metadata to store with the interaction
        """
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "user_input": user_input,
            "assistant_response": assistant_response,
            "metadata": metadata or {}
        }
        
        # Add to conversation history
        self._conversation_history.append(interaction)
        
        # Keep only the most recent 100 messages to prevent memory issues
        if len(self._conversation_history) > 100:
            self._conversation_history = self._conversation_history[-100:]
            
        self.logger.debug(f"Updated conversation history for user {user_id}")
        
    def _log_interaction(
        self,
        user_id: str,
        input_text: str,
        response: str,  # Enforcing string type
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an interaction to the database.
        
        Args:
            user_id: ID of the user making the request
            input_text: The input text from the user
            response: The response text from the AI (must be a string)
            confidence: The confidence score of the response (0.0 to 1.0)
            metadata: Additional metadata to store with the interaction
        """
        if not hasattr(self, 'db') or self.db is None:
            self.logger.debug("No database connection available for logging")
            return
            
        try:
            # Import inside the function to avoid circular imports
            from db.sql.models.request_log import AIRequestLog
            # Create and save the log entry
            log_entry = AIRequestLog(
                user_id=user_id,
                input_text=input_text,
                response_text=response,
                confidence=min(max(0.0, float(confidence)), 1.0),  # Ensure confidence is between 0 and 1
                extra_data={
                    'timestamp': datetime.utcnow().isoformat(),
                    'model': getattr(self, 'model_name', 'unknown'),
                    **(metadata or {})
                }
            )
            try:
                # Add and commit the log entry
                self.db.add(log_entry)
                self.db.commit()
                log_id = log_entry.id
                self.logger.debug(f"Logged interaction with ID: {log_id}")
            except Exception as db_error:
                self.logger.error(f"Database error in log_interaction: {db_error}", exc_info=True)
                try:
                    self.db.rollback()
                except Exception as rollback_error:
                    self.logger.error(f"Error during rollback: {rollback_error}", exc_info=True)
        
        except ImportError as import_error:
            self.logger.warning(f"Could not import database models: {import_error}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error in log_interaction: {e}", exc_info=True)
            if hasattr(self, 'db') and self.db is not None:
                try:
                    self.db.rollback()
                except Exception as rollback_error:
                    self.logger.error(f"Failed to rollback database transaction: {rollback_error}", exc_info=True)

    def _log_request(self, text: str, user_id: str) -> None:
        """
        Log a user request to the database.
        
        Args:
            text: The input text from the user
            user_id: ID of the user
        """
        self._log_interaction(
            user_id=user_id,
            input_text=text,
            response="",
            confidence=0.0,
            metadata={"type": "request"}
        )
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of conversation history items
        """
        return self._conversation_history[-limit:]
    
    def clear_conversation_history(self) -> None:
        """
        Clear the conversation history.
        """
        self._conversation_history = []
        self.logger.info("Conversation history cleared")

    def process_pipeline(self, text: str, tasks: List[str]) -> Dict[str, Any]:
        """
        Process input text through a series of tasks.
        
        Args:
            text: Input text to process
            tasks: List of tasks to perform
            
        Returns:
            Dict containing results of all tasks
        """
        self.logger.info(f"Processing pipeline for: {text}")
        results: Dict[str, Any] = {}
        
        for task in tasks:
            try:
                if task == 'sentiment_analysis':
                    if hasattr(self.nlp_analyzer, 'analyze_sentiment'):
                        sentiment = self.nlp_analyzer.analyze_sentiment(text)
                        results['sentiment'] = sentiment
                    else:
                        results['sentiment'] = {"error": "Sentiment analysis not available"}
                        
                elif task == 'entity_recognition':
                    if hasattr(self.nlp_analyzer, 'extract_entities'):
                        entities = self.nlp_analyzer.extract_entities(text)
                        results['entities'] = entities
                    else:
                        results['entities'] = []
                        
                elif task == 'generate_response':
                    response = self.process_input(text)
                    results['response'] = response
                        
                elif task == 'summarize':
                    if hasattr(self.llm, 'summarize'):
                        summary = self.llm.summarize(text)
                        results['summary'] = summary
                    else:
                        results['summary'] = {"error": "Summarization not available"}
                        
            except Exception as e:
                error_msg = f"Error in task '{task}': {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                results[f"{task}_error"] = error_msg
                
        return {
            "input": text,
            "tasks": tasks,
            "results": results,
            "status": "processed",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def generate(self, prompt: str, **kwargs) -> dict:
        """
        Unified generate method for compatibility with UI and API.
        Calls process_input and returns the response dict.
        """
        try:
            # Process the input to get a response
            result = self.process_input(prompt, **kwargs)
            
            # Ensure we have a dict with a 'response' key
            if not isinstance(result, dict) or "response" not in result:
                response_text = str(result)
                result = {"response": response_text, "success": True}
            
            # Clean up the response to ensure it's direct and natural
            response_text = result["response"]
            
            # Remove any meta-commentary about the question
            if response_text.startswith("Interessante vraag over:"):
                response_text = response_text.split(":", 1)[1].strip()
            
            # Remove any surrounding quotes and normalize whitespace
            response_text = response_text.strip("'\" \t\n")
            
            # Update the response in the result
            result["response"] = response_text
            return result
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(f"[generate] {error_msg}", exc_info=True)
            return {"response": error_msg, "success": False}

# --- HyperAdvanced AI Integration ---
# This is an optional integration and will be skipped if not available
HYPERAI_AVAILABLE = False
hyper_ai = None

try:
    import importlib
    from pathlib import Path
    import logging
    import sys
    
    HYPERAI_PATH = Path(__file__).parent.parent / "hyperadvanced_ai"
    if HYPERAI_PATH.exists() and str(HYPERAI_PATH) not in map(str, sys.path):
        sys.path.insert(0, str(HYPERAI_PATH))
    
    try:
        from hyperadvanced_ai.core.abstraction import HyperAIAbstraction
        hyper_ai = HyperAIAbstraction()
        
        # Example: dynamically load all modules listed in config
        try:
            import yaml
            config_path = HYPERAI_PATH / "config" / "default.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    for mod_path in config.get("modules", []):
                        try:
                            hyper_ai.load_module(mod_path)
                        except Exception as e:
                            logging.getLogger("models.jarvis").debug(f"[Optional] Failed to load {mod_path}: {e}")
            HYPERAI_AVAILABLE = True
            logging.getLogger("models.jarvis").info("HyperAdvanced AI integration initialized")
        except ImportError:
            logging.getLogger("models.jarvis").debug("[Optional] PyYAML not available, skipping HyperAdvanced AI config loading")
            
    except ImportError as e:
        logging.getLogger("models.jarvis").debug("[Optional] HyperAdvanced AI not available: " + str(e))
        
except Exception as e:
    logging.getLogger("models.jarvis").debug(f"[Optional] Error initializing HyperAdvanced AI: {e}")
