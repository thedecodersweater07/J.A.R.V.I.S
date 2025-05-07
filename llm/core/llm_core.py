from typing import Optional, Dict, Any
import torch
from pathlib import Path
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..memory.enhanced_memory import EnhancedMemoryManager
from ..optimization.llm_optimizer import LLMOptimizer
from db.database_manager import DatabaseManager

class LLMCore:
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.db = DatabaseManager(self.config["database"])
        self.memory = EnhancedMemoryManager(self.config["llm"]["memory"])
        self.optimizer = LLMOptimizer(self.config["llm"]["model"]["optimization"])
        
        self._initialize_model()

    def _initialize_model(self):
        model_name = self.config["llm"]["model"]["name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Apply optimizations
        self.model = self.optimizer.optimize_model(self.model)
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def generate_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        # Get relevant context from memory
        context_data = self.memory.get_context(prompt) if context is None else context
        
        # Prepare input with context
        enhanced_prompt = self._prepare_prompt(prompt, context_data)
        
        # Generate response
        inputs = self.tokenizer(enhanced_prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"].to(self.model.device),
            max_length=self.config["llm"]["model"]["max_length"],
            temperature=self.config["llm"]["model"]["temperature"],
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Store interaction in memory
        self.memory.store_interaction(prompt, response, context_data)
        
        return response

    def _prepare_prompt(self, prompt: str, context: Dict) -> str:
        if not context:
            return prompt
        
        context_str = "\n".join([
            f"Previous [{c['category']}]: {c['text']}"
            for c in context.get("relevant_history", [])
        ])
        
        return f"{context_str}\nCurrent: {prompt}"

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "data" / "config.yaml"
        
        with open(config_path) as f:
            return yaml.safe_load(f)