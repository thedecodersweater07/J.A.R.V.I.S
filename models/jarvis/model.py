import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ..base import BaseModel
from .config import JarvisConfig, JARVIS_CONFIGS
from ..gpt.transformer import TransformerLayer
from ml.models.model_manager import ModelManager
from nlp.pipeline import NLPPipeline
from llm.core import LLMCore
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class JarvisModel(BaseModel):
    def __init__(self, config_name: str):
        if config_name not in JARVIS_CONFIGS:
            raise ValueError(f"Unknown model config: {config_name}")
            
        config = JARVIS_CONFIGS[config_name]
        super().__init__(config.__dict__)
        
        # Core components
        self.embeddings = JarvisEmbeddings(config)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.hidden_dropout_prob
            ) for _ in range(config.num_hidden_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Task-specific heads can be added here
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Integration components
        self.model_manager = ModelManager()
        self.nlp_pipeline = NLPPipeline()
        self.llm_core = LLMCore()
        
        # Task routing with dynamic head sizes
        self.task_heads = nn.ModuleDict({
            'classification': nn.ModuleDict({
                task: nn.Linear(config.hidden_size, num_classes)
                for task, num_classes in config.classification_classes.items()
            }),
            'generation': self.lm_head,
            'embedding': nn.Linear(config.hidden_size, config.embedding_size),
            'qa': nn.Linear(config.hidden_size, 2)  # start/end position
        })
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        
        self.init_integrations()
        
    def init_integrations(self):
        """Initialize integration with ML/NLP/LLM components"""
        # Connect ML models
        self.ml_models = {}
        model_types = ['classifier', 'regressor', 'clustering']
        
        for model_type in model_types:
            try:
                model = self.model_manager.load_model(model_type)
                if model:
                    self.ml_models[model_type] = model
                else:
                    logger.warning(f"Could not load {model_type} model")
            except Exception as e:
                logger.error(f"Error loading {model_type} model: {e}")
                continue
                
        # Setup NLP pipeline
        self.nlp_processors = {
            'tokenizer': self.nlp_pipeline.get_tokenizer(),
            'parser': self.nlp_pipeline.get_parser(),
            'ner': self.nlp_pipeline.get_ner()
        }
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        hidden_states = self.embeddings(input_ids)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
        
        all_hidden_states = []
        all_attentions = []
        
        all_hidden_states = []
        all_attentions = []
        
        for layer in self.layers:
            hidden_states, attention_weights = layer(hidden_states)
            all_hidden_states.append(hidden_states)
            all_attentions.append(attention_weights)
            
        hidden_states = self.layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return {
            "logits": logits,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions
        }

    def route_task(self, task_name: str, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Route inputs to appropriate task head"""
        outputs = self.forward(inputs['input_ids'], inputs.get('attention_mask'))
        hidden_states = outputs['hidden_states'][-1]
        
        if task_name not in self.task_heads:
            raise ValueError(f"Unknown task: {task_name}")
            
    def process_pipeline(self, text: str, tasks: List[str]) -> Dict[str, Any]:
        """Process text through multiple pipeline stages"""
        results = {}
        
        try:
            # NLP preprocessing
            tokens = self.nlp_processors['tokenizer'](text)
            results['parsed'] = self.nlp_processors['parser'](tokens)
            results['entities'] = self.nlp_processors['ner'](tokens)
            
            # Convert to tensor inputs
            inputs = self.prepare_inputs(text)  # Changed from tokens to text
            
            # Process each requested task
            for task in tasks:
                try:
                    results[task] = self.route_task(task, inputs)
                except Exception as e:
                    logger.error(f"Error processing task {task}: {e}")
                    results[task] = {"error": str(e)}
            
            # Post-process with ML models if needed
            if 'ml_analysis' in tasks:
                ml_features = self.extract_features(results)
                
            return results
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {"error": str(e)}

    def prepare_inputs(self, text: str) -> Dict[str, torch.Tensor]:
        """Convert text to model inputs"""
        # Tokenize with padding and truncation
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to model device if needed
        if hasattr(self, 'device'):
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
        return encoded

    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text from prompt"""
        # Tokenize input
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        # Generate
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs["logits"]
            
            # Sample from logits
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            generated = [next_token.item()]
            current_length = 1
            
            while current_length < max_length:
                current_input = torch.cat([input_ids, torch.tensor([generated]).to(self.device)], dim=1)
                current_mask = torch.ones_like(current_input)
                
                outputs = self.forward(current_input, current_mask)
                next_token_logits = outputs["logits"][:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                generated.append(next_token.item())
                current_length += 1
                
                if next_token.item() == self.config.get("eos_token_id", 0):
                    break
                    
        # Decode generated tokens
        return self.tokenizer.decode(generated)

class JarvisEmbeddings(nn.Module):
    def __init__(self, config: JarvisConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = word_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
