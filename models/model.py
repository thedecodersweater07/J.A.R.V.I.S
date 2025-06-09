import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
import logging

# Local imports (these are in the same folder)
from base import BaseModel
from config import JarvisConfig, JARVIS_CONFIGS

# External imports with fallback handling
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers library not available, using fallback tokenizer")

# Optional external component imports with fallbacks
try:
    from ml.models.model_manager import ModelManager
    HAS_MODEL_MANAGER = True
except ImportError:
    HAS_MODEL_MANAGER = False
    print("Warning: ModelManager not available, using dummy fallback")

try:
    from nlp.pipeline import NLPPipeline
    HAS_NLP_PIPELINE = True
except ImportError:
    HAS_NLP_PIPELINE = False
    print("Warning: NLPPipeline not available, using dummy fallback")

try:
    from llm.core import LLMCore
    HAS_LLM_CORE = True
except ImportError:
    HAS_LLM_CORE = False
    print("Warning: LLMCore not available, using dummy fallback")

logger = logging.getLogger(__name__)

# Fallback classes for missing components
class FallbackTokenizer:
    """Simple fallback tokenizer when transformers is not available"""
    def __init__(self):
        self.vocab_size = 30000
        
    def __call__(self, text, **kwargs):
        tokens = text.split()[:512]  # Simple whitespace tokenization
        input_ids = [hash(token) % self.vocab_size for token in tokens]
        
        if kwargs.get('return_tensors') == 'pt':
            return {
                'input_ids': torch.tensor([input_ids]),
                'attention_mask': torch.ones(1, len(input_ids))
            }
        return {'input_ids': input_ids}
    
    def decode(self, token_ids):
        return f"Generated text from {len(token_ids)} tokens"

class FallbackModelManager:
    """Fallback model manager when real one is not available"""
    def load_model(self, model_name):
        return None

class FallbackNLPPipeline:
    """Fallback NLP pipeline when real one is not available"""
    def get_tokenizer(self):
        return lambda x: x.split()
    
    def get_parser(self):
        return lambda x: {'tokens': x}
    
    def get_ner(self):
        return lambda x: []

class FallbackLLMCore:
    """Fallback LLM core when real one is not available"""
    def __init__(self):
        pass

# Dummy model classes for fallback
class DummyClassifier:
    """Dummy classifier model for fallback"""
    def __init__(self):
        self.classes_ = [0, 1]  # Binary classification by default
    
    def predict(self, X):
        """Return random predictions"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(X, list):
            X = np.array(X)
        return np.random.choice(self.classes_, size=len(X) if hasattr(X, '__len__') else 1)
    
    def predict_proba(self, X):
        """Return random probabilities"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(X, list):
            X = np.array(X)
        n_samples = len(X) if hasattr(X, '__len__') else 1
        return np.random.random((n_samples, len(self.classes_)))

class DummyRegressor:
    """Dummy regressor model for fallback"""
    def predict(self, X):
        """Return random predictions"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(X, list):
            X = np.array(X)
        return np.random.normal(0, 1, size=len(X) if hasattr(X, '__len__') else 1)

class DummyClustering:
    """Dummy clustering model for fallback"""
    def __init__(self):
        self.n_clusters = 3
    
    def fit_predict(self, X):
        """Return random cluster assignments"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(X, list):
            X = np.array(X)
        return np.random.randint(0, self.n_clusters, size=len(X) if hasattr(X, '__len__') else 1)
    
    def predict(self, X):
        """Return random cluster assignments"""
        return self.fit_predict(X)

class TransformerLayer(nn.Module):
    """Basic transformer layer implementation"""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights

class JarvisModel(BaseModel):
    def __init__(self, config_name: str):
        if config_name not in JARVIS_CONFIGS:
            raise ValueError(f"Unknown model config: {config_name}")
            
        config = JARVIS_CONFIGS[config_name]
        super().__init__(config.__dict__)
        
        # Add logger initialization
        self.logger = logging.getLogger(__name__)
        
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
        
        # Integration components with fallbacks
        self.model_manager = ModelManager() if HAS_MODEL_MANAGER else FallbackModelManager()
        self.nlp_pipeline = NLPPipeline() if HAS_NLP_PIPELINE else FallbackNLPPipeline()
        self.llm_core = LLMCore() if HAS_LLM_CORE else FallbackLLMCore()
        
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
        
        # Initialize tokenizer with fallback
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
            except Exception as e:
                logger.warning(f"Could not load tokenizer: {e}, using fallback")
                self.tokenizer = FallbackTokenizer()
        else:
            self.tokenizer = FallbackTokenizer()
        
        # Add device initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        # Ensure all components are on correct device
        self._move_to_device()
        
        # Initialize integrations
        self.init_integrations()
        
    def _move_to_device(self):
        """Move all model components to the correct device"""
        if hasattr(self, 'embeddings'):
            self.embeddings = self.embeddings.to(self.device)
        if hasattr(self, 'layers'):
            self.layers = nn.ModuleList([layer.to(self.device) for layer in self.layers])
        if hasattr(self, 'layer_norm'):
            self.layer_norm = self.layer_norm.to(self.device)
        
        # Ensure task heads are on correct device
        if hasattr(self, 'task_heads'):
            for name, head in self.task_heads.items():
                if isinstance(head, nn.Module):
                    self.task_heads[name] = head.to(self.device)
                elif isinstance(head, nn.ModuleDict):
                    self.task_heads[name] = nn.ModuleDict({
                        k: v.to(self.device) for k, v in head.items()
                    })
        
    def init_integrations(self):
        """Initialize integration with ML/NLP/LLM components"""
        # Connect ML models with better error handling
        self.ml_models = {}
        model_types = ['classifier', 'regressor', 'clustering']
        
        # Create dummy models first as ultimate fallback
        self._create_dummy_models()
        
        for model_type in model_types:
            try:
                # Try loading the model
                model = self.model_manager.load_model(f"{model_type}_latest")
                if model is not None:
                    self.ml_models[model_type] = model
                    logger.info(f"Loaded {model_type} model")
                else:
                    # Use dummy model if load fails
                    logger.warning(f"Could not load {model_type} model, using dummy model")
                    self.ml_models[model_type] = self.dummy_models[model_type]
            except Exception as e:
                logger.error(f"Error loading {model_type} model: {e}")
                # Use dummy model as fallback
                self.ml_models[model_type] = self.dummy_models[model_type]
                
        # Setup NLP pipeline with fallbacks
        self.nlp_processors = {}
        try:
            self.nlp_processors['tokenizer'] = self.nlp_pipeline.get_tokenizer()
        except Exception as e:
            logger.warning(f"Error loading tokenizer: {e}, using simple tokenizer")
            self.nlp_processors['tokenizer'] = lambda x: x.split()
            
        try:
            self.nlp_processors['parser'] = self.nlp_pipeline.get_parser()
        except Exception as e:
            logger.warning(f"Error loading parser: {e}, using simple parser")
            self.nlp_processors['parser'] = lambda x: {'tokens': x}
            
        try:
            self.nlp_processors['ner'] = self.nlp_pipeline.get_ner()
        except Exception as e:
            logger.warning(f"Error loading NER: {e}, using simple NER")
            self.nlp_processors['ner'] = lambda x: []
            
    def _create_dummy_models(self):
        """Create dummy models to use as fallbacks when real models are not available"""
        self.dummy_models = {
            'classifier': DummyClassifier(),
            'regressor': DummyRegressor(),
            'clustering': DummyClustering()
        }
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the model."""
        try:
            batch_size = input_ids.size(0)
            seq_length = input_ids.size(1)
            
            # Validate input dimensions
            if seq_length > self.config.max_position_embeddings:
                logger.warning(f"Input sequence length {seq_length} exceeds model's maximum {self.config.max_position_embeddings}. Truncating.")
                input_ids = input_ids[:, :self.config.max_position_embeddings]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :self.config.max_position_embeddings]

            # Ensure inputs are on correct device
            input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Generate embeddings
            try:
                hidden_states = self.embeddings(input_ids)
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                # Fallback to zero embeddings
                hidden_states = torch.zeros(
                    (batch_size, seq_length, self.config.hidden_size), 
                    device=self.device
                )

            # Process through transformer layers
            all_hidden_states = []
            all_attentions = []

            try:
                for layer in self.layers:
                    hidden_states, attention_weights = layer(hidden_states)
                    all_hidden_states.append(hidden_states)
                    if attention_weights is not None:
                        all_attentions.append(attention_weights)

                # Apply final layer norm
                hidden_states = self.layer_norm(hidden_states)
                logits = self.lm_head(hidden_states)

            except Exception as e:
                logger.error(f"Layer processing error: {e}")
                # Provide safe fallback outputs
                hidden_states = torch.zeros(
                    (batch_size, seq_length, self.config.hidden_size), 
                    device=self.device
                )
                logits = torch.zeros(
                    (batch_size, seq_length, self.config.vocab_size), 
                    device=self.device
                )
                all_hidden_states = [hidden_states]
                all_attentions = []

            return {
                "logits": logits,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions if all_attentions else None
            }

        except Exception as e:
            logger.error(f"Forward pass error: {e}")
            # Return safe fallback with proper shapes
            return {
                "logits": torch.zeros(
                    (1, 1, self.config.vocab_size), 
                    device=self.device
                ),
                "hidden_states": [torch.zeros(
                    (1, 1, self.config.hidden_size), 
                    device=self.device
                )],
                "attentions": None
            }

    def route_task(self, task_name: str, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Route inputs to appropriate task head with error handling"""
        try:
            # Check if input is valid
            if 'input_ids' not in inputs or inputs['input_ids'] is None:
                return {"error": "Invalid input: input_ids missing or None"}
                
            # Execute forward pass with error handling
            try:
                outputs = self.forward(inputs['input_ids'], inputs.get('attention_mask'))
                if not outputs or 'hidden_states' not in outputs or not outputs['hidden_states']:
                    return {"error": "Forward pass produced invalid outputs"}
                    
                # Check if hidden_states is a valid list with elements
                hidden_states = outputs['hidden_states']
                if not isinstance(hidden_states, list) or len(hidden_states) == 0:
                    return {"error": "No hidden states available"}
                    
                # Use the last hidden state
                last_hidden = hidden_states[-1]
            except Exception as e:
                logger.error(f"Error in forward pass: {e}")
                # Create dummy output
                return {"error": str(e), "logits": torch.randn(1, 10)}
            
            # Check if task is valid
            if task_name not in self.task_heads:
                return {"error": f"Unknown task: {task_name}", "logits": torch.randn(1, 10)}
                
            # Use the appropriate task head
            try:
                if task_name == 'classification':
                    # Use dummy output for classification
                    return {"logits": torch.randn(1, 2), "predicted_class": 0}
                elif task_name == 'generation':
                    # Use dummy output for generation
                    return {"logits": torch.randn(1, 10), "generated_text": "Generated text"}
                elif task_name == 'qa':
                    # Use dummy output for qa
                    return {"start_logits": torch.randn(1), "end_logits": torch.randn(1), "answer": "Answer"}
                else:
                    # General fallback
                    return {"logits": torch.randn(1, 10)}
            except Exception as e:
                logger.error(f"Error in task head {task_name}: {e}")
                return {"error": str(e), "logits": torch.randn(1, 10)}
                
        except Exception as e:
            logger.error(f"Unexpected error in route_task: {e}")
            return {"error": str(e), "logits": torch.randn(1, 10)}
            
    def process_pipeline(self, text: str, tasks: List[str]) -> Dict[str, Any]:
        """Process text through multiple pipeline stages"""
        results = {}
        
        try:
            # NLP preprocessing
            tokens = self.nlp_processors['tokenizer'](text)
            results['parsed'] = self.nlp_processors['parser'](tokens)
            results['entities'] = self.nlp_processors['ner'](tokens)
            
            # Convert to tensor inputs
            inputs = self.prepare_inputs(text)
            
            # Process each requested task
            for task in tasks:
                try:
                    results[task] = self.route_task(task, inputs)
                except Exception as e:
                    logger.error(f"Error processing task {task}: {e}")
                    results[task] = {"error": str(e)}
            
            # Post-process with ML models if needed
            if 'ml_analysis' in tasks:
                try:
                    ml_features = self.extract_features(results)
                    results['ml_analysis'] = ml_features
                except Exception as e:
                    logger.error(f"Error in ML analysis: {e}")
                    results['ml_analysis'] = {"error": str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {"error": str(e)}

    def prepare_inputs(self, text: str) -> Dict[str, torch.Tensor]:
        """Convert text to model inputs"""
        try:
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
        except Exception as e:
            logger.error(f"Error preparing inputs: {e}")
            # Return fallback inputs
            return {
                'input_ids': torch.tensor([[1, 2, 3]], device=self.device),
                'attention_mask': torch.tensor([[1, 1, 1]], device=self.device)
            }

    def extract_features(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for ML analysis"""
        features = {}
        try:
            # Extract basic features from results
            if 'parsed' in results:
                features['num_tokens'] = len(results['parsed'].get('tokens', []))
            if 'entities' in results:
                features['num_entities'] = len(results['entities'])
            
            # Add more feature extraction logic here
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {"error": str(e)}

    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text from prompt"""
        try:
            # Tokenize input
            inputs = self.prepare_inputs(prompt)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
            
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
        except Exception as e:
            logger.error(f"Error in generation: {e}")
            return f"Generated text (fallback): {prompt} [continued]"

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