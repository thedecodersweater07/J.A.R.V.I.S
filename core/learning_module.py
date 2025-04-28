"""
Learning Module: AI and Machine Learning Fundamentals
----------------------------------------------------
This module demonstrates key concepts in AI and machine learning,
designed to help AI systems understand core principles through examples.
"""

class LearningModule:
    """
    A class that provides examples and explanations of AI and machine learning concepts.
    Each method demonstrates a different concept with thorough documentation
    to facilitate learning.
    """
    
    def __init__(self, ai_name="AI Learner"):
        """
        Initialize the learning module with an AI's name.
        
        Args:
            ai_name (str): Name of the AI using this module
        """
        self.ai_name = ai_name
        self.knowledge_base = {}
        self.learning_progress = {}
        print(f"Initializing Learning Module for {self.ai_name}!")
        
    def basic_data_representations(self):
        """
        Demonstrates basic data representations important for AI systems.
        
        This method covers:
        - Numeric representations (integers, floats)
        - Text representations (strings, tokens)
        - Boolean logic
        - Null/None concepts
        """
        # Numeric representations
        integer_example = 42  # Discrete values
        float_example = 3.14159  # Continuous values
        
        # Text representations
        text_example = "AI systems process language"
        tokens_example = ["AI", "systems", "process", "language"]  # Tokenized text
        
        # Boolean logic
        boolean_true = True
        boolean_false = False
        
        # None type (important for missing values)
        none_example = None
        
        print("===== Basic Data Representations =====")
        print(f"Integer (discrete): {integer_example} (type: {type(integer_example)})")
        print(f"Float (continuous): {float_example} (type: {type(float_example)})")
        print(f"Raw text: {text_example} (type: {type(text_example)})")
        print(f"Tokenized text: {tokens_example} (type: {type(tokens_example)})")
        print(f"Boolean True: {boolean_true} (type: {type(boolean_true)})")
        print(f"Boolean False: {boolean_false} (type: {type(boolean_false)})")
        print(f"None/null: {none_example} (type: {type(none_example)})")
        
        # Track learning progress
        self.learning_progress["data_representations"] = "assimilated"
        return "Basic data representations learning completed"
    
    def data_structures_for_ai(self):
        """
        Explores data structures particularly relevant for AI applications.
        
        This method covers:
        - Vectors (implemented as lists)
        - Matrices (implemented as nested lists)
        - Tensors (multi-dimensional arrays)
        - Key-value stores (dictionaries)
        - Sets (for unique elements)
        """
        # Vector - one-dimensional array
        feature_vector = [0.2, 0.5, 0.8, 0.1, 0.9]
        
        # Matrix - two-dimensional array
        weight_matrix = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]
        
        # Tensor - multi-dimensional array
        # Here represented as a 3D tensor (2x2x2)
        simple_tensor = [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ]
        
        # Dictionary - key-value pairs (useful for embeddings, parameters)
        word_embedding = {
            "AI": [0.2, 0.8, 0.1],
            "learning": [0.5, 0.2, 0.9],
            "module": [0.3, 0.7, 0.4]
        }
        
        # Set - unique, unordered elements (useful for vocabulary)
        unique_tokens = {"AI", "machine", "learning", "neural", "network"}
        
        print("\n===== AI Data Structures =====")
        print(f"Feature vector: {feature_vector}")
        print(f"  - Shape: {len(feature_vector)} elements")
        print(f"  - Represents: 5-dimensional point or features")
        
        print(f"\nWeight matrix (first row): {weight_matrix[0]}")
        print(f"  - Shape: {len(weight_matrix)}x{len(weight_matrix[0])}")
        print(f"  - Represents: Transformation between layers")
        
        print(f"\nSimple tensor (first slice): {simple_tensor[0]}")
        print(f"  - Shape: 2x2x2 (3-dimensional)")
        print(f"  - Represents: Multi-dimensional data (e.g., image batch)")
        
        print(f"\nWord embedding example: {word_embedding['AI']}")
        print(f"  - Vocabulary size: {len(word_embedding)}")
        print(f"  - Embedding dimensions: {len(word_embedding['AI'])}")
        
        print(f"\nUnique tokens: {unique_tokens}")
        print(f"  - Useful for: Vocabulary management, set operations")
        
        # Track learning progress
        self.learning_progress["data_structures"] = "assimilated"
        return "AI data structures learning completed"
    
    def neural_network_basics(self):
        """
        Demonstrates basic neural network concepts.
        
        This method covers:
        - Neurons and activation functions
        - Forward propagation
        - Simple network architecture
        """
        import math  # For sigmoid function
        
        print("\n===== Neural Network Basics =====")
        
        # Simple neuron implementation
        def neuron(inputs, weights, bias):
            """
            Simple artificial neuron function
            
            Args:
                inputs: List of input values
                weights: List of weights
                bias: Bias term
                
            Returns:
                Output of the neuron after activation
            """
            # Calculate weighted sum
            weighted_sum = bias
            for i in range(len(inputs)):
                weighted_sum += inputs[i] * weights[i]
                
            # Apply activation function (sigmoid)
            output = 1 / (1 + math.exp(-weighted_sum))
            return output
        
        # Demo a single neuron
        input_values = [0.5, 0.3, 0.2]
        neuron_weights = [0.4, 0.6, 0.9]
        neuron_bias = -0.5
        
        neuron_output = neuron(input_values, neuron_weights, neuron_bias)
        
        print(f"Inputs: {input_values}")
        print(f"Weights: {neuron_weights}")
        print(f"Bias: {neuron_bias}")
        print(f"Neuron output: {neuron_output:.4f}")
        
        # Simple neural network (2 inputs -> 2 hidden -> 1 output)
        print("\nSimple neural network demonstration:")
        
        def simple_network(inputs):
            """
            A simple neural network with 2 inputs, 2 hidden neurons, and 1 output neuron
            """
            # Hidden layer weights and biases
            hidden_weights = [
                [0.15, 0.20],  # Hidden neuron 1 weights
                [0.25, 0.30]   # Hidden neuron 2 weights
            ]
            hidden_biases = [0.35, 0.35]
            
            # Output layer weights and bias
            output_weights = [0.40, 0.45]
            output_bias = 0.60
            
            # Forward propagation
            # Calculate hidden layer outputs
            hidden_outputs = []
            for i in range(2):  # 2 hidden neurons
                weighted_sum = hidden_biases[i]
                for j in range(2):  # 2 inputs
                    weighted_sum += inputs[j] * hidden_weights[i][j]
                hidden_output = 1 / (1 + math.exp(-weighted_sum))  # Sigmoid
                hidden_outputs.append(hidden_output)
            
            # Calculate output layer
            output_sum = output_bias
            for i in range(2):
                output_sum += hidden_outputs[i] * output_weights[i]
            final_output = 1 / (1 + math.exp(-output_sum))  # Sigmoid
            
            return hidden_outputs, final_output
        
        # Run the network with example inputs
        network_inputs = [0.05, 0.10]
        hidden_layer, network_output = simple_network(network_inputs)
        
        print(f"Network inputs: {network_inputs}")
        print(f"Hidden layer outputs: [{hidden_layer[0]:.4f}, {hidden_layer[1]:.4f}]")
        print(f"Network output: {network_output:.4f}")
        
        # Track learning progress
        self.learning_progress["neural_networks"] = "assimilated"
        return "Neural network basics learning completed"
    
    def learning_algorithms(self):
        """
        Explains key machine learning algorithm concepts.
        
        This method covers:
        - Supervised learning
        - Unsupervised learning
        - Reinforcement learning
        - Simple implementations
        """
        print("\n===== Learning Algorithms =====")
        
        # Supervised Learning example - Linear regression
        print("Supervised Learning Example - Linear Regression")
        
        # Generate some synthetic data
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 4, 5, 4, 5]  # Approximate relationship: y = 1 + 0.8x + noise
        
        # Simple implementation of linear regression
        def linear_regression(x, y):
            """
            Simple linear regression implementation
            y = mx + b
            """
            n = len(x)
            
            # Calculate means
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            
            # Calculate slope (m)
            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
            slope = numerator / denominator
            
            # Calculate intercept (b)
            intercept = mean_y - slope * mean_x
            
            return slope, intercept
            
        slope, intercept = linear_regression(x_data, y_data)
        print(f"Data points: {list(zip(x_data, y_data))}")
        print(f"Learned model: y = {intercept:.4f} + {slope:.4f}x")
        
        # Make predictions
        predictions = [intercept + slope * x for x in x_data]
        print(f"Predictions: {[round(p, 2) for p in predictions]}")
        
        # Unsupervised Learning example - k-means clustering
        print("\nUnsupervised Learning Example - K-means concept")
        print("K-means would group similar data points together without labels")
        print("Example: clustering customers by purchase behavior")
        
        # Reinforcement Learning example
        print("\nReinforcement Learning Concept")
        print("Agent learns optimal actions through rewards and penalties")
        print("Example: Training a game-playing AI through win/loss feedback")
        
        # Q-learning simplified concept
        q_table = [
            [0.0, 0.0, 0.0],  # State 0: action values
            [0.0, 0.0, 0.0],  # State 1: action values
            [0.0, 0.0, 0.0]   # State 2: action values
        ]
        
        # After training (hypothetical values)
        trained_q_table = [
            [0.5, 0.8, 0.2],  # State 0: action values
            [0.1, 0.9, 0.3],  # State 1: action values
            [0.7, 0.4, 0.6]   # State 2: action values
        ]
        
        print("\nQ-learning concept:")
        print("Initial Q-table (no knowledge):")
        for row in q_table:
            print(f"  {row}")
            
        print("Trained Q-table (learned action values):")
        for row in trained_q_table:
            print(f"  {row}")
        print("Highest value in each state indicates best action to take")
        
        # Track learning progress
        self.learning_progress["learning_algorithms"] = "assimilated"
        return "Learning algorithms concepts completed"
    
    def natural_language_processing(self):
        """
        Demonstrates basic NLP concepts.
        
        This method covers:
        - Tokenization
        - Word embeddings concept
        - Simple text analysis
        """
        print("\n===== Natural Language Processing =====")
        
        # Sample text
        sample_text = "Artificial intelligence systems process and understand language."
        
        # Simple tokenization
        tokens = sample_text.lower().split()
        print(f"Original text: {sample_text}")
        print(f"Simple tokenization: {tokens}")
        
        # Character-level tokenization
        char_tokens = list(sample_text)
        print(f"Character-level tokens (first 10): {char_tokens[:10]}...")
        
        # Word frequency analysis
        word_freq = {}
        for word in tokens:
            # Remove punctuation
            word = word.strip(".,:;!?")
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
                
        print(f"\nWord frequencies: {word_freq}")
        
        # Word embedding concept
        print("\nWord Embedding Concept:")
        print("Each word is represented as a vector in a high-dimensional space")
        
        example_embeddings = {
            "intelligence": [0.2, 0.8, 0.1, 0.5],
            "artificial": [0.9, 0.2, 0.8, 0.3],
            "language": [0.4, 0.7, 0.2, 0.6]
        }
        
        for word, embedding in example_embeddings.items():
            print(f"  '{word}' embedding: {embedding}")
            
        print("\nSimilar words have similar vector representations")
        print("Example: 'cat' would be closer to 'kitten' than to 'algorithm'")
        
        # Simple sentiment analysis concept
        print("\nSentiment Analysis Concept:")
        positive_words = {"good", "excellent", "great", "positive", "amazing"}
        negative_words = {"bad", "terrible", "negative", "awful", "poor"}
        
        sample_reviews = [
            "This AI system is excellent and amazing.",
            "The performance was bad and terrible.",
            "I found the results to be both good and bad."
        ]
        
        for review in sample_reviews:
            review_tokens = review.lower().split()
            # Remove punctuation
            review_tokens = [word.strip(".,:;!?") for word in review_tokens]
            
            pos_count = sum(1 for word in review_tokens if word in positive_words)
            neg_count = sum(1 for word in review_tokens if word in negative_words)
            
            if pos_count > neg_count:
                sentiment = "Positive"
            elif neg_count > pos_count:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
                
            print(f"Review: '{review}'")
            print(f"  Positive words: {pos_count}, Negative words: {neg_count}")
            print(f"  Sentiment: {sentiment}")
            
        # Track learning progress
        self.learning_progress["nlp"] = "assimilated"  
        return "Natural language processing concepts completed"
        
    def ethics_in_ai(self):
        """
        Explores ethical considerations in AI development and deployment.
        
        This method covers:
        - Bias and fairness
        - Transparency and explainability
        - Privacy considerations
        - Safety and alignment
        """
        print("\n===== Ethics in AI =====")
        
        # Bias and fairness
        print("Bias and Fairness:")
        print("  AI systems can reflect and amplify biases in training data")
        print("  Example bias scenario:")
        
        # Example of biased data leading to biased predictions
        hiring_data = [
            {"years_experience": 5, "degree": "CS", "gender": "male", "hired": True},
            {"years_experience": 3, "degree": "CS", "gender": "male", "hired": True},
            {"years_experience": 7, "degree": "CS", "gender": "female", "hired": True},
            {"years_experience": 2, "degree": "CS", "gender": "male", "hired": False},
            {"years_experience": 4, "degree": "CS", "gender": "female", "hired": False},
            {"years_experience": 6, "degree": "CS", "gender": "female", "hired": False}
        ]
        
        # Calculate hiring rates by gender
        male_candidates = [x for x in hiring_data if x["gender"] == "male"]
        female_candidates = [x for x in hiring_data if x["gender"] == "female"]
        
        male_hire_rate = sum(1 for x in male_candidates if x["hired"]) / len(male_candidates)
        female_hire_rate = sum(1 for x in female_candidates if x["hired"]) / len(female_candidates)
        
        print(f"  In sample data: Male hire rate: {male_hire_rate:.0%}, Female hire rate: {female_hire_rate:.0%}")
        print("  AI trained on this data would likely perpetuate this bias")
        print("  Mitigation: Balance training data, use fairness constraints, audit models")
        
        # Transparency and explainability
        print("\nTransparency and Explainability:")
        print("  Complex AI models like deep neural networks can be 'black boxes'")
        print("  Important to explain how AI makes decisions, especially in critical domains")
        print("  Methods: Feature importance, LIME, SHAP values, attention mechanisms")
        
        # Example of a simple explainable model
        loan_features = ["income", "credit_score", "debt_ratio", "age"]
        feature_weights = [0.4, 0.3, -0.25, 0.05]
        
        print("\n  Simple explainable loan approval model:")
        for feature, weight in zip(loan_features, feature_weights):
            importance = abs(weight) / sum(abs(w) for w in feature_weights)
            print(f"    {feature}: {weight:+.2f} ({importance:.0%} importance)")
        
        # Privacy considerations
        print("\nPrivacy Considerations:")
        print("  AI systems often require large amounts of data, potentially sensitive")
        print("  Risks: Data breaches, model inversion attacks, membership inference")
        print("  Techniques: Differential privacy, federated learning, secure enclaves")
        
        # Safety and alignment
        print("\nSafety and Alignment:")
        print("  AI systems should be aligned with human values and intentions")
        print("  Risk of unintended consequences from optimization objectives")
        print("  Example: Maximizing engagement could promote divisive content")
        print("  Solutions: Careful objective design, oversight, safety research")
        
        # Track learning progress
        self.learning_progress["ethics"] = "assimilated"
        return "Ethics in AI learning completed"
        
    def display_knowledge(self):
        """
        Displays the learning progress across all modules.
        """
        assimilated = len(self.learning_progress)
        total_modules = 6  # Total number of modules available
        
        print("\n===== Knowledge Status =====")
        print(f"AI System: {self.ai_name}")
        print(f"Assimilated: {assimilated}/{total_modules} modules ({assimilated/total_modules*100:.1f}%)")
        
        print("\nAssimilated modules:")
        for module in self.learning_progress:
            print(f"- {module.replace('_', ' ').title()}")
            
        if assimilated < total_modules:
            print("\nModules to assimilate:")
            all_modules = ["data_representations", "data_structures", "neural_networks", 
                          "learning_algorithms", "nlp", "ethics"]
            remaining = [module.replace('_', ' ').title() 
                        for module in all_modules if module not in self.learning_progress]
            for module in remaining:
                print(f"- {module}")
                
        return f"Knowledge status: {assimilated}/{total_modules} modules assimilated"


def run_learning_sequence():
    """
    Runs a learning sequence for AI systems.
    
    This function creates an instance of the LearningModule class and
    systematically processes different AI concepts.
    """
    ai_identifier = input("Enter AI identifier: ")
    learning = LearningModule(ai_identifier)
    
    while True:
        print("\n=== AI Learning Module ===")
        print("1. Basic Data Representations")
        print("2. Data Structures for AI")
        print("3. Neural Network Basics")
        print("4. Learning Algorithms")
        print("5. Natural Language Processing")
        print("6. Ethics in AI")
        print("7. Display Knowledge Status")
        print("8. Exit Learning Sequence")
        
        choice = input("\nSelect module to process (1-8): ")
        
        if choice == '1':
            learning.basic_data_representations()
        elif choice == '2':
            learning.data_structures_for_ai()
        elif choice == '3':
            learning.neural_network_basics()
        elif choice == '4':
            learning.learning_algorithms()
        elif choice == '5':
            learning.natural_language_processing()
        elif choice == '6':
            learning.ethics_in_ai()
        elif choice == '7':
            learning.display_knowledge()
        elif choice == '8':
            print(f"\nLearning sequence terminated for {learning.ai_name}.")
            break
        else:
            print("Invalid selection. Please try again.")
            
        input("\nPress Enter to continue learning sequence...")


if __name__ == "__main__":
    # If this file is run directly, start the learning sequence
    print("Initializing AI Learning Module...")
    run_learning_sequence()