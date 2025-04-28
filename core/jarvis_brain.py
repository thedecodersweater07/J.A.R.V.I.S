# jarvis_brain.py
# Main controller for the Jarvis AI system

import logging
import time
from threading import Thread

# Import system modules
try:
    from decision_engine import DecisionEngine
    from learning_module import LearningModule
    from memory_manager import MemoryManager
    from command.context_analyzer import ContextAnalyzer
    from command.executor import Executor
    from command.parser import Parser
    from command.priority_handler import PriorityHandler
    from command.feedback import FeedbackSystem
    
    # Security modules
    from security.encryption import EncryptionSystem
    from security.firewall import Firewall
    from security.intrusion_detection import IntrusionDetection
    from security.self_protect import SelfProtectionSystem
    from security.threat_analyzer import ThreatAnalyzer
    
except ImportError as e:
    logging.error(f"Failed to import module: {e}")
    exit(1)

class JarvisBrain:
    """Main controller for the Jarvis AI system"""
    
    def __init__(self, config_path="config/jarvis_config.json"):
        """Initialize the Jarvis brain system"""
        self.name = "J.A.R.V.I.S"
        self.version = "1.0.0"
        self.running = False
        self.config_path = config_path
        self.startup_time = None
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='logs/jarvis.log'
        )
        
        self.logger = logging.getLogger("JarvisBrain")
        self.logger.info(f"Initializing {self.name} v{self.version}")
        
        try:
            # Initialize core components
            self.memory = MemoryManager()
            self.parser = Parser()
            self.context = ContextAnalyzer()
            self.decision_engine = DecisionEngine()
            self.executor = Executor()
            self.priority = PriorityHandler()
            self.learning = LearningModule()
            self.feedback = FeedbackSystem()
            
            # Initialize security components
            self.encryption = EncryptionSystem()
            self.firewall = Firewall()
            self.intrusion = IntrusionDetection()
            self.self_protection = SelfProtectionSystem()
            self.threat_analyzer = ThreatAnalyzer()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.critical(f"Initialization failed: {e}")
            raise
    
    def start(self):
        """Start the Jarvis AI system"""
        if self.running:
            self.logger.warning("Jarvis is already running")
            return False
        
        self.logger.info(f"Starting {self.name} v{self.version}")
        try:
            # Start security systems first
            self.encryption.start()
            self.firewall.start()
            self.intrusion.start()
            self.self_protection.start()
            self.threat_analyzer.start()
            
            # Start core components
            self.memory.start()
            self.context.start()
            self.decision_engine.start()
            self.executor.start()
            self.priority.start()
            self.learning.start()
            self.feedback.start()
            
            # Mark system as running
            self.running = True
            self.startup_time = time.time()
            self.logger.info(f"{self.name} started successfully")
            
            # Start the main processing loop in a separate thread
            self._processing_thread = Thread(target=self._main_loop)
            self._processing_thread.daemon = True
            self._processing_thread.start()
            
            return True
        
        except Exception as e:
            self.logger.critical(f"Failed to start Jarvis: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Stop the Jarvis AI system"""
        if not self.running:
            self.logger.warning("Jarvis is not running")
            return False
        
        self.logger.info(f"Stopping {self.name}")
        try:
            # Stop core components in reverse order
            self.feedback.stop()
            self.learning.stop()
            self.priority.stop()
            self.executor.stop()
            self.decision_engine.stop()
            self.context.stop()
            self.memory.stop()
            
            # Stop security components
            self.threat_analyzer.stop()
            self.self_protection.stop()
            self.intrusion.stop()
            self.firewall.stop()
            self.encryption.stop()
            
            # Mark system as stopped
            self.running = False
            self.logger.info(f"{self.name} stopped successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False
    
    def _main_loop(self):
        """Main processing loop for Jarvis"""
        self.logger.info("Main processing loop started")
        try:
            while self.running:
                # Process any pending tasks or commands
                self.process_cycle()
                time.sleep(0.1)  # Small sleep to prevent CPU hogging
        except Exception as e:
            self.logger.error(f"Error in main processing loop: {e}")
            self.stop()
    
    def process_cycle(self):
        """Run a single processing cycle"""
        if not self.running:
            return
        
        try:
            # Check for security threats
            threats = self.threat_analyzer.check_threats()
            if threats:
                self.handle_threats(threats)
                
            # Get input and context
            input_data = self.parser.get_pending_input()
            if input_data:
                context = self.context.analyze(input_data)
                priority = self.priority.determine(input_data, context)
                
                # Make decisions
                decisions = self.decision_engine.process(input_data, context, priority)
                
                # Execute decisions
                results = self.executor.execute(decisions)
                
                # Learn from interaction
                self.learning.learn_from_interaction(input_data, context, decisions, results)
                
                # Store in memory
                self.memory.store_interaction(input_data, context, decisions, results)
                
        except Exception as e:
            self.logger.error(f"Error during processing cycle: {e}")
    
    def handle_threats(self, threats):
        """Handle detected security threats"""
        for threat in threats:
            self.logger.warning(f"Handling security threat: {threat['type']}")
            self.self_protection.respond_to_threat(threat)
    
    def get_status(self):
        """Get the current status of the Jarvis system"""
        status = {
            "name": self.name,
            "version": self.version,
            "running": self.running,
            "uptime": time.time() - self.startup_time if self.startup_time else 0,
            "components": {
                "memory": self.memory.get_status(),
                "parser": self.parser.get_status(),
                "context": self.context.get_status(),
                "decision_engine": self.decision_engine.get_status(),
                "executor": self.executor.get_status(),
                "priority": self.priority.get_status(),
                "learning": self.learning.get_status(),
                "feedback": self.feedback.get_status(),
                "security": {
                    "encryption": self.encryption.get_status(),
                    "firewall": self.firewall.get_status(),
                    "intrusion": self.intrusion.get_status(),
                    "self_protection": self.self_protection.get_status(),
                    "threat_analyzer": self.threat_analyzer.get_status()
                }
            }
        }
        return status

# Run the Jarvis system if script is executed directly
if __name__ == "__main__":
    print(f"Starting J.A.R.V.I.S system...")
    
    try:
        # Create and start the Jarvis brain
        jarvis = JarvisBrain()
        if jarvis.start():
            print(f"J.A.R.V.I.S v{jarvis.version} is now running")
            
            # Keep the main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Shutting down J.A.R.V.I.S...")
                jarvis.stop()
                print("J.A.R.V.I.S has been shut down")
        else:
            print("Failed to start J.A.R.V.I.S")
    
    except Exception as e:
        print(f"Critical error: {e}")
        logging.critical(f"Critical startup error: {e}")