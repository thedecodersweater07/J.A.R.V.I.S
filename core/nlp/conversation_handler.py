import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
import random
from datetime import datetime
from core.nlp.language_processor import LanguageProcessor

logger = logging.getLogger(__name__)

@dataclass
class DialogContext:
    last_intent: str
    conversation_history: List[Dict]
    user_mood: str
    jarvis_mood: str
    timestamp: datetime
    topic: str

class ConversationHandler:
    def __init__(self, language_processor: LanguageProcessor, knowledge_base=None):
        self.nlp = language_processor
        self.knowledge_base = knowledge_base
        self.context = DialogContext(
            last_intent="",
            conversation_history=[],
            user_mood="neutral",
            jarvis_mood="helpful",
            timestamp=datetime.now(),
            topic="general"
        )
        self._init_response_templates()

    def _init_response_templates(self):
        self.responses = {
            "greeting": [
                "Hallo! Fijn je te spreken. Hoe kan ik je helpen?",
                "Hey! Wat leuk je te zien. Waar kan ik je mee helpen?",
                "Goedendag! Ik sta klaar om je te assisteren."
            ],
            "how_are_you": [
                "Met mij gaat het uitstekend! Ik leer elke dag bij en word steeds slimmer. Hoe gaat het met jou?",
                "Heel goed! Ik ben enthousiast om je te helpen. En met jou?",
                "Prima! Ik ben net geüpdatet met nieuwe kennis en klaar om je te assisteren!"
            ],
            "about_me": [
                "Ik ben JARVIS, een geavanceerd AI-systeem gebaseerd op het nieuwste in machine learning en natuurlijke taalverwerking.",
                "Als je persoonlijke AI-assistent help ik je graag met allerlei taken, van eenvoudige vragen tot complexe analyses."
            ],
            "unknown": [
                "Interessant! Kun je me daar meer over vertellen?",
                "Ik begrijp je vraag. Laat me even nadenken over het beste antwoord.",
                "Dat is een goede vraag. Laat me mijn kennis raadplegen voor een goed antwoord."
            ]
        }

    def process_input(self, user_input: str) -> str:
        try:
            # Clean and normalize input
            processed_input = self.nlp.process(user_input.lower().strip())
            
            # Analyze intent and sentiment
            intent = self._analyze_intent(processed_input)
            sentiment = self._analyze_sentiment(processed_input)
            
            # Update context
            self._update_context(intent, sentiment, processed_input)
            
            # Generate appropriate response
            response = self._generate_response(intent, processed_input)
            
            # Update conversation history
            self.context.conversation_history.append({
                "user": processed_input,
                "jarvis": response,
                "intent": intent,
                "timestamp": datetime.now()
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return "Sorry, er is iets misgegaan. Kun je dat op een andere manier zeggen?"

    def _analyze_intent(self, text: str) -> str:
        # Basic intent recognition - should be expanded with ML model
        if any(word in text for word in ['hallo', 'hey', 'hi', 'hoi']):
            return "greeting"
        elif any(word in text for word in ['hoe gaat', 'hoe is', 'alles goed']):
            return "how_are_you"
        elif any(word in text for word in ['wie ben', 'wat ben', 'vertel over jezelf']):
            return "about_me"
        return "unknown"

    def _analyze_sentiment(self, text: str) -> str:
        # Basic sentiment - should be expanded with ML model
        positive_words = ['goed', 'fijn', 'geweldig', 'leuk']
        negative_words = ['slecht', 'jammer', 'vervelend']
        
        if any(word in text for word in positive_words):
            return "positive"
        elif any(word in text for word in negative_words):
            return "negative"
        return "neutral"

    def _update_context(self, intent: str, sentiment: str, text: str):
        self.context.last_intent = intent
        self.context.user_mood = sentiment
        # Update topic based on content analysis
        if any(word in text for word in ['weer', 'temperatuur', 'regen']):
            self.context.topic = "weather"
        elif any(word in text for word in ['nieuws', 'gebeurd', 'vandaag']):
            self.context.topic = "news"

    def _generate_response(self, intent: str, text: str) -> str:
        # Get base response from templates
        base_responses = self.responses.get(intent, self.responses["unknown"])
        response = random.choice(base_responses)
        
        # Enhance response based on context
        if self.context.topic != "general":
            # Add topic-specific information
            if self.context.topic == "weather" and self.knowledge_base:
                weather_info = self.knowledge_base.query("weather")
                if weather_info:
                    response += f" Ik zie dat het weer {weather_info} is."
                    
        return response
