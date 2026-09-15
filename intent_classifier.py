import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import re
from typing import Dict, List, Tuple, Optional

class IntentClassifier:
    def __init__(self, training_file='training_data.json'):
        """Initialize the intent classifier with spaCy and Sentence Transformers"""
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_md')
        except:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_md'])
            self.nlp = spacy.load('en_core_web_md')
        
        # Load sentence transformer for better semantic understanding
        self.transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load training data
        with open(training_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Prepare intent patterns and embeddings
        self.intent_patterns = {}
        self.intent_embeddings = {}
        self.intent_responses = {}
        
        for intent in self.data['intents']:
            intent_name = intent['intent']
            patterns = intent['patterns']
            responses = intent['responses']
            
            self.intent_patterns[intent_name] = patterns
            self.intent_responses[intent_name] = responses
            
            # Create embeddings for patterns
            pattern_embeddings = self.transformer.encode(patterns)
            self.intent_embeddings[intent_name] = pattern_embeddings
        
        # Store entity patterns
        self.entities = self.data.get('entities', {})
        self.entity_patterns = self._compile_entity_patterns()
    
    def _compile_entity_patterns(self):
        """Compile regex patterns for entity extraction"""
        patterns = {}
        for entity_type, values in self.entities.items():
            patterns[entity_type] = re.compile('|'.join(values), re.IGNORECASE)
        return patterns
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using spaCy NER and custom patterns"""
        entities = {
            'courses': [],
            'dates': [],
            'amounts': [],
            'names': []
        }
        
        # Use spaCy NER
        doc = self.nlp(text)
        
        # Extract entities using spaCy
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            elif ent.label_ == 'MONEY':
                entities['amounts'].append(ent.text)
            elif ent.label_ == 'PERSON':
                entities['names'].append(ent.text)
            elif ent.label_ in ['ORG', 'PRODUCT']:
                entities['courses'].append(ent.text)
        
        # Use custom patterns for course names
        if 'course_names' in self.entities:
            course_pattern = re.compile('|'.join(self.entities['course_names']), re.IGNORECASE)
            found_courses = course_pattern.findall(text)
            entities['courses'].extend(found_courses)
        
        # Use custom patterns for months
        if 'months' in self.entities:
            month_pattern = re.compile('|'.join(self.entities['months']), re.IGNORECASE)
            found_months = month_pattern.findall(text)
            entities['dates'].extend(found_months)
        
        # Use custom patterns for fee amounts
        if 'fee_amounts' in self.entities:
            fee_pattern = re.compile('|'.join(self.entities['fee_amounts']), re.IGNORECASE)
            found_fees = fee_pattern.findall(text)
            entities['amounts'].extend(found_fees)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def classify_intent(self, text: str, threshold: float = 0.5) -> Tuple[str, float, Dict]:
        """Classify the intent of the user query"""
        # Generate embedding for the query
        query_embedding = self.transformer.encode([text])[0]
        
        best_intent = None
        best_score = 0
        all_scores = {}
        
        # Compare with each intent's patterns
        for intent_name, pattern_embeddings in self.intent_embeddings.items():
            # Calculate similarity with all patterns for this intent
            similarities = cosine_similarity([query_embedding], pattern_embeddings)[0]
            max_similarity = np.max(similarities)
            all_scores[intent_name] = max_similarity
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_intent = intent_name
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # If score is below threshold, use fallback
        if best_score < threshold:
            return 'general', best_score, entities
        
        return best_intent, best_score, entities
    
    def get_response(self, intent: str, entities: Dict) -> str:
        """Get response for the classified intent"""
        if intent in self.intent_responses:
            responses = self.intent_responses[intent]
            response = np.random.choice(responses)
            
            # Personalize response with extracted entities
            if entities['courses']:
                response += f"\n\nYou mentioned {', '.join(entities['courses'])}. Would you like more details about these?"
            
            if entities['dates']:
                response += f"\n\nRegarding {', '.join(entities['dates'])}, would you like specific dates?"
            
            return response
        
        return "🤔 I'm not sure about that. Could you please rephrase your question? I can help with courses, fees, exams, placements, and more!"
    
    def get_all_intents(self) -> List[str]:
        """Get list of all intents"""
        return list(self.intent_patterns.keys())

# Singleton instance
intent_classifier = None

def get_intent_classifier():
    global intent_classifier
    if intent_classifier is None:
        intent_classifier = IntentClassifier()
    return intent_classifier