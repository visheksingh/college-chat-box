from flask import Flask, render_template, request, jsonify, session
from intent_classifier import get_intent_classifier
import json
import uuid
from datetime import datetime
import re

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Initialize intent classifier
classifier = get_intent_classifier()

# Load knowledge base for fallback
try:
    with open('knowledge_base.json', 'r', encoding='utf-8') as f:
        college_data = json.load(f)
except:
    college_data = {}

class CollegeChatbot:
    def __init__(self):
        self.conversation_history = {}
        self.classifier = classifier
    
    def get_response(self, user_query: str, session_id: str) -> dict:
        """Process user query and return response with metadata"""
        
        # Store conversation history
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        self.conversation_history[session_id].append({
            'role': 'user',
            'content': user_query,
            'timestamp': datetime.now().isoformat()
        })
        
        # Classify intent
        intent, confidence, entities = self.classifier.classify_intent(user_query)
        
        # Get response based on intent
        response_text = self.classifier.get_response(intent, entities)
        
        # Additional logic for specific entities
        response_text = self._enhance_response_with_entities(response_text, entities, intent)
        
        # Store bot response
        bot_response = {
            'text': response_text,
            'intent': intent,
            'confidence': float(confidence),
            'entities': entities,
            'timestamp': datetime.now().isoformat()
        }
        
        self.conversation_history[session_id].append({
            'role': 'bot',
            'content': response_text,
            'intent': intent
        })
        
        return bot_response
    
    def _enhance_response_with_entities(self, response: str, entities: dict, intent: str) -> str:
        """Enhance response with extracted entities"""
        
        # If course names found, add specific details
        if entities['courses'] and intent == 'course_inquiry':
            course_details = self._get_course_details(entities['courses'])
            if course_details:
                response += f"\n\n📖 {course_details}"
        
        # If dates found and exam intent
        if entities['dates'] and intent == 'exam_inquiry':
            response += f"\n\n📅 Would you like more specific dates for {', '.join(entities['dates'])}?"
        
        # If amounts found and fee intent
        if entities['amounts'] and intent == 'fee_inquiry':
            response += f"\n\n💳 These amounts might vary based on scholarships. Contact financial aid for exact figures."
        
        return response
    
    def _get_course_details(self, course_names: list) -> str:
        """Get additional details for specific courses"""
        details = []
        for course in course_names:
            if 'B.Tech' in course or 'engineering' in course.lower():
                details.append("B.Tech programs are 4 years with specialization options")
            elif 'MBA' in course:
                details.append("MBA is a 2-year program with dual specialization option")
            elif 'BCA' in course:
                details.append("BCA is a 3-year program with practical training")
            elif 'BBA' in course:
                details.append("BBA is a 3-year program with industry exposure")
        
        if details:
            return " • " + "\n • ".join(details)
        return ""

# Initialize chatbot
chatbot = CollegeChatbot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not user_message:
            return jsonify({'error': 'Please enter a message'}), 400
        
        # Get response from chatbot
        bot_response = chatbot.get_response(user_message, session_id)
        
        return jsonify({
            'response': bot_response['text'],
            'intent': bot_response['intent'],
            'confidence': bot_response['confidence'],
            'entities': bot_response['entities'],
            'session_id': session_id
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'response': "I'm having trouble processing your request. Please try again.",
            'error': str(e)
        }), 500

@app.route('/conversation/<session_id>', methods=['GET'])
def get_conversation(session_id):
    """Get conversation history for a session"""
    history = chatbot.conversation_history.get(session_id, [])
    return jsonify(history)

@app.route('/intents', methods=['GET'])
def get_intents():
    """Get all available intents"""
    return jsonify(classifier.get_all_intents())

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎓 COLLEGE CHATBOT WITH NLP IS RUNNING!")
    print("🤖 Features:")
    print("   • Intent Classification using Transformers")
    print("   • Entity Recognition with spaCy")
    print("   • Semantic Understanding (not just keywords)")
    print("📍 Open: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)