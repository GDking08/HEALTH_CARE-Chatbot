import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import csv
import random
import datetime
from collections import defaultdict
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# Load data
symptoms_conditions_df = pd.read_csv('data/symptoms_conditions1.csv')
conditions_treatments_df = pd.read_csv('data/conditions_treatments.csv')

symptoms_conditions_dict = {}
for symptom, group in symptoms_conditions_df.groupby('Symptom'):
    symptoms_conditions_dict[symptom] = group['Condition'].tolist()

conditions_treatments_dict = {}
for condition, group in conditions_treatments_df.groupby('Condition'):
    conditions_treatments_dict[condition] = group['Treatment'].tolist()
# User state
user_state = defaultdict(lambda: {'name': None, 'conversation_stage': 'get_name', 'condition': None, 'duration': None, 'symptoms': []})

# Load doctors and appointments
def load_csv(filename):
    with open(filename, 'r') as f:
        return list(csv.DictReader(f))

doctors = load_csv('data/doctors.csv')
appointments = load_csv('data/appointments.csv')

def save_appointment(appointment):
    fieldnames = ['ID', 'Name', 'Time', 'Date', 'Illness', 'Doctor', 'Title', 'Description']
    with open('data/appointments.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(appointment)

def get_available_slots(date):
    booked_slots = [apt['Time'] for apt in appointments if apt.get('Date') == date]
    all_slots = [f"{h:02d}:00" for h in range(8, 21)]
    return [slot for slot in all_slots if slot not in booked_slots]

def find_closest_slot(preferred_time, available_slots):
    preferred_minutes = int(preferred_time[:2]) * 60 + int(preferred_time[3:])
    closest_slot = min(available_slots, key=lambda x: abs(preferred_minutes - (int(x[:2]) * 60 + int(x[3:]))))
    return closest_slot

def get_greeting(name):
    greetings = [
        f" Hello {name}! How can I assist you today?",
        f"Hi there, {name}!  What brings you here?",
        f"Greetings, {name}!  How may I help you?",
        f"Welcome, {name}!  What would you like to know?",
        f"Hey {name}!  How can I be of service today?"
    ]
    return random.choice(greetings)



def is_greeting(message):
    greetings = ['hi', 'hello', 'hey', 'greetings', 'hola']
    return any(greeting in message.lower() for greeting in greetings)

def preprocess_text(text):
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token not in stop_words]

def compute_lps(pattern):
    """Compute the Longest Prefix Suffix (LPS) array for KMP algorithm."""
    lps = [0] * len(pattern)
    length = 0  
    i = 1

    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

def kmp_search(text, pattern):
    """Perform KMP search to check if 'pattern' exists in 'text'."""
    N = len(text)
    M = len(pattern)
    lps = compute_lps(pattern)

    i = j = 0  # Index for text and pattern
    while i < N:
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == M:
            return True  # Pattern found in text
            j = lps[j - 1]
        elif i < N and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return False  # Pattern not found

def match_symptoms(user_input):
    """Match symptoms using the KMP algorithm."""
    matched_symptoms = []
    
    for symptom in symptoms_conditions_dict.keys():
        if kmp_search(user_input.lower(), symptom.lower()):
            matched_symptoms.append(symptom)

    return matched_symptoms

hospitals_df = pd.read_csv('data/tamil_nadu_hospital.csv')
# User session states
user_state = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.json
        user_input = data.get('message', '').strip().lower()
        session_id = data.get('session_id', 'default')

        # Initialize or retrieve the user session state
        state = user_state.setdefault(session_id, {'name': None, 'conversation_stage': 'get_name', 'condition': None, 'duration': None, 'symptoms': [], 'district': None})

        # Stage 1: Get user name
        if state['conversation_stage'] == 'get_name':
            if user_input == 'reset':
                # Reset the session if user wants to start over
                user_state[session_id] = {'name': None, 'conversation_stage': 'get_name', 'condition': None, 'duration': None, 'symptoms': [], 'district': None}
                return jsonify({'response': "Let's start over. What should I call you?"})

            # Store the user's name and transition to the next stage
            state['name'] = user_input
            state['conversation_stage'] = 'ask_symptoms'
            return jsonify({'response': f"Greetings, {user_input.capitalize()}! How may I help you? Please tell me your symptoms."})

        user_name = state['name']

        # Handle symptom input (Stage 2)
        if state['conversation_stage'] == 'ask_symptoms':
            # Match the user's input to known symptoms
            matched_symptoms = match_symptoms(user_input)
            state['symptoms'].extend(matched_symptoms)

            if matched_symptoms:
                # If symptoms are matched, find possible conditions
                all_conditions = [condition for symptom in state['symptoms'] for condition in symptoms_conditions_dict.get(symptom, [])]
                if all_conditions:
                    # Determine the most likely condition based on matched symptoms
                    condition = max(set(all_conditions), key=all_conditions.count)
                    state['condition'] = condition
                    # Fetch treatments based on the identified condition
                    treatments = conditions_treatments_dict.get(condition, ["Consult a healthcare professional"])
                    response = f"{user_name}, based on your symptoms, you may have {condition}. Suggested treatments: {', '.join(treatments)}.\n\nHow many days have you been experiencing these symptoms?"
                    state['conversation_stage'] = 'ask_duration'  # Transition to asking for duration
                else:
                    response = f"I've noted your symptoms, {user_name}. Could you provide more details about how you're feeling?"
            else:
                response = f"I'm not sure about your condition based on the information provided, {user_name}. Could you tell me more about your symptoms?"

            return jsonify({'response': response})

        # Handle duration input (Stage 3)
        if state['conversation_stage'] == 'ask_duration':
            try:
                duration = int(user_input)
                state['duration'] = duration
                state['conversation_stage'] = 'ask_district'

                # If the symptoms have been present for 5 or more days, ask for the district to suggest nearby hospitals
                if duration >= 5:
                    response = f"{user_name}, since you've been experiencing these symptoms for {duration} days, I recommend consulting a doctor. Please provide your district name to find nearby hospitals."
                else:
                    response = f"I understand, {user_name}. Since it's been less than 5 days, monitor your symptoms. If they worsen, consult a doctor. Need any more help?"
            except ValueError:
                response = f"I'm sorry, {user_name}, but I didn't understand that. Could you please enter the number of days you've been experiencing these symptoms?"

            return jsonify({'response': response})

        # Handle district input (Stage 4)
        if state['conversation_stage'] == 'ask_district':
            state['district'] = user_input.capitalize()
            hospitals = hospitals_df[hospitals_df['District'].str.lower() == user_input.lower()]['Hospital'].tolist()

            if hospitals:
                hospitals_list = "\n".join(hospitals)
                response = f"The government hospitals in {state['district']} are:\n{hospitals_list}\n\nWould you like to book an appointment at one of these hospitals? (Yes/No)"
                state['conversation_stage'] = 'ask_appointment'
            else:
                response = f"Sorry, I couldn't find any hospitals in {state['district']}. Please check the spelling or try another district."

            return jsonify({'response': response})

        # Handle appointment question (Stage 5)
        if user_input in ['yes', 'no']:
            if state['conversation_stage'] == 'ask_appointment':
                if user_input == 'yes':
                    state['conversation_stage'] = 'ask_appointment_time'
                    return jsonify({'response': "Please enter your preferred appointment time (HH:MM format)."})
                elif user_input == 'no':
                    user_state.pop(session_id, None)  # Terminate the session
                    return jsonify({'response': "Alright! Take care! If you need help later, come again."})

        # Handle appointment time input (Stage 6)
        if state['conversation_stage'] == 'ask_appointment_time':
            return jsonify({'response': "Please enter your preferred appointment time (HH:MM format)."})

        # If the conversation stage is not recognized or an error occurs, handle appropriately
        return jsonify({'response': "I'm sorry, I didn't understand that. Can you please clarify?"})

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({'response': "I apologize, but an error occurred. Please try again or contact support if the problem persists."})


@app.route('/book_appointment', methods=['POST'])
def handle_appointment():
    try:
        data = request.json  # Get request data
        app.logger.info(f"Received data: {data}")  # Log incoming data

        session_id = data.get('session_id', 'default')
        preferred_time = data.get('preferred_time')

        if session_id not in user_state:
            app.logger.error(f"Session ID {session_id} not found in user_state.")
            return jsonify({'response': "Session expired or not found. Please start again."})

        state = user_state[session_id]
        name = state.get('name')
        illness = state.get('condition')

        if not all([name, illness, preferred_time]):
            app.logger.error(f"Missing details: Name={name}, Illness={illness}, Preferred Time={preferred_time}")
            return jsonify({'response': "I'm missing some details. Could you provide all necessary details for booking an appointment?"})

        # Use current date for the appointment
        current_date = datetime.datetime.now()
        appointment_date = current_date.strftime("%Y-%m-%d")

        # Check if preferred_time is valid
        try:
            preferred_datetime = datetime.datetime.strptime(f"{appointment_date} {preferred_time}", "%Y-%m-%d %H:%M")
        except ValueError:
            app.logger.error(f"Invalid time format received: {preferred_time}")
            return jsonify({'response': "Invalid time format. Please enter the time in HH:MM format."})

        if preferred_datetime <= current_date:
            return jsonify({'response': f"I'm sorry {name}, but the requested time has already passed. Please choose a future time."})

        # Get available slots
        available_slots = get_available_slots(appointment_date)
        app.logger.info(f"Available slots: {available_slots}")

        if preferred_time not in available_slots:
            closest_slot = find_closest_slot(preferred_time, available_slots)
            return jsonify({'response': f"The slot at {preferred_time} is unavailable. The closest available slot is {closest_slot}. Would you like to book this slot? (Yes/No)"})

        # Assign a random doctor
        assigned_doctor_info = random.choice(doctors)
        assigned_doctor = assigned_doctor_info['Name']
        appointment_id = f"APPT-{random.randint(1000, 9999)}"
        district = user_state[session_id].get('district', 'Unknown')
        hospital = user_state[session_id].get('hospital', 'Unknown')
        hospitals = hospitals_df[hospitals_df['District'].str.lower() == district.lower()]['Hospital'].tolist()
        hospital = hospitals[0] if hospitals else "No hospital found"

        # Create appointment
        appointment = {
            'ID': appointment_id,
            'Name': name,
            'Time': preferred_time,
            'Date': appointment_date,
            'Illness': illness,
            'Doctor': assigned_doctor,
            'Title': f"Appointment for {illness}",
            'Description': f"Consultation for {illness} symptoms"
        
        }

        save_appointment(appointment)

        response = f"Great news, {name}! Your appointment is booked.\n"
        response += f"Appointment ID: {appointment_id}\n"
        response += f"Doctor: {assigned_doctor}\n"
        response += f"Date: {appointment_date}\n"
        response += f"Time: {preferred_time}\n"
        response += f"District: {district}\n"
        response += f"Hospital: {hospital}\n"
        response += f"Please visit the hospital on time.\n"

        return jsonify({'response': response})

    except Exception as e:
        app.logger.error(f"An error occurred while booking the appointment: {str(e)}")
        return jsonify({'response': "An error occurred while booking. Please try again or contact support."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)