import pandas as pd
import json

# Define the symptom questions mapping to match the desired output format
symptom_mapping = {
    'Do you have little interest or pleasure in doing things?': 'Little interest or pleasure',
    'Do you feel down, depressed, or hopeless?': 'Feeling down',
    'Do you have trouble falling or staying asleep, or do you sleep too much?': 'Sleep',
    'Do you feel tired or have little energy?': 'Energy',
    'Do you have poor appetite or tend to overeat?': 'Appetite',
    'Do you feel bad about yourself or that you are a failure or have let yourself or your family down?': 'Self-worth',
    'Do you have trouble concentrating on things, such as reading, work, or watching television?': 'Concentration',
    'Have you been moving or speaking so slowly that other people have noticed, or the opposite—being fidgety or restless?': 'Motor activity',
    'Have you had thoughts of self-harm or felt that you would be better off dead?': 'Suicidal thoughts'
}

def get_supportive_message(score, severity):
    if score >= 20:
        return (
            f"Your PHQ-9 score is {score}, which suggests {severity.lower()} symptoms of depression. "
            "I'm really sorry you're feeling this way. Please consider reaching out to a mental health professional as soon as possible. "
            "You're not alone, and there are people who care and want to help you. If you ever feel unsafe, please contact a helpline or someone you trust immediately."
        )
    elif score >= 10:
        return (
            f"Your PHQ-9 score is {score}, indicating {severity.lower()} symptoms of depression. "
            "It sounds like you're going through a tough time. Remember, you're not alone—consider talking to a mental health professional or someone you trust. "
            "Small steps like keeping a routine, staying connected with friends, and practicing self-care can help."
        )
    else:
        return (
            f"Your PHQ-9 score is {score}, which suggests {severity.lower()} symptoms of depression. "
            "It's great that you're managing well. Remember to take care of yourself and reach out if you ever need support. "
            "If you notice any changes in your mood or daily life, don't hesitate to talk to someone you trust."
        )

# Read the CSV file
csv_file = '/workspace/Bolt/project/backend/phq9.csv'
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: The file '{csv_file}' was not found. Please ensure the file exists in the correct directory.")
    exit(1)

# Print column names for debugging
print("CSV Column Names:")
print(df.columns.tolist())

# Verify that all expected columns exist
missing_columns = [col for col in symptom_mapping.keys() if col not in df.columns]
if missing_columns:
    print(f"Error: The following columns are missing or named differently in the CSV: {missing_columns}")
    print("Please update the symptom_mapping dictionary to match the exact column names in the CSV.")
    exit(1)

# Initialize the output JSONL file
output_file = 'dataset.jsonl'

with open(output_file, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        # Construct the user message
        user_content = "Evaluate the following depression symptoms:\n\n"
        for column, label in symptom_mapping.items():
            # Safely access the column and handle potential missing values
            response = str(row[column]).strip() if pd.notnull(row[column]) else "No response"
            user_content += f"- {label}: {response}\n"
        
        # Calculate PHQ-9 score and severity, handling missing values
        phq_score = row['PHQ-9 Score'] if pd.notnull(row['PHQ-9 Score']) else 0
        severity = row['Severity Level'] if pd.notnull(row['Severity Level']) else "Unknown"
        
        # Construct the assistant message using the supportive message function
        assistant_content = get_supportive_message(phq_score, severity)
        
        # Create the JSON object
        json_obj = {
            "messages": [
                {
                    "role": "user",
                    "content": user_content.strip()
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ]
        }
        
        # Write the JSON object as a single line
        json.dump(json_obj, f, ensure_ascii=False)
        f.write('\n')

print(f"JSONL file '{output_file}' has been generated successfully.")