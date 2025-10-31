import json
import time # Added for safety margin between API calls
from google import genai
from google.genai import types

# --- 1. Function for Individual Question Analysis (Initial Call) ---

def generate_suggestions_for_one_question(justification: dict, question_text: str, api_key: str) -> str:
    """
    Calls the Gemini API to get personalized, detailed suggestions for a single question.
    """
    try:
        client = genai.Client(api_key=api_key)
        justification_json = json.dumps(justification, indent=4)

        system_instruction = (
            "You are an expert interview coach providing detailed, question-specific feedback. "
            "Analyze the scores and the original question. "
            "For the 'Context Similarity Score', use the question to explain the score. "
            "Provide detailed feedback structured with clear headings: 'Summary', 'Improvement Points', and 'Strengths'."
        )

        prompt = (
            f"Provide detailed feedback for this single question based on the data.\n\n"
            f"ORIGINAL QUESTION:\n\"{question_text}\"\n\n"
            f"PERFORMANCE DATA:\n{justification_json}"
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            ),
        )
        return response.text

    except Exception as e:
        return f"ERROR: API call failed for question: {question_text[:30]}...: {e}"

# --- 2. Function for Combined Overall Analysis (Second Call) ---

def generate_overall_summary(all_feedback_data: dict, api_key: str) -> str:
    """
    Takes all individual question feedback and generates a consolidated, high-level summary.
    """
    try:
        client = genai.Client(api_key=api_key)
        
        # Prepare the consolidated text input for the summarization model
        consolidated_analysis = ""
        
        for i, (question, data) in enumerate(all_feedback_data.items(), 1):
            
            # Use data.get() for safer access, though the structure is now fixed
            metrics_json = json.dumps(data.get('justification', {}), indent=2)
            detailed_feedback = data.get('detailed_feedback', 'No detailed feedback available.')
            
            consolidated_analysis += f"--- QUESTION {i}: {question} ---\n"
            consolidated_analysis += f"METRICS:\n{metrics_json}\n"
            consolidated_analysis += f"DETAILED FEEDBACK:\n{detailed_feedback}\n\n"

        system_instruction = (
            "You are a Senior Interview Performance Analyst. Your task is to review all the provided detailed "
            "feedback and performance metrics for a full interview session. "
            "Provide a high-level, strategic summary of the candidate's performance. "
            "The output MUST be a maximum of 3 paragraphs, focusing on overarching trends (e.g., 'Eyes were steady across all questions, but filler words spiked during behavioral questions'). "
            "Use the headings: 'Overall Performance Synthesis' and 'Key Strategic Advice'."
        )

        prompt = (
            f"Review the following detailed analysis for all questions in an interview session. "
            f"Provide a concise, high-level summary (max 3 paragraphs) that identifies overarching strengths and consistent areas for improvement.\n\n"
            f"FULL INTERVIEW ANALYSIS:\n{consolidated_analysis}"
        )

        response = client.models.generate_content(
            model='gemini-2.5-pro', # Using Pro for superior summarization
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3
            ),
        )
        return response.text

    except Exception as e:
        return f"An error occurred during overall summarization: {e}"


# --- Sample Usage ---

def get_suggestions(questions_and_scores):
    # **IMPORTANT: REPLACE THIS WITH YOUR ACTUAL API KEY**
    # NOTE: I am keeping the provided example key but stressing it MUST be replaced/valid.
    GEMINI_API_KEY = "Your api key here" 

    if GEMINI_API_KEY == "your api key here":
         # In a real environment, this would check if the key is valid.
         # For demonstration, we assume the user will replace it or handle auth elsewhere.
         pass 

    print("--- 1/2: Generating Detailed Feedback for Each Question ---")
    
    # Store the results with the detailed feedback appended
    intermediate_data_for_summary = {}
    
    for question_text, justification_data in questions_and_scores.items():
        print(f"  -> Processing: {question_text[:40]}...")
        
        # Get detailed feedback for the current question
        detailed_feedback = generate_suggestions_for_one_question(
            justification=justification_data,
            question_text=question_text,
            api_key=GEMINI_API_KEY
        )
        
        # Store all data needed for the final summary call
        intermediate_data_for_summary[question_text] = {
            "justification": justification_data,
            "detailed_feedback": detailed_feedback
        }
        
        # Add a small delay to respect API rate limits, especially for 5 consecutive calls
        time.sleep(1) 
    
    print("\n--- 2/2: Generating Combined 2-3 Paragraph Overall Summary ---")

    # FIX APPLIED: Correctly pass the processed data (intermediate_data_for_summary) 
    # which contains the 'justification' and 'detailed_feedback' keys.
    overall_summary = generate_overall_summary(intermediate_data_for_summary, GEMINI_API_KEY)


    # 3. Print the final, consolidated summary
    print("\n" + "="*80)
    print("FINAL CONSOLIDATED INTERVIEW PERFORMANCE SYNTHESIS")
    print("="*80)
    print(overall_summary)
    print("="*80) 

    return overall_summary 
