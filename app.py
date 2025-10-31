from flask import Flask, render_template, request, redirect, url_for, session
import os
import extract_audio as ea
import speech_to_text as stt
import facial_evaluation as fe
import user_suggestions as us
import pandas as pd 

app = Flask(__name__)
app.secret_key = "your_secret_key" # needed for session management

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

QUESTIONS = [
    "What excites you most about this opportunity?",
    "Describe a challenge you overcame.",
    "What are your strongest skills?",
    "Where do you see yourself in 3 years?",
    "Why should we hire you?"
]

@app.route('/')
def index():
    session["step"] = 0
    return redirect(url_for("question", qid=1))

@app.route('/question/<int:qid>', methods=['GET', 'POST'])
def question(qid):
    if qid < 1 or qid > len(QUESTIONS):
        return redirect(url_for("thankyou"))

    if request.method == "POST":
        video = request.files.get("video")
        if video:
            filename = f"answer_{qid}.webm"
            video.save(os.path.join(UPLOAD_FOLDER, filename))
        next_qid = qid + 1
        if next_qid > len(QUESTIONS):
            ea.extract_audios()
            stt.extract_text() 
            return redirect(url_for("results"))
        return redirect(url_for("question", qid=next_qid))

    return render_template("index.html", 
        question=QUESTIONS[qid - 1],
        qid=qid,
        total=len(QUESTIONS)
    )

@app.route('/thankyou')
def thankyou():
    return "<h2>Thank you! All answers have been saved.</h2>"

@app.route('/results')
def results():
    results_data = fe.get_score()

    print(results_data)
    
    # Convert to DataFrame
    results = pd.DataFrame(results_data).T  # Transpose so questions are rows

    # Optionally reset index for a cleaner look
    results = results.reset_index().rename(columns={'index': 'Question'})

    results = zip(results['Question'].tolist(), results['Happy/Neutral Expression %'].tolist(), results['Negative Emotion %'].tolist(), results['Eye Steadiness %'].tolist(), 
                  results['Pitch Score'].tolist(), results['Filler Word Score'].tolist(), results['Speech Score'].tolist(), results['Context Similarity Score'].tolist(), 
                  results['Final Confidence Score'].tolist())
    
    notes = us.get_suggestions(results_data)

    return render_template('results.html', results = results, notes = notes)

if __name__ == '__main__':
    app.run(debug=True) 
