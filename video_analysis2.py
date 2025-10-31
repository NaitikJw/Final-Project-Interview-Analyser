import cv2
from fer import FER

def calculate_confidence_from_emotions(emotion_scores):
    """
    Given a dict of emotion confidence scores (e.g., {'happy': 0.7, 'sad': 0.1, etc.}),
    calculates a single confidence score between 0 and 1.
    """
    weights = {
        "happy": 1.0,
        "neutral": 0.8,
        "surprise": 0.6,
        "sad": -0.7,
        "angry": -0.8,
        "fear": -0.7,
        "disgust": -0.8
    }

    score = 0.0
    total_weight = 0.0

    for emotion, weight in weights.items():
        if emotion in emotion_scores:
            score += weight * emotion_scores[emotion]
            total_weight += abs(weight)

    # Normalize to range [0,1]
    normalized_score = (score + total_weight) / (2 * total_weight) if total_weight > 0 else 0
    normalized_score = max(0.0, min(1.0, normalized_score))

    return normalized_score

class FacialExpressionConfidenceAnalyzer:
    def __init__(self):
        # Load model once
        self.detector = FER()

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        confidence_scores = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Detect emotions on the current frame
            results = self.detector.detect_emotions(frame)
            if results:
                # Get emotion scores dict from first detected face
                emotions = results[0]["emotions"]
                confidence = calculate_confidence_from_emotions(emotions)
                confidence_scores.append(confidence)

        cap.release()
        print(f"Processed {frame_count} frames, confidence scores gathered for {len(confidence_scores)} frames.")
        return confidence_scores

    def get_average_confidence(self, confidence_scores):
        if not confidence_scores:
            return 0.0
        return sum(confidence_scores) / len(confidence_scores)


if __name__ == "__main__":
    video_file = "recorded_answer.avi"  # Replace with your actual video file path
    analyzer = FacialExpressionConfidenceAnalyzer()

    frame_confidences = analyzer.analyze_video(video_file)
    avg_confidence = analyzer.get_average_confidence(frame_confidences)

    print(f"Average confidence score for the video: {avg_confidence:.3f}")
