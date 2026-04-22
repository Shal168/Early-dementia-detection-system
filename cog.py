# cognitive_test.py (Hospital-Level Cognitive Assessment System)

import time

# ==========================
# CONFIGURATION
# ==========================
WEIGHTS = {
    "memory": 6,
    "visual": 6,
    "clock": 4,
    "sequence": 8,
    "attention": 6
}

MAX_SCORES = {
    "memory": 7,
    "visual": 6,
    "clock": 3,
    "sequence": 10,
    "attention": 10
}

# ==========================
# VALIDATION FUNCTION
# ==========================
def validate_input(data):
    errors = []

    for key in MAX_SCORES:
        if key not in data:
            errors.append(f"{key} is missing")
        elif not isinstance(data[key], (int, float)):
            errors.append(f"{key} must be a number")
        elif data[key] < 0 or data[key] > MAX_SCORES[key]:
            errors.append(f"{key} must be between 0 and {MAX_SCORES[key]}")

    return errors


# ==========================
# NORMALIZATION
# ==========================
def normalize(score, max_score, weight):
    return (score / max_score) * weight


# ==========================
# MAIN ANALYSIS FUNCTION
# ==========================
def analyze_cognitive(data):

    start_time = time.time()

    # ==========================
    # VALIDATE INPUT
    # ==========================
    errors = validate_input(data)
    if errors:
        return {
            "status": "error",
            "errors": errors
        }

    # Extract values
    memory = data["memory"]
    visual = data["visual"]
    clock = data["clock"]
    sequence = data["sequence"]
    attention = data["attention"]

    # ==========================
    # SCORING
    # ==========================
    memory_score = normalize(memory, MAX_SCORES["memory"], WEIGHTS["memory"])
    visual_score = normalize(visual, MAX_SCORES["visual"], WEIGHTS["visual"])
    clock_score = normalize(clock, MAX_SCORES["clock"], WEIGHTS["clock"])
    sequence_score = normalize(sequence, MAX_SCORES["sequence"], WEIGHTS["sequence"])
    attention_score = normalize(attention, MAX_SCORES["attention"], WEIGHTS["attention"])

    total_score = round(
        memory_score +
        visual_score +
        clock_score +
        sequence_score +
        attention_score, 2
    )

    # ==========================
    # CLASSIFICATION
    # ==========================
    if total_score >= 26:
        risk = "Normal Cognitive Function"
        severity = "Low Risk"
    elif total_score >= 22:
        risk = "Mild Cognitive Impairment"
        severity = "Mild"
    elif total_score >= 15:
        risk = "Moderate Cognitive Impairment"
        severity = "Moderate"
    else:
        risk = "Severe Cognitive Impairment"
        severity = "High Risk"

    # ==========================
    # CONFIDENCE
    # ==========================
    confidence = round(total_score / 30, 2)

    # ==========================
    # CLINICAL INTERPRETATION
    # ==========================
    if severity == "Low Risk":
        interpretation = "Cognitive performance is within expected limits."
        recommendation = "Routine monitoring recommended."
    elif severity == "Mild":
        interpretation = "Mild cognitive decline detected."
        recommendation = "Periodic reassessment advised."
    elif severity == "Moderate":
        interpretation = "Noticeable cognitive impairment."
        recommendation = "Clinical consultation recommended."
    else:
        interpretation = "Severe cognitive impairment detected."
        recommendation = "Immediate neurological evaluation required."

    end_time = time.time()

    # ==========================
    # FINAL STRUCTURED OUTPUT
    # ==========================
    return {
        "status": "success",

        "patient_input": {
            "memory": f"{memory}/{MAX_SCORES['memory']}",
            "visual": f"{visual}/{MAX_SCORES['visual']}",
            "clock": f"{clock}/{MAX_SCORES['clock']}",
            "sequence": f"{sequence}/{MAX_SCORES['sequence']}",
            "attention": f"{attention}/{MAX_SCORES['attention']}"
        },

        "normalized_scores": {
            "memory_score": round(memory_score, 2),
            "visual_score": round(visual_score, 2),
            "clock_score": round(clock_score, 2),
            "sequence_score": round(sequence_score, 2),
            "attention_score": round(attention_score, 2)
        },

        "summary": {
            "total_score": total_score,
            "confidence": confidence,
            "risk_level": risk,
            "severity": severity
        },

        "clinical_report": {
            "interpretation": interpretation,
            "recommendation": recommendation
        },

        "processing_time": round(end_time - start_time, 3)
    }


# ==========================
# TEST MODE (CLI)
# ==========================
if __name__ == "__main__":

    print("\n===== COGNITIVE TEST (HOSPITAL MODE) =====")

    data = {
        "memory": int(input("Memory (0-7): ")),
        "visual": int(input("Visual (0-6): ")),
        "clock": int(input("Clock (0-3): ")),
        "sequence": int(input("Sequence (0-10): ")),
        "attention": int(input("Attention (0-10): "))
    }

    result = analyze_cognitive(data)

    if result["status"] == "error":
        print("\n❌ INPUT ERRORS:")
        for e in result["errors"]:
            print("-", e)
        exit()

    print("\n===== COGNITIVE REPORT =====")

    print("\n--- Patient Input ---")
    for k, v in result["patient_input"].items():
        print(f"{k}: {v}")

    print("\n--- Scores ---")
    for k, v in result["normalized_scores"].items():
        print(f"{k}: {v}")

    print("\n--- Summary ---")
    print(f"Total Score : {result['summary']['total_score']}/30")
    print(f"Confidence  : {result['summary']['confidence']}")
    print(f"Risk        : {result['summary']['risk_level']}")
    print(f"Severity    : {result['summary']['severity']}")

    print("\n--- Clinical Interpretation ---")
    print(result["clinical_report"]["interpretation"])
    print("Recommendation:", result["clinical_report"]["recommendation"])

    print("\nProcessing Time:", result["processing_time"], "seconds")
