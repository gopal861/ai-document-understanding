import time
import json
import requests
import statistics
from typing import List, Dict
from datetime import datetime
import os


# ============================================================
# CONFIGURATION — MATCHES YOUR REAL DEPLOYMENT
# ============================================================

ENDPOINT = "https://ai-document-understanding.onrender.com/ask"

DATASET_PATH = "evaluation_dataset.json"

OUTPUT_DIR = "evaluation_output"

TIMEOUT_SECONDS = 120

RETRY_COUNT = 3

RETRY_DELAY_SECONDS = 2


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(path: str) -> List[Dict]:

    with open(path, "r", encoding="utf-8") as f:

        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a list")

    if len(dataset) == 0:
        raise ValueError("Dataset empty")

    return dataset


# ============================================================
# SEND REQUEST TO YOUR PRODUCTION SYSTEM
# ============================================================

def query_system(document_id: str, question: str) -> Dict:

    payload = {
        "document_id": document_id,
        "question": question
    }

    last_error = None

    for attempt in range(RETRY_COUNT):

        try:

            start_time = time.time()

            response = requests.post(
                ENDPOINT,
                json=payload,
                timeout=TIMEOUT_SECONDS
            )

            latency = time.time() - start_time

            if response.status_code == 200:

                data = response.json()

                return {
                    "success": True,
                    "latency": latency,
                    "answer": data.get("answer", ""),
                    "refused": data.get("refused", False),
                    "sources_used": data.get("sources_used", 0),
                    "confidence_score": data.get("confidence_score", None)
                }

            else:

                last_error = f"HTTP {response.status_code}"

        except Exception as e:

            last_error = str(e)

        time.sleep(RETRY_DELAY_SECONDS)

    return {
        "success": False,
        "error": last_error
    }


# ============================================================
# ACCURACY SCORING — AUDIT SAFE
# ============================================================

def compute_accuracy(answer: str, expected_keywords: List[str]) -> float:

    if not answer:
        return 0.0

    answer_lower = answer.lower()

    matches = 0

    for keyword in expected_keywords:

        if keyword.lower() in answer_lower:
            matches += 1

    return matches / len(expected_keywords)


# ============================================================
# CORRECT PERCENTILE CALCULATION
# ============================================================

def percentile(values: List[float], p: float) -> float:

    if not values:
        return 0.0

    values_sorted = sorted(values)

    k = int(round((p / 100) * (len(values_sorted) - 1)))

    return values_sorted[k]


# ============================================================
# MAIN EVALUATION LOGIC
# ============================================================

def run_evaluation():

    dataset = load_dataset(DATASET_PATH)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    results = []

    latencies = []

    accuracies = []

    grounded_count = 0

    success_count = 0

    refusal_count = 0

    print("\nStarting evaluation\n")

    for idx, item in enumerate(dataset):

        document_id = item["document_id"]

        question = item["question"]

        expected_keywords = item["expected_answer_contains"]

        print(f"[{idx+1}/{len(dataset)}] {question}")

        response = query_system(document_id, question)

        if not response["success"]:

            results.append({
                "question": question,
                "success": False
            })

            continue

        success_count += 1

        latency = response["latency"]

        answer = response["answer"]

        refused = response["refused"]

        sources_used = response["sources_used"]

        accuracy = compute_accuracy(answer, expected_keywords)

        grounded = (sources_used > 0) and (not refused)

        if grounded:
            grounded_count += 1

        if refused:
            refusal_count += 1

        latencies.append(latency)

        accuracies.append(accuracy)

        results.append({

            "question": question,

            "document_id": document_id,

            "answer": answer,

            "accuracy": accuracy,

            "grounded": grounded,

            "refused": refused,

            "sources_used": sources_used,

            "latency": latency,

            "timestamp": timestamp
        })

    total = len(dataset)

    success_rate = success_count / total

    grounded_rate = grounded_count / total

    refusal_rate = refusal_count / total

    avg_accuracy = statistics.mean(accuracies)

    avg_latency = statistics.mean(latencies)

    p50_latency = percentile(latencies, 50)

    p95_latency = percentile(latencies, 95)

    p99_latency = percentile(latencies, 99)

    summary = {

        "total_questions": total,

        "success_rate": success_rate,

        "grounded_answer_rate": grounded_rate,

        "refusal_rate": refusal_rate,

        "answer_accuracy": avg_accuracy,

        "avg_latency": avg_latency,

        "p50_latency": p50_latency,

        "p95_latency": p95_latency,

        "p99_latency": p99_latency,

        "timestamp": timestamp,

        "endpoint": ENDPOINT
    }

    # SAVE RESULTS

    results_path = f"{OUTPUT_DIR}/detailed_results_{timestamp}.json"

    summary_path = f"{OUTPUT_DIR}/summary_{timestamp}.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== FINAL METRICS ===\n")

    print(json.dumps(summary, indent=2))

    print(f"\nDetailed results saved: {results_path}")
    print(f"Summary saved: {summary_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_evaluation()