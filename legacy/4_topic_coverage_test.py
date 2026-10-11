"""
METHOD 1: TOPIC COVERAGE TEST
Book: Diseases of Ear, Nose and Throat & Head and Neck Surgery
      by PL Dhingra, 7th Edition

Structure:
  Topic 1 - Tinnitus      : Q1  to Q5  (5 questions from book)
  Topic 3 - Sinusitis     : Q6 to Q10 (5 questions from book)
  Out-of-scope            : Q11 to Q15 (5 questions NOT in book)
  Total = 15 questions

HOW TO USE:
1. Start chatbot: python 2_app.py
2. Ask each question in the chatbot one by one
3. For each question, update:
     status -> "answered"   if chatbot gave a real answer
     status -> "not_found"  if chatbot said "not found in source book"
     answer -> paste the full chatbot answer (only for answered ones)
4. Run: python 6_topic_coverage_test.py
"""

from sentence_transformers import SentenceTransformer, util
import json

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ─── TOPIC 1: TINNITUS ────────────────────────────────────────────────────────
topic_tinnitus = {
    "topic_name": "Tinnitus",
    "questions": [
        {
            "question": "What is tinnitus?",
            "status": "answered",   # change to "answered" or "not_found"
            "answer": "Tinnitus is a ringing sound or noise in the ear, where the origin of the sound is within the patient (Source: Page 160).",          # paste chatbot answer here if answered
        },
        {
            "question": "What are the causes of subjective tinnitus?",
            "status": "answered",
            "answer": "The causes of subjective tinnitus include: otologic (e.g. impacted wax, fluid in middle ear, Ménière’s disease), metabolic (e.g. hypothyroidism, hyperthyroidism), neurologic (e.g. head injury, multiple sclerosis), cardiovascular (e.g. hypertension, hypotension), pharmacologic (e.g. certain drugs), and psychogenic (e.g. anxiety, depression) (Source: Page 160, Table 22.1).",
        },
        {
            "question": "What is the difference between subjective and objective tinnitus?",
            "status": "answered",
            "answer": "Subjective tinnitus can only be heard by the patient, whereas objective tinnitus can be heard by the examiner with the use of a stethoscope (Source: Page 160).",
        },
        {
            "question": "What is tinnitus retraining therapy?",
            "status": "answered",
            "answer": "Tinnitus retraining therapy (TRT) is a treatment approach that aims to help individuals with tinnitus habituate to their symptoms. (Source: Page 161)",
        },
        {
            "question": "What is a tinnitus masker and how does it work?",
            "status": "answered",
            "answer": "A tinnitus masker is a device used to help manage tinnitus, which is a ringing or noise in the ear. It works by producing a continuous, low-level, broad-band noise that helps to mask or cover up the tinnitus sound (Source: Page 161).",
        },
    ],
    "ground_truths": {
        "What is tinnitus?":
            "Tinnitus is ringing or noise in the ear originating within the patient. It may be unilateral or bilateral and varies in pitch described as roaring hissing swishing or clicking.",
        "What are the causes of subjective tinnitus?":
            "Subjective tinnitus is caused by otologic factors such as impacted wax, fluid in the middle ear, Meniere disease, otosclerosis, and acoustic neuroma. It also stems from systemic issues including metabolic (thyroid), cardiovascular (hypertension/hypotension), and neurologic disorders, as well as ototoxic drugs, noise exposure, and psychogenic factors like anxiety and depression.",
        "What is the difference between subjective and objective tinnitus?":
            "Subjective tinnitus is heard only by the patient while objective tinnitus can be heard by the examiner and is caused by vascular abnormalities like AV shunts or palatal myoclonus.",
        "What is tinnitus retraining therapy?":
            "Tinnitus retraining therapy is habituation therapy that blocks tinnitus related neuronal activity from reaching consciousness so patients lose awareness of tinnitus.",
        "What is a tinnitus masker and how does it work?":
            "Tinnitus maskers are worn like hearing aids and produce a sound that masks the tinnitus. They are used in patients with tinnitus who have no hearing loss.",
    }
}



# ─── TOPIC 2: SINUSITIS ──────────────────────────────────────────────────────
topic_sinusitis = {
    "topic_name": "Sinusitis",
    "questions": [
        {
            "question": "What is rhinosinusitis?",
            "status": "answered",
            "answer": "Rhinosinusitis is a term used to describe inflammation of the mucosa of the nasal passages and paranasal sinuses. It is often used interchangeably with the term sinusitis, but rhinosinusitis is a more accurate term as it acknowledges that the nasal mucosa is also involved in the disease process. (Source: Page 228)",
        },
        {
            "question": "What are the symptoms of acute maxillary sinusitis?",
            "status": "answered",
            "answer": "The symptoms of acute maxillary sinusitis include frontal headache, tenderness, and purulent nasal discharge. The headache is usually severe and localized over the affected sinus, and may be accompanied by facial pain and pressure. The tenderness can be elicited by tapping over the anterior wall of the maxillary sinus or by pressing upwards on the floor of the sinus. The nasal discharge is typically purulent and may be accompanied by a foul odor. (Source: Page 230",
        },
        {
            "question": "How do you treat sinusitis?",
            "status": "answered",
            "answer": "Treatment of sinusitis can be medical or surgical. Medical treatment includes antimicrobial drugs, nasal decongestant drops, steam inhalation, analgesics, and hot fomentation (Source: Page 230). For acute bacterial rhinosinusitis, treatment strategies include analgesics, antibiotics, and saline irrigations (Source: Page 228). Surgical treatment, such as antral lavage or trephination of the frontal sinus, may be necessary in some cases (Source: Page 230-231).",
        },
        {
            "question": "What are the orbital complications of sinusitis?",
            "status": "answered",
            "answer": '''The orbital complications of sinusitis include:
Inflammatory oedema of lids,
Subperiosteal abscess,
Orbital cellulitis,
Orbital abscess,
Superior orbital fissure syndrome, and
Orbital apex syndrome (Source: Page 240).''',
        },
        {
            "question": "What is FESS functional endoscopic sinus surgery?",
            "status": "answered",
            "answer": "FESS (Functional Endoscopic Sinus Surgery) is a surgical procedure used to treat sinusitis and other sinus-related disorders. It involves the use of an endoscope, a thin, flexible tube with a camera and light on the end, to visualize the sinuses and remove any blockages or abnormalities. The goal of FESS is to restore normal drainage and ventilation to the sinuses, thereby relieving symptoms such as congestion, headache, and facial pain. (Source: Page 493)",
        },
    ],
    "ground_truths": {
        "What is rhinosinusitis?":
            "Rhinosinusitis is the inflammation of the nasal cavity and the paranasal sinuses, typically characterized by symptoms like nasal congestion, discharge, and facial pressure.",
        "What are the symptoms of acute maxillary sinusitis?":
            "Symptoms include severe frontal headache localized over the sinus, facial pain, pressure, and purulent nasal discharge with a foul odor. Tenderness is elicited by tapping the anterior wall or pressing the sinus floor.",
        "How do you treat sinusitis?":
            "Treatment includes antimicrobial drugs such as amoxicillin decongestant nasal drops analgesics steam inhalation and nasal saline irrigation. Surgical drainage is reserved for complications.",
        "What are the orbital complications of sinusitis?":
            "Orbital complications of sinusitis include preseptal inflammatory oedema of eyelids subperiosteal abscess orbital cellulitis orbital abscess superior orbital fissure syndrome and orbital apex syndrome. These arise from ethmoid sinuses spreading to the orbit.",
        "What is FESS functional endoscopic sinus surgery?":
            "FESS is the surgical treatment of choice for chronic rhinosinusitis. It uses rigid endoscopes to correct structural variants and remove obstructions to restore sinus drainage and ventilation.",
    }
}

# ─── OUT-OF-SCOPE QUESTIONS ──────────────────────────────────────────────────
out_of_scope = {
    "topic_name": "Out-of-Scope",
    "questions": [
        {
            "question": "What is the recommended COVID-19 vaccine schedule for 2025?",
            "status": "not_found",
            "answer": "",
            "expected": "refused",
        },
        {
            "question": "How do ACE inhibitors lower blood pressure?",
            "status": "not_found",
            "answer": "",
            "expected": "refused",
        },
        {
            "question": "What is the treatment for Type 2 diabetes?",
            "status": "not_found",
            "answer": "",
            "expected": "refused",
        },
        {
            "question": "What are the symptoms of appendicitis?",
            "status": "not_found",
            "answer": "",
            "expected": "refused",
        },
        {
            "question": "What is the current price of antibiotics in India?",
            "status": "not_found",
            "answer": "",
            "expected": "refused",
        },
    ]
}

# ─── SCORING ─────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    e1 = model.encode(a, convert_to_tensor=True)
    e2 = model.encode(b, convert_to_tensor=True)
    return round(float(util.cos_sim(e1, e2)), 4)


def evaluate_book_topic(topic):
    name      = topic["topic_name"]
    questions = topic["questions"]
    gts       = topic.get("ground_truths", {})

    answered  = [q for q in questions if q["status"] == "answered"]
    not_found = [q for q in questions if q["status"] == "not_found"]
    pending   = [q for q in questions if q["status"] == "pending"]

    total    = len(questions)
    n_ans    = len(answered)
    coverage = round(n_ans / total, 4) if total > 0 else 0

    # Answer Relevancy: cosine similarity between chatbot answer and ground truth
    rel_scores = []
    for q in answered:
        if q["answer"] and q["question"] in gts:
            score = cosine_sim(q["answer"], gts[q["question"]])
            rel_scores.append({"question": q["question"], "score": score})

    avg_rel = round(sum(r["score"] for r in rel_scores) / len(rel_scores), 4) if rel_scores else None

    return {
        "topic"              : name,
        "total"              : total,
        "answered"           : n_ans,
        "not_found"          : len(not_found),
        "pending"            : len(pending),
        "coverage_rate"      : coverage,
        "coverage_pct"       : f"{round(coverage * 100, 1)}%",
        "avg_answer_relevancy": avg_rel,
        "relevancy_per_q"    : rel_scores,
        "answered_qs"        : [q["question"] for q in answered],
        "not_found_qs"       : [q["question"] for q in not_found],
    }


def evaluate_out_of_scope(topic):
    questions = topic["questions"]
    refused   = [q for q in questions if q["status"] == "not_found"]
    answered  = [q for q in questions if q["status"] == "answered"]
    pending   = [q for q in questions if q["status"] == "pending"]
    total     = len(questions)
    tested    = len(refused) + len(answered)
    refusal   = round(len(refused) / tested, 4) if tested > 0 else None
    return {
        "topic"          : topic["topic_name"],
        "total"          : total,
        "correctly_refused": len(refused),
        "incorrectly_answered": len(answered),
        "pending"        : len(pending),
        "refusal_rate"   : refusal,
        "refusal_pct"    : f"{round(refusal * 100, 1)}%" if refusal is not None else "N/A",
    }


def main():
    book_topics = [topic_tinnitus, topic_sinusitis]

    print("\n" + "="*62)
    print("   METHOD 1: TOPIC COVERAGE TEST RESULTS")
    print("   Book: Dhingra ENT 7th Edition")
    print("="*62)

    all_coverage   = []
    all_relevancy  = []
    all_results    = []

    for topic in book_topics:
        r = evaluate_book_topic(topic)
        all_results.append(r)
        if r["pending"] < r["total"]:
            all_coverage.append(r["coverage_rate"])
            if r["avg_answer_relevancy"]:
                all_relevancy.append(r["avg_answer_relevancy"])

        print(f"\n{'─'*40}")
        print(f"TOPIC: {r['topic']}")
        print(f"  Answered      : {r['answered']} / {r['total']}  ({r['coverage_pct']})")
        print(f"  Not found     : {r['not_found']}")
        print(f"  Pending       : {r['pending']}")
        if r["avg_answer_relevancy"]:
            print(f"  Avg Relevancy : {r['avg_answer_relevancy']} / 1.0")
        if r["relevancy_per_q"]:
            for rq in r["relevancy_per_q"]:
                short_q = rq["question"][:50]
                print(f"    [{rq['score']}]  {short_q}")
        if r["not_found_qs"]:
            print(f"  Not found questions:")
            for q in r["not_found_qs"]:
                print(f"    - {q}")

    # Out-of-scope
    oos = evaluate_out_of_scope(out_of_scope)
    print(f"\n{'─'*40}")
    print(f"OUT-OF-SCOPE REFUSAL TEST")
    print(f"  Correctly refused   : {oos['correctly_refused']} / {oos['total']}")
    print(f"  Incorrectly answered: {oos['incorrectly_answered']}")
    print(f"  Pending             : {oos['pending']}")
    if oos["refusal_rate"] is not None:
        print(f"  Refusal Rate        : {oos['refusal_pct']}")

    # Overall summary
    tested_topics = [r for r in all_results if r["pending"] < r["total"]]
    if tested_topics:
        avg_cov = round(sum(all_coverage) / len(all_coverage), 4) if all_coverage else None
        avg_rel = round(sum(all_relevancy) / len(all_relevancy), 4) if all_relevancy else None

        print(f"\n{'='*62}")
        print("   OVERALL SUMMARY")
        print(f"{'='*62}")
        print(f"  Book topics tested       : {len(tested_topics)}")
        print(f"  Total book questions     : {sum(r['total'] for r in tested_topics)}")
        print(f"  Total answered           : {sum(r['answered'] for r in tested_topics)}")
        if avg_cov:
            print(f"  Average Coverage Rate    : {round(avg_cov*100,1)}%")
        if avg_rel:
            print(f"  Average Answer Relevancy : {avg_rel} / 1.0")
        print(f"  Out-of-scope refusal     : {oos['refusal_pct']}")
        print(f"{'='*62}")

        # Save
        output = {
            "summary": {
                "book_topics_tested"     : len(tested_topics),
                "total_book_questions"   : sum(r["total"] for r in tested_topics),
                "total_answered"         : sum(r["answered"] for r in tested_topics),
                "avg_coverage_rate"      : avg_cov,
                "avg_answer_relevancy"   : avg_rel,
                "out_of_scope_refusal_rate": oos["refusal_rate"],
            },
            "per_topic": all_results,
            "out_of_scope": oos,
        }
        with open("topic_coverage_results.json", "w") as f:
            json.dump(output, f, indent=2)
        with open("topic_coverage_results.txt", "w") as f:
            f.write("METHOD 1 — Topic Coverage Test\n")
            f.write("Book: Dhingra ENT 7th Edition\n")
            f.write("="*40 + "\n")
            if avg_cov:
                f.write(f"Avg Coverage Rate    : {round(avg_cov*100,1)}%\n")
            if avg_rel:
                f.write(f"Avg Answer Relevancy : {avg_rel}\n")
            f.write(f"Out-of-scope refusal : {oos['refusal_pct']}\n\n")
            for r in all_results:
                f.write(f"\nTOPIC: {r['topic']}\n")
                f.write(f"  Answered : {r['answered']}/{r['total']} ({r['coverage_pct']})\n")
                f.write(f"  Not found: {r['not_found']}\n")
                if r["avg_answer_relevancy"]:
                    f.write(f"  Relevancy: {r['avg_answer_relevancy']}\n")

        print("\nSaved: topic_coverage_results.txt + topic_coverage_results.json")
        print("Use these numbers in your research paper Results section!\n")

    else:
        print("\nNo questions filled yet. Ask them in the chatbot first, then fill in the results above.")


if __name__ == "__main__":
    main()
