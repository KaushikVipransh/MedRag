"""
METHOD 2: CONSISTENCY TEST
Ask the same question 3 times, measure how similar the answers are.
A high score (>0.85) means the chatbot gives stable, reliable answers.

HOW TO USE:
1. Go to your chatbot at http://localhost:7860
2. For each question below, ask it 3 separate times (clear chat between each)
3. Copy each answer into the variables below
4. Run: python 5_consistency_test.py
"""

from sentence_transformers import SentenceTransformer, util
import json

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────────────────────────────────────────
# FILL IN YOUR ANSWERS BELOW
# Copy the full chatbot answer each time (including the Sources: line)
# ─────────────────────────────────────────────────────────────────────────────

questions_and_answers = {

    "Q1: what is Ménière’s Disease?": {
        "answer_1": "Ménière's disease, also called endolymphatic hydrops, is a disorder of the inner ear where the endolymphatic system is distended with endolymph. It is characterized by (i) vertigo, (ii) sensorineural hearing loss, (iii) tinnitus, and (iv) aural fullness (Source: Page 126).",
        "answer_2": "Ménière's disease, also called endolymphatic hydrops, is a disorder of the inner ear where the endolymphatic system is distended with endolymph. It is characterized by (i) vertigo, (ii) sensorineural hearing loss, (iii) tinnitus, and (iv) aural fullness (Source: Page 126).",
        "answer_3": "Ménière's disease, also called endolymphatic hydrops, is a disorder of the inner ear where the endolymphatic system is distended with endolymph. It is characterized by (i) vertigo, (ii) sensorineural hearing loss, (iii) tinnitus, and (iv) aural fullness (Source: Page 126).",
    },

    "Q2: what are the PRENATAL CAUSES of hearing loss?": {
        "answer_1": "According to Page 144, the prenatal causes of hearing loss can be divided into two categories: Infant factors: (a) Inner ear anomalies due to genetic or nongenetic causes, which can be nonsyndromic (affecting the inner ear alone) or syndromic (part of a larger syndrome). Specific anomalies include: (a) Scheibe dysplasia (cochleosaccular dysplasia), (b) Alexander dysplasia (affecting the basal turn of the membranous cochlea), (c) Bing-Siebenmann dysplasia (complete absence of the membranous labyrinth), (d) Michel aplasia (complete absence of the bony and membranous labyrinth), and (e) Mondini dysplasia (incomplete partition of the cochlea). Maternal factors: (a) Infections during pregnancy (e.g., toxoplasmosis, rubella, cytomegalovirus, herpes), (b) certain drugs taken during pregnancy (e.g., streptomycin, gentamicin, tobramycin, amikacin, quinine, chloroquine, thalidomide), (c) radiation exposure during the first trimester, and (d) other factors such as nutritional deficiency, diabetes, toxemia, and thyroid deficiency, as well as maternal alcoholism. These prenatal causes can contribute to hearing loss in children. (Source: Page 144)",
        "answer_2": "According to Page 144, the prenatal causes of hearing loss can be divided into two categories:Infant factors: (a) Inner ear anomalies due to genetic or nongenetic causes, which can be nonsyndromic (affecting the inner ear alone) or syndromic (part of a larger syndrome). Specific anomalies include: (a) Scheibe dysplasia (cochleosaccular dysplasia), (b) Alexander dysplasia (affecting the basal turn of the membranous cochlea), (c) Bing-Siebenmann dysplasia (complete absence of the membranous labyrinth), (d) Michel aplasia (complete absence of the bony and membranous labyrinth), and (e) Mondini dysplasia (incomplete partition between the scalae due to the absence of the osseous spiral lamina). (f) Enlarged vestibular aqueduct, (g) Semicircular canal malformations. Maternal factors: (a) Infections during pregnancy (e.g., toxoplasmosis, rubella, cytomegalovirus, herpes), (b) Drugs during pregnancy (e.g., streptomycin, gentamicin, tobramycin, amikacin, quinine, chloroquine, thalidomide), (c) Radiation to the mother in the first trimester, and (d) Other factors such as nutritional deficiency, diabetes, toxemia, and thyroid deficiency, as well as maternal alcoholism.(Source: Page 144)",
        "answer_3": '''According to Page 144, the prenatal causes of hearing loss can be divided into two categories:
Infant factors: (a) Inner ear anomalies due to genetic or nongenetic causes, which can be nonsyndromic (affecting the inner ear alone) or syndromic (part of a larger syndrome). (b) Specific anomalies include:

Scheibe dysplasia (cochleosaccular dysplasia)
Alexander dysplasia (affecting the basal turn of the membranous cochlea)
Bing-Siebenmann dysplasia (complete absence of the membranous labyrinth)
Michel aplasia (complete absence of the bony and membranous labyrinth)
Mondini dysplasia (incomplete partition of the cochlea)
Enlarged vestibular aqueduct
Semicircular canal malformations
Maternal factors: (a) Infections during pregnancy (e.g., toxoplasmosis, rubella, cytomegalovirus, herpes) (b) Drugs during pregnancy (e.g., streptomycin, gentamicin, tobramycin, amikacin, quinine, chloroquine, thalidomide) (c) Radiation to the mother in the first trimester (d) Other factors (e.g., nutritional deficiency, diabetes, toxemia, thyroid deficiency, maternal alcoholism) (Source: Page 144)

''',
    },

    "Q3: What is tinnitus?": {
        "answer_1": "Tinnitus is a ringing sound or noise in the ear, where the origin of the sound is within the patient (Source: Page 160).",
        "answer_2": "Tinnitus is a ringing sound or noise in the ear that originates from within the patient, characterized by a variety of sounds such as roaring, hissing, swishing, rustling, or clicking (Source: Page 145).",
        "answer_3": "Tinnitus is a ringing sound or noise in the ear, characterized by its origin being within the patient, and can vary in pitch and loudness (Source: Page 160).",
    },

    "Q4: how do you treat sinusitis?": {
        "answer_1": "Treatment of sinusitis can be medical or surgical. Medical treatment includes antimicrobial drugs, nasal decongestant drops, steam inhalation, analgesics, and hot fomentation (Source: Page 230). For acute bacterial rhinosinusitis, treatment strategies include analgesics, antibiotics, and saline irrigations (Source: Page 228). Surgical treatment, such as antral lavage or trephination of the frontal sinus, may be necessary in some cases (Source: Page 230-231).",
        "answer_2": '''Treatment of sinusitis can be medical or surgical. Medical treatment includes antimicrobial drugs, nasal decongestant drops, steam inhalation, analgesics, and hot fomentation (Source: Page 230).

For acute bacterial rhinosinusitis, treatment strategies include analgesics, antibiotics, and saline irrigations (Source: Page 228).

Surgical treatment, such as antral lavage or trephination of the frontal sinus, may be necessary in some cases (Source: Page 230-231).

It's also important to note that treatment should be based on the severity and duration of symptoms, as well as the presence of any underlying conditions or complications (Source: Page 228-231).''',
        "answer_3": '''According to the provided context, treatment for sinusitis can be medical or surgical.

Medical treatment includes:

Antimicrobial drugs such as ampicillin, amoxicillin, erythromycin, or doxycycline (Page 230).
Nasal decongestant drops like 1 percent 
ephedrine or 0.1 percent xylo- or oxymetazoline (Page 230).
Steam inhalation with menthol or Tr. Benzoin Co. for symptomatic relief (Page 230).
Analgesics like paracetamol for pain relief (Page 230).
Hot fomentation to the affected sinus for soothing and relief (Page 230).
Surgical treatment includes:

Antral lavage, which is rarely necessary and done only when medical treatment fails (Page 230).
Trephination of the frontal sinus, which involves draining the sinus externally through an incision (Page 231).
It's also mentioned that treatment strategies for acute bacterial rhinosinusitis include analgesics, antibiotics, and saline irrigations (Page 228).

Additionally, for acute viral rhinosinusitis, treatment is symptomatic, using topical nasal decongestants, antihistamines, and analgesics, with plenty of fluid intake encouraged (Page 228).

Please note that these treatments are based on the information provided in the context and may not be comprehensive or applicable to all cases of sinusitis. (Source: Pages 228-231)

''',
    },

    "Q5:Nasal septum consists of what parts?": {
        "answer_1": '''The nasal septum consists of three parts:

Columellar Septum,
Membranous Septum, and
Septum Proper (Source: Page 180).''',
        "answer_2": '''The nasal septum consists of three parts:

Columellar Septum,
Membranous Septum, and
Septum Proper (Source: Page 180).''',
        "answer_3": '''The nasal septum consists of three parts:

Columellar Septum,
Membranous Septum, and
Septum Proper (Source: Page 180).
''',
    },

}

# ─────────────────────────────────────────────────────────────────────────────

def cosine(a, b):
    e1 = model.encode(a, convert_to_tensor=True)
    e2 = model.encode(b, convert_to_tensor=True)
    return round(float(util.cos_sim(e1, e2)), 4)

def score_question(q_label, answers):
    a1 = answers["answer_1"]
    a2 = answers["answer_2"]
    a3 = answers["answer_3"]

    if "PASTE ANSWER" in a1 or "PASTE ANSWER" in a2 or "PASTE ANSWER" in a3:
        return None

    s12 = cosine(a1, a2)
    s13 = cosine(a1, a3)
    s23 = cosine(a2, a3)
    avg = round((s12 + s13 + s23) / 3, 4)

    return {
        "question": q_label,
        "sim_1_vs_2": s12,
        "sim_1_vs_3": s13,
        "sim_2_vs_3": s23,
        "avg_consistency": avg,
        "consistent": avg >= 0.85
    }


def main():
    print("\n" + "="*60)
    print("   METHOD 2: CONSISTENCY TEST RESULTS")
    print("="*60)

    all_scores = []
    skipped    = []

    for q_label, answers in questions_and_answers.items():
        result = score_question(q_label, answers)
        if result is None:
            skipped.append(q_label)
            continue

        all_scores.append(result["avg_consistency"])

        status = "CONSISTENT" if result["consistent"] else "INCONSISTENT"
        print(f"\n{q_label}")
        print(f"  Answer 1 vs Answer 2 : {result['sim_1_vs_2']}")
        print(f"  Answer 1 vs Answer 3 : {result['sim_1_vs_3']}")
        print(f"  Answer 2 vs Answer 3 : {result['sim_2_vs_3']}")
        print(f"  Average Consistency  : {result['avg_consistency']}  [{status}]")

    if skipped:
        print(f"\n  Skipped (not filled): {len(skipped)} questions")
        for s in skipped:
            print(f"    - {s}")

    if all_scores:
        overall = round(sum(all_scores) / len(all_scores), 4)
        print("\n" + "="*60)
        print(f"  OVERALL CONSISTENCY SCORE: {overall} / 1.0")
        print(f"  Questions evaluated      : {len(all_scores)}")
        if overall >= 0.90:
            print("  Rating: EXCELLENT - Answers are highly stable")
        elif overall >= 0.85:
            print("  Rating: GOOD - Answers are reliably consistent")
        elif overall >= 0.75:
            print("  Rating: FAIR - Some variation between answers")
        else:
            print("  Rating: POOR - High variation, may need lower temperature")
        print("="*60)

        # Save results
        results = {
            "overall_consistency": overall,
            "questions_evaluated": len(all_scores),
            "per_question": [score_question(q, a) for q, a in questions_and_answers.items()
                             if score_question(q, a) is not None]
        }
        with open("consistency_results.json", "w") as f:
            json.dump(results, f, indent=2)
        with open("consistency_results.txt", "w") as f:
            f.write("METHOD 2 — Consistency Test Results\n")
            f.write("="*40 + "\n")
            f.write(f"Overall Consistency Score: {overall}\n")
            f.write(f"Questions evaluated: {len(all_scores)}\n\n")
            for r in results["per_question"]:
                f.write(f"\n{r['question']}\n")
                f.write(f"  1v2: {r['sim_1_vs_2']} | 1v3: {r['sim_1_vs_3']} | 2v3: {r['sim_2_vs_3']}\n")
                f.write(f"  Avg: {r['avg_consistency']} — {'CONSISTENT' if r['consistent'] else 'INCONSISTENT'}\n")

        print("\nSaved: consistency_results.txt + consistency_results.json")
        print("Use the OVERALL CONSISTENCY SCORE in your research paper!\n")


if __name__ == "__main__":
    main()
