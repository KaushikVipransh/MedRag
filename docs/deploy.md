# Deploying MedRAG (free tiers)

MedRAG runs on Streamlit Community Cloud (the app) plus a Supabase project (accounts, PDFs,
chunks, usage, encrypted keys). Both are free and need no credit card. You need a Groq API key
for the shared quota.

Hugging Face Docker Spaces were the original plan, but since mid-2026 new free accounts can't
create Docker or Gradio Spaces on free CPU (it needs PRO). The `Dockerfile` is kept for any
Docker host.

## 1. Create the Supabase project
1. Sign in at supabase.com and create a new project (free plan). Choose a **US East** region:
   the app runs in the US and makes several database calls per question, so this keeps answers
   fast for everyone. Save the database password somewhere safe.
2. In **Project Settings → API**, copy the **Project URL**, the **anon public** key and the
   **service_role** key. The service-role key bypasses all security rules: it goes only into
   the app's secrets and your local `.env`, never into code, chat or screenshots.

## 2. Create the tables and the storage bucket
1. Open **SQL Editor → New query**, paste the contents of `supabase/schema.sql`, and run it.
   It is safe to run again.
2. Check **Table Editor** shows `books`, `chunks`, `usage` and `user_keys`, and **Storage**
   shows a private bucket named `books`.

## 3. Auth settings
1. **Authentication → Sign In / Providers → Email:** keep Email enabled. Turn **Confirm email**
   off (the app creates users already confirmed; this also stops Supabase sending mail it
   can't deliver on the free sender).
2. Password reset by email is not offered by the app. The free email sender is heavily
   rate-limited; this is a known limitation.

## 4. Generate the encryption secret
Run once and keep the output secret:

    docker compose run --rm dev python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

If you lose or change it, saved user Groq keys can no longer be read; users simply add them again.

## 5. Check it locally against Supabase
Add to `.env`: `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`, `MEDRAG_FERNET_KEY`
(and `GROQ_API_KEY`, already there), plus `MEDRAG_BACKEND=local` so ordinary local runs stay
the single-book app. Then run the round-trip test (it uses fake models, so no Groq quota):

    docker compose run --rm -e MEDRAG_BACKEND=supabase dev pytest -m slow tests/test_live_supabase.py -q

Every app start (and the live test) marks books still being indexed as failed
("interrupted"), because each process assumes it owns the indexing queue. Running the app
locally against the production project therefore fails any upload the live app is indexing
at that moment; prefer a separate Supabase project for local experiments.

## 6. Deploy on Streamlit Community Cloud
1. Sign in at share.streamlit.io with the GitHub account that owns the repository.
2. **Create app → Deploy a public app from GitHub:** repository `<you>/MedRag`, branch
   `v3-multiuser` (or `main` once merged), main file `app.py`, and pick an app URL.
3. **Advanced settings:** Python version **3.12**. Under **Secrets**, paste:

       GROQ_API_KEY = "..."
       SUPABASE_URL = "https://<project-ref>.supabase.co"
       SUPABASE_ANON_KEY = "..."
       SUPABASE_SERVICE_KEY = "..."
       MEDRAG_FERNET_KEY = "..."

4. **Deploy.** The first build installs the requirements (CPU-only torch) and the first visit
   downloads the three models (about 1.3 GB), so it takes several minutes. Later deploys
   happen automatically on every push to that branch.

## 7. Check it
Open the app, sign up, upload a small text PDF you're allowed to use, wait for **Ready**,
ask a question, open a source card, delete the book.

## Free-tier behaviour
- **The app sleeps** after 12 hours without visitors; the next visit wakes it (a few minutes
  while the models download and load). Books being indexed at that moment are marked
  "interrupted"; users delete and re-upload them.
- **Memory:** the app uses about 1.4–1.8 GB with all three models loaded. Community Cloud
  allows up to 2.7 GB but only guarantees 690 MB, so under heavy load the app may be
  restarted.
- **Supabase pauses** a free project after 7 days without activity. A paused project does
  not wake by itself: open the Supabase dashboard and click **Restore project** (free). Until
  then the app shows "The storage service is waking up or unavailable".
- **Limits** (change with environment variables or secrets, see `medrag/config.py`): 3 books
  per user, 50 MB / 1,000 pages per PDF, 60 books in total, 900 MB of PDFs, uploads pause
  above 400 MB of database, 10 shared-key answers per user per day.
- **Groq:** the shared key's free daily tokens are shared by all users; after their 10 daily
  answers, or if the shared key runs out, users add their own free key in Settings.
