# Deploying MedRAG (free tiers)

MedRAG runs as a Hugging Face Docker Space (the app) plus a Supabase project (accounts, PDFs,
chunks, usage, encrypted keys). Both are free. You need a Groq API key for the shared quota.

## 1. Create the Supabase project
1. Sign in at supabase.com and create a new project (free plan). Pick a region near your users and save the database password somewhere safe.
2. In **Project Settings → API**, copy the **Project URL**, the **anon public** key and the **service_role** key. The service-role key bypasses all security rules: it goes only into Space secrets and your local `.env`, never into code, chat or screenshots.

## 2. Create the tables and the storage bucket
1. Open **SQL Editor → New query**, paste the contents of `supabase/schema.sql`, and run it. It is safe to run again.
2. Check **Table Editor** shows `books`, `chunks`, `usage` and `user_keys`, and **Storage** shows a private bucket named `books`.

## 3. Auth settings
1. **Authentication → Sign In / Providers → Email:** keep Email enabled. Turn **Confirm email** off (the app creates users already confirmed; this also stops Supabase sending mail it can't deliver on the free sender).
2. Password reset by email is not offered by the app. The free email sender is heavily rate-limited; this is a known limitation.

## 4. Generate the encryption secret
Run once and keep the output secret:

    docker compose run --rm dev python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

If you lose or change it, saved user Groq keys can no longer be read; users simply add them again.

## 5. Try it locally against Supabase (optional)
Add to `.env`: `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`, `MEDRAG_FERNET_KEY` (and `GROQ_API_KEY`, already there). Then:

    docker compose run --rm dev pytest -m slow tests/test_live_supabase.py -q
    docker compose run --rm -p 8501:8501 dev streamlit run app.py --server.address=0.0.0.0

With Supabase variables in `.env`, the app starts in multi-user mode. Set `MEDRAG_BACKEND=local` to get the single-book app back.

## 6. Create the Space
1. At huggingface.co → **New Space**: SDK **Docker**, template **Blank**, hardware **CPU basic (free)**, visibility public or private.
2. **Settings → Variables and secrets → New secret** for each: `GROQ_API_KEY`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`, `MEDRAG_FERNET_KEY`.
3. Create a Hugging Face **write** access token (Settings → Access Tokens).

## 7. Push the code
From the repo root:

    git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
    git push space v3-multiuser:main

Use your Hugging Face username and the write token as the password. The Space builds the `Dockerfile` (about 10 minutes the first time) and starts on port 7860 as declared in `README.md`'s front matter. Later deploys: push again.

## 8. Check it
Open the Space, sign up, upload a small text PDF you're allowed to use, wait for **Ready**, ask a question, open a source card, delete the book.

## Free-tier behaviour
- **The Space sleeps** after 48 hours without visitors; the next visit restarts it (about a minute while models load). Books being indexed at that moment are marked "interrupted"; users delete and re-upload them.
- **Supabase pauses** a free project after 7 days without activity. A paused project does not wake by itself: open the Supabase dashboard and click **Restore project** (free). Until then the app shows "The storage service is waking up or unavailable".
- **Limits** (change with environment variables, see `medrag/config.py`): 3 books per user, 50 MB / 1,000 pages per PDF, 60 books in total, 900 MB of PDFs, uploads pause above 400 MB of database, 10 shared-key answers per user per day.
- **Groq:** the shared key's free daily tokens are shared by all users; after their 10 daily answers, or if the shared key runs out, users add their own free key in Settings.
