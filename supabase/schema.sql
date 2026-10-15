-- MedRAG multi-user schema. Run once in the Supabase SQL editor (safe to re-run).
-- The app talks to Postgres with the service-role key (server-side only), which bypasses RLS;
-- the app filters every query by user_id. RLS below is defence in depth for any future
-- anon/authenticated access: users may read only their own rows and write nothing directly.

create table if not exists public.books (
  id uuid primary key,
  user_id uuid not null references auth.users (id) on delete cascade,
  title text not null check (char_length(title) between 1 and 200),
  filename text not null,
  storage_path text not null,
  size_bytes bigint not null check (size_bytes >= 0),
  pages int not null default 0,
  chunks int not null default 0,
  status text not null check (status in ('processing', 'ready', 'failed')),
  progress text not null default '',
  error text not null default '',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists books_user_id_idx on public.books (user_id);
create index if not exists books_status_idx on public.books (status);

create table if not exists public.chunks (
  id text primary key,
  book_id uuid not null references public.books (id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  text text not null,
  page_index int not null,
  page_label text not null,
  section text not null default '',
  topic text not null default '',
  embedding real[] not null
);
create index if not exists chunks_book_id_idx on public.chunks (book_id);
create index if not exists chunks_user_id_idx on public.chunks (user_id);

create table if not exists public.usage (
  user_id uuid not null references auth.users (id) on delete cascade,
  day date not null,
  shared_key_questions int not null default 0 check (shared_key_questions >= 0),
  primary key (user_id, day)
);

create table if not exists public.user_keys (
  user_id uuid primary key references auth.users (id) on delete cascade,
  encrypted_groq_key text not null,
  created_at timestamptz not null default now()
);

alter table public.books enable row level security;
alter table public.chunks enable row level security;
alter table public.usage enable row level security;
alter table public.user_keys enable row level security;  -- no policy: nobody but the server

drop policy if exists "read own books" on public.books;
create policy "read own books" on public.books for select to authenticated using (user_id = auth.uid());
drop policy if exists "read own chunks" on public.chunks;
create policy "read own chunks" on public.chunks for select to authenticated using (user_id = auth.uid());
drop policy if exists "read own usage" on public.usage;
create policy "read own usage" on public.usage for select to authenticated using (user_id = auth.uid());

create or replace function public.increment_usage(p_user uuid, p_day date)
returns int language sql security invoker set search_path = '' as $$
  insert into public.usage (user_id, day, shared_key_questions) values (p_user, p_day, 1)
  on conflict (user_id, day) do update set shared_key_questions = public.usage.shared_key_questions + 1
  returning shared_key_questions;
$$;

create or replace function public.database_bytes()
returns bigint language sql stable security invoker set search_path = '' as $$
  select pg_database_size(current_database());
$$;

revoke all on function public.increment_usage(uuid, date) from public, anon, authenticated;
revoke all on function public.database_bytes() from public, anon, authenticated;
grant execute on function public.increment_usage(uuid, date) to service_role;
grant execute on function public.database_bytes() to service_role;

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('books', 'books', false, 52428800, array['application/pdf'])
on conflict (id) do nothing;
