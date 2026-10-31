# Knowledge base

Ari answers factual questions **only** from the Markdown files in this folder (stage v2 and up).
Every `.md` file except this README is chunked by `##` heading and indexed; each section ends with a
`Source:` line, which the console turns into a link on the citation. The URL lines are stripped from
what the model reads, so answers cite `[n]` instead of reading URLs aloud.

Researched from the web on 2026-10-31, mostly from the official sites (emines-ingenieur.org, um6p.ma):

- `emines_overview.md`: identity, history, mission, UM6P/OCP link, partners, accreditation, key figures, leadership
- `emines_programs.md`: preparatory and engineering cycles, final-year options
- `emines_admissions.md`: eligibility, entrance exams, fees, scholarships
- `emines_internships_careers.md`: internships, international mobility, employment outcomes
- `emines_student_life.md`: student bureaus and clubs
- `emines_practical.md`: address, hours, getting there
- `um6p_overview.md`, `um6p_campuses.md`, `um6p_schools.md`, `um6p_research.md`, `um6p_campus_life.md`, `um6p_practical.md`
- `data/campus.json`: real UM6P Ben Guerir facilities for the `campus_directory` and `robot_navigate` tools
  (locations and hours only where published; walking times are demo values for the simulated robot)

Caveats recorded during research:
- um6p.ma served an expired TLS certificate on 2026-10-31; facts from those pages came from search-engine
  summaries of the official pages and should be re-checked when the site is fixed.
- Some figures differ between sources (student numbers, campus area, train times); files give both, with years.
- Opening hours of most campus facilities are not published anywhere, so Ari says so instead of guessing.

To update: edit or add files, and restart. The index is rebuilt automatically when the content changes.
