# Knowledge base

Ari answers factual questions **only** from the Markdown files in this folder (stage v2 and up).
Every `.md` file except this README is chunked by heading and indexed into the vector store.

> **Sample data.** Opening hours, rooms, rules and timetables in these files are illustrative
> placeholders written for the demo. Replace them with official EMINES documents before showing
> Ari to real users. Delete `data/index/` (or just edit a file: the index fingerprint changes)
> and the index rebuilds on the next start.

- `about_ari.md`: what the robot is and what it can do
- `emines_overview.md`: the school, at a high level
- `campus_services.md`: library, FabLab, cafeteria, IT help desk, reception
- `academic_rules.md`: attendance, exams, grading, internships
- `data/campus.json`: places used by the `campus_directory` and `robot_navigate` tools
- `data/schedule.json`: timetable used by the `class_schedule` tool
