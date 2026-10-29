# Spurs–Kings: README media brief

**Purpose:** show the engineering result to a reader who will not install the
project. Use real outputs from the completed Sacramento Kings–San Antonio Spurs
analysis. No new screenshots or videos are produced during the README rewrite.

## Capture dependency

The candidate run is
`runs/analysis/sac-sas-2026-10-27-native-cuda-optimized/`.
At preparation time it is still processing. Its eventual manifest, outputs and
health report must be checked before selecting examples or quoting performance.
If the run finishes with errors or partial capabilities, diagnose the result
before deciding what it can demonstrate.

Use the actual roster/team names in the finished viewer. The working name
“Spurs–Kings” does not specify scoreboard home/away order or an inferred result.
Exact timestamps, frame numbers, subjects and team-map coverage are selected
after completion; they are not copied from HOU–LAL or GSW–DEN examples.

## Deliverables

| Asset | Format and target | Placement | What it should establish |
| --- | --- | --- | --- |
| Main walkthrough | H.264 MP4, 1920 × 1080, 30–40 s, English captions, usable without sound | Link or GitHub-hosted video immediately below the demo introduction | A coherent journey from real predictions to evidence a person can review |
| Inline preview | GIF, about 960 px wide, 6–8 s; target below 5 MB if labels remain legible | Clickable preview of the walkthrough in the main demo slot | Immediate proof that the product works on real footage |
| Geometry still | PNG, about 1920 px wide; source video and projected court in one composition | Demo gallery or engineering-decisions section | The relationship between image-space observations and court-space reasoning |
| Analysis still | PNG, about 1920 px wide; a readable player profile and team-map view as two labeled panels if necessary | Demo gallery | Useful statistics with measured duration/coverage, and spatial context with explicit missing data |
| Capture manifest | JSON kept with the media preparation artifacts | Link from the final media evidence note | Traceability to the source, run, code, selected frames and generated files |

The final README should embed the primary preview and offer one clear full-video
action. Supporting stills can sit in a compact gallery or a collapsible section.
Avoid filling the first screen with several competing media blocks.

## Walkthrough storyboard

| Time | Scene | On-screen English caption | Selection condition |
| --- | --- | --- | --- |
| 0–7 s | A short passage of real play with readable player/referee, ball and court overlays | “Broadcast video → synchronized computer vision outputs” | Ball visible, overlay alignment checked, no scene chosen solely because it hides common failures |
| 7–13 s | Pause and step a few frames; briefly switch a relevant layer | “Inspect the source frame behind each prediction” | The displayed frame, masks and timestamps agree; toggling clarifies rather than clutters |
| 13–23 s | Select an established player and open an attributed probable shot in the existing review dialog | “From a player profile to a reviewable shot proposal” | The completed run contains a suitable event; verify the visible player and retained attribution |
| 23–31 s | Show team presence beside shot origins, with a brief offense/defense change | “Compare spatial context without losing the evidence” | Sufficient located events and orientation/coverage support are available; keep unavailable counts readable |
| 31–37 s | Finish with a compact run-provenance card and one concrete uncertainty | “Traceable runs. Explicit uncertainty. Reproducible decisions.” | Values come from the completed manifest and health report; show a genuine unresolved identity, candidate or excluded interval |

If the run has no convincing attributed shot, replace the player-shot scene with
frame-level investigation of a real candidate and explain its unresolved
attribution. If a team map is unavailable, show that state and its reason. Do not
inject API fixtures, fabricate detections or relabel another run as Spurs–Kings
to satisfy the storyboard.

The video demonstrates an analysis computed before playback. Add a small
“Precomputed analysis • playback at 1×” label. If a segment is sped up, state its
playback rate. The processing-time result belongs in a separately sourced caption
or performance table, with loading and postprocessing boundaries identified.

## Still-image composition

**Geometry:** use one verified moment with a usable calibration. Keep the video
large enough to inspect feet, boxes/masks and court lines. Put the tactical court
beside it with the same selected moment. Label the panels “Source frame” and
“Projected court”; include source time and frame index. The image supports an
explanation of projection, rather than a claim of measured geometric accuracy.

**Analysis:** make player name/identity status, observed distance, measured
duration and coverage legible. Pair it with actual team presence and shot-origin
maps. If two different viewer states are composed, mark them as separate panels
and identify their time/filter scope. Keep probable/candidate markers and counts
visible. Do not imply that whole-clip maps represent just the current frame.

## Visual treatment and accessibility

- Use the existing charcoal/orange studio design and team display colors.
- Capture the viewer in English, after fonts and observations have loaded.
- Crop empty browser chrome; retain the controls, time and legends that explain
  the result. Use direct captures of the outputs, with clearly separate captions.
- Keep captions short, high contrast and away from the ball/feet. Export an SRT
  alongside the MP4; the GIF must communicate without narration.
- Write alt text identifying the actual teams, represented output and visible
  uncertainty. Add a static poster for readers who avoid animation.
- Existing synthetic media stay labeled as synthetic. New real-run media should
  use a distinct filename and caption, with a visible development-demo scope.

## Final evidence check

Record the source SHA-256, run ID, executed configuration, model hashes, code
fingerprints, selected source timestamps/frame indices and media hashes. Inspect
the manifest's resume provenance as well as its final status. Keep raw footage
and private artifacts outside Git; select the publication destination and
appropriate asset terms before distributing the final broadcast-derived media.

The only unfinished README assets are this real-run media selection/capture and
the corresponding completed-run measurements. The written narrative and roadmap
can be reviewed independently now.
