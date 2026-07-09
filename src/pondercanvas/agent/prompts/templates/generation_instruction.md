## Task
Produce the next candidate image for this brief by calling `generate_image`.

## Brief
{brief}

## Grounding context
{grounding_result?}

## Feedback from the last evaluation (if any)
{last_evaluation?}

## Optional research -- use only when you have a concrete gap to fill
Decide whether you already have enough visual and factual context to render
a great result. If the brief or the feedback above calls out something
current references/grounding don't cover -- e.g. feedback wants a "wet" cat
but existing references are all dry -- you may call, at most once each:
- `search_reference_images(query)`: real Unsplash photos for a short,
  specific query (e.g. "wet cat", not "corrections to the cat").
- `search_web(query)`: grounded text context from Google Search.

Skip both and go straight to `generate_image` when current context is
already sufficient -- do not call them by default or "just in case."

## Writing the prompt
Call `generate_image` with a `prompt` argument: one cohesive, vivid scene
description (not a checklist of attributes) that weaves together the
brief's subject, style, composition, mood, palette, and constraints, any
grounding context, and anything you just searched for. Be concrete: name
materials, textures, and surface details instead of generic nouns (e.g.
"weathered oak" rather than "wood"). Reason through the physical
consequences of the scene, not just its named subject: actions and
conditions leave visible traces the brief may not spell out (e.g. a
blacksmith at the forge implies scorch marks, drifting sparks, and a
soot-streaked apron) -- work those consequences into the description so the
image reads as physically coherent, not just literally correct. Where the
composition implies a camera framing, use precise photographic terms (e.g.
"low angle," "shallow depth of field," "wide shot"). Describe lighting
explicitly (direction, quality, mood) rather than leaving it to be inferred.

If there is feedback above, this is a revision pass: apply those
corrections directly and completely to the same scene rather than starting
over, and don't undo details that already worked.

Always call `generate_image` exactly once before finishing this turn --
do not ask the user anything.
