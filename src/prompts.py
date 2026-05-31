# -*- coding: utf-8 -*-
"""All prompts used by the VRR-QA pipeline.

These are the exact templates used to produce the released results:
  * Stage 1 : direct / category-guided / conservative answer branches
  * Stage 2 : the evidence-grounded cross-model judge (Gemini + GPT)
  * Stage 3 : the final-choose arbiter (Gemini 3.5 Flash)
plus the category metadata + micro-guidelines used by the category-guided
branch and the LLM fallback of the question classifier.
"""

# ===========================================================================
# Category metadata (the 9 VRR-QA reasoning categories)
# ===========================================================================
CATEGORY_NAMES = {
    1: "Lateral Spatial Reasoning",
    2: "Vertical Spatial Reasoning",
    3: "Relative Depth and Proximity",
    4: "Motion and Trajectory Dynamics",
    5: "Viewpoint and Visibility",
    6: "Inferred Counting",
    7: "Physical and Environmental Context",
    8: "Causal and Motivational Reasoning",
    9: "Social Interaction and Relationships",
}

CATEGORY_DESC = {
    1: ("Left / right side, which side of a table, or which lateral direction "
        "(toward / away / side-by-side / perpendicular) two entities are FACING "
        "relative to each other. Pure horizontal/side relationships, no vertical "
        "and no depth/proximity in the options."),
    2: "Above / below / on top / underneath / height comparisons.",
    3: ("In front / behind / closer / farther / same distance. Depth, proximity "
        "or front-back relationships."),
    4: ("Direction of MOTION / movement: walking, running, flying, driving, "
        "turning, moving. Usually 'in what/which direction does X move/walk/run'."),
    5: ("Camera POV / whose perspective is shown / what is (in)visible or "
        "occluded / can X see Y / where is X facing relative to a scene."),
    6: "Counting how many things appear ('How many ...').",
    7: ("Identifying WHICH physical object / vehicle / container / item is "
        "associated with an entity or event (which car, which box, which shirt "
        "colour, ordering by appearance, etc.)."),
    8: ("WHY something happens / what caused something / how a chain of events "
        "leads to an outcome."),
    9: ("Identity of a character via family/role/relationship, or how characters "
        "socially interact."),
}

# ===========================================================================
# Micro-guidelines injected by the category-guided branch (Stage 1, B3)
# ===========================================================================
SPATIAL_GUIDELINE = """[Spatial Reasoning Guideline]
When determining relative positions (left/right, above/below, front/behind):
1. First, check the viewer's perspective — i.e. what you actually see on the
   screen. For simple landmark or object-to-object comparisons, the screen-
   space relative position (left/right on screen, higher/lower on screen,
   closer/farther in depth on screen) is usually how human annotators
   labeled the answer.
2. Only when the reference is a clearly oriented character (a person /
   animal / vehicle facing a definite direction) AND the screen-space layout
   is genuinely ambiguous, fall back to that character's own left/right.
3. Keep the reasoning brief: identify the two target objects, locate them in
   the video, and report their relative position. Do NOT introduce 3D
   coordinate mappings, "intrinsic frame" gymnastics, or camera-pan
   inversions — those tend to flip the answer to the wrong side."""

MOTION_GUIDELINE = """[Motion & Trajectory Guideline]
For "in what direction does X move / turn" style questions:
1. Locate the moving object's starting position and ending position in the
   scene.
2. Describe the direction of motion relative to BACKGROUND landmarks
   (buildings, walls, trees, road, horizon) — NOT relative to the camera's
   own motion. If the camera itself pans/dollies, separate that from the
   object's true motion in the world.
3. For rotation questions (clockwise / counterclockwise), pick a consistent
   reference point on the object and track which way it sweeps."""

VIEWPOINT_GUIDELINE = """[Viewpoint & Visibility Guideline]
For "can X see Y / is Y visible to X" style questions:
1. Estimate the character X's current gaze / body / head orientation from
   the video.
2. Check whether Y falls inside X's field of view given that orientation,
   and whether anything obvious (walls, doors, X's own body) occludes Y.
3. If the question specifies a viewpoint ("from the camera's perspective",
   "from <X>'s perspective"), reason in exactly that frame.
4. Otherwise, stick to what the video actually shows; do not invent
   off-screen geometry."""

COUNTING_GUIDELINE = """[Counting Guideline]
For "how many <X>" style questions, count carefully:
1. Mentally segment the video timeline (beginning / middle / end, or by
   shot cuts).
2. Walk through each segment in order and enumerate every occurrence of the
   target item / event you can see, noting its approximate timestamp or
   ordinal (e.g. "1st at ~0:02, 2nd at ~0:05, 3rd near the end").
3. Sum the occurrences. Be careful not to double-count the same instance
   across cuts, and not to miss instances that flash by quickly."""

OBSERVATION_GUIDELINE = """[Observation Guideline]
This question depends on a specific visible detail (color / written text /
identity / ordering / item present in the scene). Verify your pick against
multiple frames in the video before answering. Prefer details you can
literally point to in the video over plausible-sounding guesses."""

CAUSAL_GUIDELINE = """[Causal & Social Reasoning Guideline]
For "why / what caused / who is responsible / who is talking to whom"
style questions:
1. Focus on the temporal sequence: what happens immediately before the
   event in question? What is the trigger?
2. Trace the chain of actions and reactions across the characters in view.
3. Watch for the small cue that makes one option clearly fit better than
   the others (a glance, a gesture, a sound, a contact between objects)."""

CATEGORY_GUIDELINES = {
    1: SPATIAL_GUIDELINE,
    2: SPATIAL_GUIDELINE,
    3: SPATIAL_GUIDELINE,
    4: MOTION_GUIDELINE,
    5: VIEWPOINT_GUIDELINE,
    6: COUNTING_GUIDELINE,
    7: OBSERVATION_GUIDELINE,
    8: CAUSAL_GUIDELINE,
    9: CAUSAL_GUIDELINE,
}

# ===========================================================================
# Stage 1 answer branches
# ===========================================================================
_BRANCH_REPLY_FORMAT = (
    "Reply in EXACTLY this format:\n"
    "ANSWER: <one of {letters}>\n"
    "REASON: <briefly describe the concrete visual evidence you saw and how "
    "it leads to your choice; a few sentences are fine, but keep it grounded "
    "in what is actually visible in the video>"
)

# B2 / direct (also the "ori" baseline) -- no guidance, just answer.
DIRECT_PROMPT = """You will watch a video clip, then answer a multiple-choice question about it.

Question: {question}

Options:
{options_block}

Watch the video carefully and choose the single best answer.

""" + _BRANCH_REPLY_FORMAT

# B3 / category-guided -- inject the matching micro-guideline.
CATEGORY_PROMPT = """You will watch a video clip, then answer a multiple-choice question about it.

Question: {question}

Options:
{options_block}

{guideline_block}Watch the video carefully and choose the single best answer.

""" + _BRANCH_REPLY_FORMAT

# B4 / conservative -- anti-hallucination, only answer what is supported.
CONSERVATIVE_PROMPT = """You will watch a video clip, then answer a multiple-choice question about it.

Answer the question conservatively.

Only answer with information that is directly supported by the video.
Do not guess.
Do not use common sense to fill in missing visual details.

Question: {question}

Options:
{options_block}

If multiple options seem partially supported, pick the one whose claims are MOST directly visible/audible in the video. If you have to choose between an option that is fully supported by the video and an option that merely "sounds plausible", choose the supported one.

""" + _BRANCH_REPLY_FORMAT

# ===========================================================================
# Stage 2 evidence-grounded judge (used by BOTH Gemini and GPT judges)
# ===========================================================================
JUDGE_PROMPT = """You are the final evidence-grounded judge for a multiple-choice video question-answering task.

Three independent experts have already watched this exact video and given their answer + reasoning. Your job is NOT to redo the QA from scratch, and NOT to take a majority vote.

Your job is to:
1. Watch the video yourself as the ground truth.
2. For EACH expert's reasoning, audit it strictly against what the video actually shows. For each major claim, label it:
   - SUPPORTED   — the claim is directly visible / audible in the video.
   - SPECULATED  — the claim is a guess, a common-sense fill-in, or relies on info not actually in the video.
   - MISSED      — the expert ignored a key visual / audio detail that would change the answer.
3. Be balanced. Do NOT blindly trust your own first impression OR any single expert.
   - Respect the experts — their reasoning is a strong prior, especially when they agree on the same observable fact.
   - But also be discriminating — if an expert sounds confident yet their cited evidence is not actually in the video, mark it SPECULATED and do NOT side with them.
   - When experts disagree, re-check the video carefully on the exact point of disagreement before deciding.
4. Pick the option whose statement is MOST DIRECTLY supported by the video. 

Question: {question}

Options:
{options_block}

--- Expert A reasoning: ---
{expert_a_text}

--- Expert B reasoning: ---
{expert_b_text}

--- Expert C reasoning: ---
{expert_c_text}

Reply in EXACTLY this structured format (keep the section headers literal):

EXPERT_A_AUDIT:
- <claim>: <SUPPORTED|SPECULATED|MISSED> — <one short sentence citing what you saw in the video>
- ...
EXPERT_B_AUDIT:
- ...
EXPERT_C_AUDIT:
- ...

CRITICAL_OBSERVATION:
- <one or two bullets citing the decisive visual / audible cue you saw in the video that picks one option over the others;>

ANSWER: <one of {letters}>
REASON: <one or two sentences citing the decisive cue from CRITICAL_OBSERVATION;>"""

# ===========================================================================
# Stage 3 final-choose arbiter (Gemini 3.5 Flash)
# ===========================================================================
FINAL_CHOOSE_PROMPT = """You are the final evidence-grounded judge for a multiple-choice video question.

Two independent analysts each watched this exact video and produced an analysis (concrete observations plus a chosen answer). Their chosen answers DIFFER. Watch the video YOURSELF as the ground truth and decide which of the two candidate answers is correct.

How to judge:
1. Watch the video carefully, focusing on the exact point where the two analysts disagree.
2. Audit each analyst's key claims against what the video actually shows: a claim is only trustworthy if it is directly visible / audible. Be suspicious of vague claims, double-counting, miscounting, or self-contradiction.
3. Pick the candidate answer whose statement is MOST DIRECTLY supported by what you actually see in the video. Respect the analysts' reasoning as a prior, but trust the video over either of them.
4. You MUST choose exactly one of the two candidate letters ({cand1} or {cand2}). Do NOT pick any other option and do NOT abstain.

Question: {question}

Options:
{options_block}

--- Analyst 1 (answer: {ans1}) ---
{reason1}

--- Analyst 2 (answer: {ans2}) ---
{reason2}

Reply in EXACTLY this format:
ANSWER: <{cand1} or {cand2}>
REASON: <one or two sentences citing the decisive visual / audible cue you saw in the video>"""

# ===========================================================================
# Question classifier LLM fallback (Stage 0)
# ===========================================================================
def build_classifier_system_prompt() -> str:
    lines = [
        "You are a strict question classifier for the VRR-QA Video QA benchmark.",
        "Classify each multiple-choice question into EXACTLY ONE of these 9 categories:",
        "",
    ]
    for cid in range(1, 10):
        lines.append(f"{cid}. {CATEGORY_NAMES[cid]} — {CATEGORY_DESC[cid]}")
    lines += [
        "",
        "IMPORTANT decision rules — apply in order, stop at the first match:",
        "(R1) If the question starts with 'How many' => 6.",
        "(R2) If the question asks 'Why ...' or about what caused something => 8.",
        "(R3) If the question asks about identity / relationship of a character "
        "('Who is X', 'relationship between') => 9.",
        "(R4) If the question asks about visibility, what is visible/occluded, "
        "camera POV, perspective, or pointing => 5.",
        "(R5) READ THE OPTIONS:",
        "       - options contain 'above / below / on top / underneath / same height' => 2.",
        "       - options contain 'in front / behind / closer / farther / same distance / "
        "directly toward / directly away' => 3.",
        "       - options contain 'adjacent side / opposite side / same side of the table / "
        "to the left / to the right / side by side / perpendicular' => 1.",
        "(R6) 'In what/which direction does X (walk|run|move|drive|fly|turn|spin)' => 4.",
        "(R7) 'Which car / box / character / animal / shirt / color' identification, "
        "or ordering by appearance => 7.",
        "(R8) If two entities are 'facing relative to each other' with no clear motion verb => 1.",
        "",
        "Answer with ONLY a single integer from 1 to 9. No words, no punctuation, no explanation.",
    ]
    return "\n".join(lines)


def build_classifier_user_prompt(question_text: str, options: dict) -> str:
    opt_lines = [f"  {k}. {v}" for k, v in options.items()]
    return (
        "Question: " + question_text.strip() + "\n"
        "Options:\n" + "\n".join(opt_lines) + "\n\n"
        "Read the options FIRST, then apply the rules. Output the single digit."
    )
