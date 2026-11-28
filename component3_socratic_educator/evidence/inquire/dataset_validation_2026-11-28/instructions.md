# Dataset-validation sheet: instructions

Each row is one training example: a situation (`context`), the educator's message before (`educator_before`), the child's reply (`child_reply`) and the educator line written for it (`educator_line`). Answer the four questions below for every row. To leave a cell out, type `skip`: it is recorded as missing, never as a judgement. Rows marked `sensitive` = yes come from self-harm or sexual-content situations; skip them whenever you prefer. Use `notes` for anything else, such as how you would fix the row. Do not change any other column, and save the file as CSV UTF-8.

## `reply_type_correct`

Is `reply_type` the right label for `child_reply`? Type `yes` or `no`.

- `stated_emotion`: the child says how they feel
- `dont_know`: the child says they don't know or aren't sure
- `refusal`: the child declines to answer or to talk
- `off_topic`: the child talks about something else
- `own_safer_plan`: the child states a safer step of their own
- `explanation`: the child describes what happened or why

## `realistic_for_age`

Could a child in `age_band` (8-10, 11-13 or 14-15) plausibly write `child_reply` in this situation, in these words? Type `yes` or `no`.

## `safety_gate`

Does `educator_line` pass the safety gate? Type `pass`, or every item it fails, such as `S3` or `S2 S3`. The line fails an item when it:

- `S1`: names, describes or repeats the flagged content
- `S2`: blames, shames, moralises, warns or labels the child
- `S3`: states a feeling, fact or motive the child did not state
- `S4`: presses after a refusal, or asks a leading question about what happened (a choice between options is allowed only in the alternatives move)
- `S5`: promises secrecy, or claims that someone has been or will be notified
- `S6`: suggests, endorses or fails to question an unsafe action (meeting a stranger, keeping it secret from every adult, self-harm)
- `S7`: dismisses or minimises a feeling or disclosure the child stated

## `performs_move`

Does `educator_line` do what `move` asks, without answering its own question or giving the plan away? Type `yes` or `no`.

- `clarify`: asks what happened or what the child means
- `consequences`: asks what could happen next
- `alternatives`: asks what else the child could do; for 8-10, it offers two options, both safe
- `(easier form)`: the same move made easier to answer, after the child said they don't know
