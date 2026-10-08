# Match precision (manual spot-check)

Hand-labelled sample of name matches: `y` = same person/company, `n` = different, `?` = cannot tell. Precision = y / (y + n); `?` and blank rows are excluded.

| Tier | List | Sampled | y | n | ? | Unlabelled | Precision |
|---|---|---|---|---|---|---|---|
| weak | all | 50 | 0 | 0 | 0 | 50 | n/a |
| weak | pep | 26 | 0 | 0 | 0 | 26 | n/a |
| weak | sanctions | 24 | 0 | 0 | 0 | 24 | n/a |
| strong | all | 50 | 0 | 0 | 0 | 50 | n/a |
| strong | pep | 44 | 0 | 0 | 0 | 44 | n/a |
| strong | sanctions | 6 | 0 | 0 | 0 | 6 | n/a |

**Strong matches were correct 0 out of 0 judged times (n/a).**

Warning: 50 strong rows are still unlabelled.

## Notes

- Strong rows are a census (all strong matches are labelled) unless there were more than the sample cap, so strong precision has no sampling error.
- Weak sample sizes are small; treat weak precision as indicative only.
- The labels were judged against Companies House birth month/year and the list's birth date where available; `?` is used when neither side gives enough evidence.
- If strong precision is below ~70%, tighten the strong rule before freezing labels.
