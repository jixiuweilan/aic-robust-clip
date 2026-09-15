# Local Competition Data

Competition data is intentionally ignored by Git and must remain private.

Use isolated stage directories:

```text
data/
  preliminary/
    train/
    test/
  second_round/
    train/
    test/
  semifinal/
    train/
    test/
```

Do not mix data across stages. In particular, preliminary data cannot be used
during the second round, and neither preliminary nor second-round data can be used
during the semifinal. Test images are prediction-only and must never enter
training, pseudo-labeling, self-supervised learning, or model selection.

Before training code is added, create a read-only manifest with relative path,
byte size, and SHA-256 for each organizer-provided file. Keep split generation
deterministic and restricted to the current stage's training data.
