# V8 development log

Append-only. V6 and V7 are preserved. V7 is an exploratory negative result and
is not evidence of confirmation success.

## Preregistration phase

- Baseline verified: branch `qiskit`, clean commit
  `f892ee321be7d40460228303a820ae70046fa3e8`.
- Implemented the complete executable gate engine and multiplicity-corrected
  paired bootstrap before architecture execution.
- Power analysis froze 16 development, 16 internal-validation and 24 encrypted
  confirmation seed triples.
- Confirmation seeds are committed as a tracked Fernet ciphertext. The key is
  outside the repository; clean clones can verify the commitment without being
  able to reveal the bank.
- No V8 quantum circuit has been executed and no V8 scientific result exists at
  this point.
