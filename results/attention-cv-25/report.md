# Quantum attention 25-epoch cross-validation

This report compares fixed QCSE and learned classical input encodings feeding the same causal quantum attention model. The corpus uses both `phrases` and `cleaned`, balanced sentence sampling, and a shared vocabulary.

The run used 5,000 sentences, 37,867 token examples, 10,864 vocabulary words, 25 epochs per fit, 5,000 replacement draws per epoch, and a 30,331/7,536 development/test example split. Each setup used two independent 20% validation shuffle splits inside development, then a fresh refit on all development examples before one held-out test score.

![Validation curves](validation-curves.png)

![Test comparison](test-comparison.png)

| Setup | CV validation CE mean ± SD | CV cosine top-1 | CV cosine top-5 | Test CE | Test cosine top-1 | Test cosine top-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qcse-1layer | 5.6910 ± 0.0311 | 4.689% | 18.576% | 5.7268 | 4.352% | 18.073% |
| classical-1layer | 4.6028 ± 0.0235 | 11.138% | 31.242% | 4.6084 | 9.939% | 30.480% |
| qcse-2layer | 5.6932 ± 0.0578 | 4.185% | 17.362% | 5.7180 | 4.114% | 15.446% |
| classical-2layer | 4.6085 ± 0.0326 | 11.024% | 29.891% | 4.6209 | 9.740% | 31.303% |

The lowest mean validation cross-entropy was `classical-1layer`. Superiority should be judged from the repeated validation mean and spread; the single held-out test score is reported for confirmation and was not used to select a setup.

The experiment is a statevector simulation of the paper's bilinear attention adaptation. It uses real overlaps, causal masking and normalized postselection; it does not claim finite-shot hardware performance or an efficient block-encoding oracle. The local run used a balanced 5,000-sentence cap so all 1,000 phrase rows were retained alongside cleaned sentences. Omit `--max-sentences` on a GPU allocation to use the full curated pool.

Generated files: `results.json`, per-fold histories/checkpoints, `validation-curves.png`, and `test-comparison.png`.
