| Model | Test NLL | Perplexity | Top-1 | Top-5 | Known-target NLL |
|---|---:|---:|---:|---:|---:|
| Classical embedding (full training) | 2.4388 ± 0.0052 | 11.46 | 41.25% | 72.26% | 2.5291 |
| Bigram + learned unigram mixture | 2.9662 ± 0.0000 | 19.42 | 31.63% | 60.95% | 3.0920 |
| Classical calibrated bigram | 2.9662 ± 0.0000 | 19.42 | 31.63% | 60.95% | 3.0920 |
| Predictive state + quantum walk | 2.9662 ± 0.0003 | 19.42 | 31.70% | 60.87% | 3.0894 |
| Classical bigram | 2.9679 | 19.45 | 31.63% | 60.94% | 3.0939 |
| Recency-pooled bigram | 3.1263 | 22.79 | 32.39% | 58.36% | 3.2631 |
| Classical directed diffusion | 3.4615 ± 0.0001 | 31.86 | 26.83% | 52.44% | 3.6145 |
| Classical embedding (8k training) | 3.6039 ± 0.0182 | 36.74 | 30.22% | 54.47% | 3.7462 |
| Classical undirected diffusion | 4.0350 ± 0.0001 | 56.54 | 17.26% | 41.37% | 4.2155 |
| Graph quantum walk | 4.2964 ± 0.0008 | 73.44 | 8.72% | 38.02% | 4.4271 |
| Unigram | 4.7583 | 116.55 | 9.18% | 28.99% | 5.0037 |
| Graph labels scrambled (refitted) | 4.7595 ± 0.0000 | 116.69 | 9.18% | 28.99% | 5.0044 |
| Graph removed (refitted) | 4.7607 ± 0.0000 | 116.83 | 9.18% | 28.98% | 5.0064 |
| QCSE encoding, categorical loss | 5.1826 ± 0.0018 | 178.15 | 7.47% | 22.18% | 5.2888 |
| QCSE encoding, bit loss | 5.2759 ± 0.0014 | 195.57 | 5.93% | 18.36% | 5.3674 |
| QCSE encoding, joint Born readout | 5.3826 ± 0.0182 | 217.62 | 7.73% | 21.76% | 5.5318 |
