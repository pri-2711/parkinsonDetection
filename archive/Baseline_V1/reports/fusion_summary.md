# Late-fusion configuration

The modality probabilities are combined by weighted decision-level fusion:

`P_final = w_speech * P_speech + w_handwriting * P_handwriting + w_tremor * P_tremor`

Weights are each modality's nested group-aware OOF ensemble ROC-AUC divided by the sum of the three ROC-AUC values. The default classification threshold is 0.5. Missing modalities are omitted and the remaining weights are renormalized by `src.fusion.fuse_probabilities`.

| Modality | OOF ROC-AUC | Raw weight | Normalized weight |
|---|---:|---:|---:|
| speech | 0.835347 | 0.835347 | 0.335332 |
| handwriting | 0.805257 | 0.805257 | 0.323253 |
| tremor | 0.850500 | 0.850500 | 0.341415 |

These cohorts are unmatched. This configuration does not create patient matches and does not establish fused patient-level accuracy, F1, or ROC-AUC. Telemonitoring motor-UPDRS regression remains separate.
