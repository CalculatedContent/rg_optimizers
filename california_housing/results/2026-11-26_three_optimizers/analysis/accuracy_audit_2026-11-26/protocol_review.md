# California Housing protocol accuracy review

This read-only audit reviewed the frozen source and the 15 primary result records. It did not train a model, retune a setting, or alter the experiment source or results.

## Conclusion

No implementation error was found in the split, normalization, training objective, evaluation units, clipping order, or magnitude-graft arithmetic. This conclusion concerns the implemented fixed experiment; it does not establish globally optimal hyperparameters, a causal graft-only comparison, or the statistical validity of an asymptotic power-law interpretation.

## Checks performed

- All 19 files in `SOURCE_SHA256.json` pass `python -m california_housing.verify`.
- All 15 primary result records report source hashes matching those frozen files.
- The split includes all 20,640 dataset rows exactly once, with no intersection among training, validation, and test sets. Its concatenated indices exactly match `numpy.random.default_rng(20261126).permutation(20640)`.
- Recomputed training-only feature and target means and population standard deviations exactly match the prepared archive.
- Reconstructed all three splits from the cached raw dataset: standardized float32 features, standardized float32 targets, and original float64 targets all match exactly (maximum absolute discrepancies 0).
- All five seed triplets have matching initialization hashes and minibatch-order hashes across optimizers.
- The abstract syntax tree of `MagnitudeGraft.step` is identical to the existing Shakespeare implementation. The parameter selector changes to the five MLP weights as declared.

Machine-readable evidence: `protocol_data_check.json` and `protocol_run_check.json`.

## Source review

`data.py` computes both input and target scalers from training rows only. It stores original targets separately, so evaluation does not reconstruct truth from rounded standardized targets.

`train.py` uses a scalar prediction per row and MSE against the corresponding standardized scalar target. Evaluation converts float32 model outputs to float64, reverses target scaling, and calculates RMSE and MAE against the saved original target. Consequently RMSE 0.5 is 0.5 units of $100,000, approximately $50,000; it is not a percentage error.

The trainer clips the global gradient norm before each optimizer update. It uses the same sampled minibatch sequence within each seed. The 2,000-update checkpoint is the prefix of the declared 20,000-update schedule, including the same 100-update warmup and cosine horizon.

`calibrate.py` selects the candidate with the lowest full validation RMSE at update 2,000. Its calibration path neither evaluates the test set nor fits alpha. Frozen settings are written before confirmation. Training records use the selected configurations consistently.

`optimizer.py` retains Adam's coordinatewise direction and moment histories, rescaling each of five matrix updates to the declared same-history momentum norm. Biases retain native AdamW updates. The momentum reference is computed from this run's clipped gradient history; it is not borrowed from the separately trained SGDM control.

## Interpretation limits

1. The comparison uses separately calibrated recipes. Native AdamW uses peak LR 0.003. Modified AdamW uses base peak LR 0.001 and momentum-reference peak LR 0.03. SGDM uses peak LR 0.003. The SGDM control therefore does not use the modified optimizer's reference rate, and the AdamW arms do not share an identical base rate.
2. The calibration uses one development seed, three candidates per optimizer, and only the early 2,000-update validation endpoint. It does not establish the best 20,000-update settings. Selected SGDM is at the bottom of its rate grid, while native and modified AdamW are at the top of theirs; the fixed grid was deliberately not expanded after confirmation.
3. Weight decay is zero in all arms. Native AdamW therefore has no active decoupled weight-decay term in this experiment. There is also no dropout or early-stopping selection; late held-out deterioration can be ordinary overfitting rather than a numerical failure.
4. Seed confidence intervals are conditional on the fixed split and fixed recipe. They measure training-seed variation, not uncertainty from resampling the housing dataset or selecting a different geography.
5. The data represent census block-group median housing values. Dollar-valued errors refer to that dataset's target and units; they are not modern individual-house appraisal errors.
6. Alpha near 2 is a spectral hypothesis target, not a definition of successful price prediction. Predictive performance and spectral behavior must be evaluated separately.
