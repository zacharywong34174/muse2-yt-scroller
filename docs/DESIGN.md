# Design notes

The prototype treats a deliberate blink as a classification event. The raw EEG
is intentionally not cleaned of ocular artifacts: those artifacts carry the
control input. Each 100-sample window becomes 20 scalar time-domain features.
The same extraction function runs during both training and inference.

## Timing

At 256 Hz a 100-sample window spans approximately 391 ms; a 20-sample slide is
approximately 78 ms. The implementation primes 100 samples, advances 20, then
makes its first prediction, matching the original loop. Four non-neutral
predictions and two neutral observations add time before an action. Processing,
Bluetooth/LSL buffering, and OS handling also contribute. These values describe
algorithm settings, not measured response latency.

## The event rule

An `NB` before the fourth blink-labelled prediction cancels the episode. From
the fourth blink prediction onward, the strongest `LB` or `RB` probability is
retained. After two `NB` observations, emit at most one key if that probability
reaches 0.5, then reset. An intervening blink prediction does not reset the neutral
counter. Ties retain the first maximum, with LB first within a prediction.

The original code stacked all retained probability rows and took their global
maximum. A running maximum yields the same choice while bounding memory if an
episode continues indefinitely. Probability columns now follow the model's
`classes_` instead of assuming a fixed column index for NB.

This heuristic can reject short intentional blinks or accumulate intermittent
predictions. It is retained for behavioral continuity, not presented as an optimal
or clinically validated detector. A future comparison should measure false actions
per minute, missed intentional blinks, and latency on held-out sessions.

## Channel and sampling assumptions

The original code assigned the first four stream samples to AF7, AF8, TP9, TP10.
The MuseLSL source instead lists TP9, AF7, AF8, TP10 before its auxiliary channel
([upstream constants](https://github.com/alexandrebarachant/muse-lsl/blob/master/muselsl/constants.py)).
The new code uses LSL labels, or an explicit full `--channel-order` if metadata is
absent. It rejects missing or duplicate electrode names. This is deliberately
safer than silently guessing the source's ordering.

The original plotting recorder consumed four samples per update but retained
only the last. Its initial buffer was collected differently, while the live
classifier used every sample. Thus the old recording could mix rates, and its
model cannot be assumed compatible with this release. The new recorder retains
all received samples; a saved model carries its declared training rate, which
must match the live stream. Recollect and retrain for this release. No original
model or raw recording is automatically converted or overwritten.

## Boundaries

The core demo and feature/decision code have no Bluetooth, LSL, browser, or
keyboard dependency. Hardware imports occur only in hardware functions.
Keyboard output is disabled unless explicitly requested. Model files and all
recordings are local artifacts, ignored by Git. The application neither records
browser content nor signs into a website.
