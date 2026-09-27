# Traffic event detection and accident anticipation — Team submission

## How to install and run

```bash
python -m venv venv
# Windows: venv\Scripts\activate      Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

Model weights are committed directly in `weights/yolov8s.pt` (~22 MB, well
under the 5 GB limit) — no download step or internet access is needed at
evaluation time.

## Approach

**Pipeline:** fixed-camera video → OpenCV frame decode (every 3rd frame,
resized to 1920px wide) → YOLOv8s detector → ByteTrack tracker → cached
per-video track table (`cache/<video>_<size>.csv`) → hand-written rules on
hand-drawn zones (`zones.json`: road polygon, crosswalks, stop line) → event
segments.

**What is learned:** object detection (YOLOv8s, pretrained on COCO, from
[Ultralytics](https://github.com/ultralytics/ultralytics), AGPL-3.0 licence)
and multi-object tracking (ByteTrack, MIT licence, bundled with Ultralytics).
No custom training was done; the pretrained COCO weights already detect the
relevant classes (person, car, bus, truck, motorcycle) reliably on this
camera's footage.

**What is rule-based:** every event class is a hand-written rule over the
detector+tracker output and the zone geometry (see `src/rules_*.py`):

- `stopped_vehicle`: a vehicle stationary on the road for ≥10s, excluding
  vehicles that spend most of that time near a queue of other standing
  vehicles (a signal queue), via a nearby-standing-vehicle count.
- `jaywalking`: a pedestrian's foot position on the road, outside every
  crosswalk polygon (with margin), for ≥1.5s, filtered to require real
  movement (rejects static false detections on poles/signs).
- `failure_to_yield`: a vehicle overlapping a crosswalk polygon while a
  pedestrian is also on it, filtered to require the vehicle actually moved a
  meaningful distance during the overlap (excludes vehicles simply queued
  near the crossing waiting at a red light).
- `congestion`: the share of on-road tracked vehicles that are simultaneously
  stationary, sustained for ≥15s.

Classes not implemented in this submission (`accident`, `near_miss`,
`red_light`, `wrong_way`, `illegal_u_turn`, `illegal_turn`,
`solid_line_crossing`, `stop_line`, `road_obstacle`, `fire_smoke`) were left
out under time constraints rather than shipped as unreliable rules that would
add false positives; `CLASSES` in `solution.py` only lists what we predict,
per the FAQ ("only by removing classes you never predict").

**Part B (RiskEstimator):** a causal, frame-by-frame heuristic
(`src/risk.py`). It runs its own lightweight detector+tracker (stride-4
frames, small input size, for speed) and combines three signals into a
smoothed [0,1] score: (1) time-to-collision between converging vehicle pairs
on the road, (2) sudden deceleration of a single vehicle, (3) a pedestrian on
the road close to a moving vehicle. No trained accident classifier was built;
this matches the task's own suggestion that "good anticipation signals do
not need a trained accident model."

**Zones:** hand-drawn once on a frame from the fixed camera
(`draw_zones.py`) and reused for every video, since the camera angle is
constant across all sample and test videos.

## Datasets and licences

- No external training datasets were used (pretrained COCO weights only).
- Ultralytics YOLOv8: AGPL-3.0 (https://github.com/ultralytics/ultralytics)
- ByteTrack (via Ultralytics `bytetrack.yaml`): MIT

## Determinism

`random`, `numpy`, and `torch` seeds are fixed to 0 in `solution.py` and
`src/tracking.py`. Detection/tracking runs on CPU or GPU depending on
availability (Ultralytics auto-selects); results may vary by a small amount
of floating-point noise across hardware, as allowed by the rules.

## Team

- [Name 1] — Part A pipeline, tracking, rules, Part B risk estimator
- [Name 2] — labeling / dev set, evaluation, zone drawing
- [Name 3] — website, EDA, live demo

## Known limitations / honest failure cases

- Rules are tuned against manual spot-checks on our own sample video(s), not
  a full independent label set for every class, given time constraints.
- The signal light colour (for `red_light`/`stop_line`) was not reliably
  readable at dusk in our sample footage, so those classes were left out
  rather than shipped as guesses.
- `stopped_vehicle`/`congestion` thresholds (queue-neighbor distance,
  "alone" duration) were tuned on one camera's traffic density and may need
  adjustment for a busier or sparser scene in the hidden test set.
