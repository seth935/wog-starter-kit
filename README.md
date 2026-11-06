# Weights of Giants — Season Zero Starter Kit
### The 1-Watt Edge Challenge: Fixed Perimeter & Local NVR Security

Welcome to **Season Zero** of the [Weights of Giants (WOG)](https://weightsofgiants.com) micro-vision benchmarking arena.

This template provides everything you need to package, test, and submit your oriented bounding box (OBB) detection and tracking models for evaluation on constrained ARM silicon.

---

## 1. Challenge Parameters & Limits

* **Target Hardware:** Reference ARM Cortex-A53 quad-core silicon (`cortex-a53-512mb`).
* **Power & Compute Envelope:** $\sim 1\text{W}$ power draw, 512MB RAM cap.
* **Model Weight Limit:** $\le 5\text{MB}$ quantized INT8 weights bundle (ONNX / TFLite / NCNN).
* **Speed Floor:** $\ge 5.0\text{ FPS}$ required to place; **$15.0\text{ FPS}$ target** for full speed credit.
* **Tracking Floor:** $\text{HOTA} \ge 40.0\%$ to place; $\text{HOTA} \ge 50.0\%$ for season prize payouts.
* **Orientation:** Oriented Bounding Boxes (OBBs) with angle $\theta$ in radians: `[cx, cy, w, h, theta, conf, cls]`.

---

## 2. Repository Structure

```
├── model.json            # Declarative execution manifest (runtime, thresholds, stages)
├── models/
│   └── detector_int8.onnx # Quantized model weights
├── verify_local.py       # Pre-submission verification script
└── README.md
```

### The Declarative Manifest (`model.json`)
The arena does not execute submitter scripts. The target referee node parses `model.json` to configure the pre-installed runtime:

```json
{
  "runtime": "onnxruntime",
  "threads": 4,
  "domains": ["indoor", "urban", "rural"],
  "frame_policy": {
    "skip": 0,
    "resize": "letterbox"
  },
  "stages": [
    {
      "type": "detector",
      "model": "models/detector_int8.onnx",
      "input_size": [512, 512],
      "conf_threshold": 0.35,
      "iou_threshold": 0.45
    },
    {
      "type": "tracker",
      "algo": "bytetrack-lite",
      "params": {
        "track_thresh": 0.4,
        "track_buffer": 30,
        "match_thresh": 0.8
      }
    }
  ]
}
```

---

## 3. Local Verification Before Submission

Run the verifier on your local machine:

```bash
pip install numpy onnxruntime
python verify_local.py
```

This verifies:
1. Manifest syntax and caps against the 1-Watt Edge profile.
2. Absence of unsafe deserialization artifacts (`.pkl`, `.pt`, `.joblib`).
3. Weight file size ($\le 5\text{MB}$).
4. Real 30-frame inference run with latency, FPS, and OBB tensor output validation.

---

## 4. Submitting Your Model

1. Fork or clone this repository to your personal GitHub or Hugging Face account.
2. Train, quantize, and export your model into `models/`.
3. Update `model.json` with your thresholds and input resolution.
4. Run `python verify_local.py` to confirm the model clears all checks.
5. Go to [https://weightsofgiants.com](https://weightsofgiants.com), connect your GitHub or Hugging Face account, and click **Submit a Model**.
6. The pinned commit will be screened in an automated ARM sandbox and queued for physical refereeing on the target hardware.

---

## 5. Community & Support

* **Discord Server:** [https://discord.gg/weightsofgiants](https://discord.gg/weightsofgiants) — Real-time chat, referee rig updates, hardware banter, and model optimization advice.
* **GitHub Discussions:** [GitHub Discussions](https://github.com/weights-of-giants) — Code questions, bug reports, and starter-kit help.
* **Guide & FAQ:** [https://weightsofgiants.com/info](https://weightsofgiants.com/info) — Full scoring rules and hardware specifications.

