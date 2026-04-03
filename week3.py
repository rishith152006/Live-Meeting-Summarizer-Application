# ============================================================
# WEEK 3 - MASTER RUNNER: Run All Tasks in Sequence
# ============================================================
# Run this file to execute the full Week 3 pipeline:
#   Task 1: Download + Mix Audio
#   Task 2: Speaker Diarization
#   Task 3: Speech-to-Text
#   Task 4: Merge → Speaker Transcript
#   Task 5: DER Accuracy Report
#
# SETUP (run once):
#   pip install pyannote.audio openai-whisper torch torchaudio pyannote.metrics
#
# Set your HuggingFace token below before running!
# ============================================================

import os
import json
import subprocess
import torch
import torchaudio
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm

# ============================================================
# CONFIGURATION - Edit these
# ============================================================

SESSION = "ES2016a"
HF_TOKEN = "your_huggingface_token_here"   # 👈 Replace with your HF token
AUDIO_DIR = f"amicorpus/{SESSION}/audio"
OUTPUT_DIR = "week3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# ============================================================
# TASK 1: Download + Mix Audio
# ============================================================
print("\n" + "="*60)
print("TASK 1: Downloading and Mixing Audio Files")
print("="*60)

BASE_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus"
headset_files = [f"{SESSION}.Headset-{i}.wav" for i in range(4)]

for filename in headset_files:
    out_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(out_path):
        url = f"{BASE_URL}/{SESSION}/audio/{filename}"
        subprocess.run(["wget", "-P", AUDIO_DIR, url], check=True)
        print(f"Downloaded: {filename}")
    else:
        print(f"Already exists: {filename}")

# Mix headsets
waveforms, sample_rate = [], None
for i in range(4):
    path = os.path.join(AUDIO_DIR, f"{SESSION}.Headset-{i}.wav")
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveforms.append(waveform)
    sample_rate = sr

max_len = max(w.shape[1] for w in waveforms)
padded = [torch.nn.functional.pad(w, (0, max_len - w.shape[1])) for w in waveforms]
mixed = torch.stack(padded).mean(dim=0)
mixed = mixed / mixed.abs().max()

MIXED_PATH = os.path.join(AUDIO_DIR, f"{SESSION}_mixed.wav")
torchaudio.save(MIXED_PATH, mixed, sample_rate)
print(f"Mixed audio saved: {MIXED_PATH}")

# ============================================================
# TASK 2: Speaker Diarization
# ============================================================
print("\n" + "="*60)
print("TASK 2: Running Speaker Diarization")
print("="*60)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = pipeline.to(device)
print(f"Using device: {device}")

diarization = pipeline(MIXED_PATH, num_speakers=4)

segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    segments.append({
        "speaker": speaker,
        "start": round(turn.start, 3),
        "end": round(turn.end, 3),
        "duration": round(turn.end - turn.start, 3)
    })

segments_path = os.path.join(OUTPUT_DIR, "diarization_segments.json")
with open(segments_path, "w") as f:
    json.dump(segments, f, indent=2)

RTTM_PATH = os.path.join(OUTPUT_DIR, f"{SESSION}.rttm")
with open(RTTM_PATH, "w") as f:
    diarization.write_rttm(f)

print(f"Diarization complete: {len(segments)} segments saved")

# ============================================================
# TASK 3: Speech-to-Text
# ============================================================
print("\n" + "="*60)
print("TASK 3: Running Speech-to-Text (Whisper)")
print("="*60)

model = whisper.load_model("medium")
result = model.transcribe(MIXED_PATH, word_timestamps=True, language="en", verbose=False)

stt_segments = []
for seg in result["segments"]:
    stt_segments.append({
        "start": round(seg["start"], 3),
        "end": round(seg["end"], 3),
        "text": seg["text"].strip()
    })

stt_path = os.path.join(OUTPUT_DIR, "stt_segments.json")
with open(stt_path, "w") as f:
    json.dump(stt_segments, f, indent=2)

print(f"Transcription complete: {len(stt_segments)} segments saved")

# ============================================================
# TASK 4: Merge Diarization + STT → Speaker Transcript
# ============================================================
print("\n" + "="*60)
print("TASK 4: Merging Diarization with STT")
print("="*60)

def get_overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))

merged = []
for stt in stt_segments:
    best_speaker, best_overlap = "Unknown", 0
    for diar in segments:
        overlap = get_overlap(stt["start"], stt["end"], diar["start"], diar["end"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = diar["speaker"]
    merged.append({"speaker": best_speaker, "start": stt["start"], "end": stt["end"], "text": stt["text"]})

# Group by consecutive speaker
transcript_lines = []
current_speaker, current_text, current_start = None, [], None
for seg in merged:
    if seg["speaker"] != current_speaker:
        if current_speaker:
            transcript_lines.append({"speaker": current_speaker, "start": current_start, "text": " ".join(current_text).strip()})
        current_speaker, current_text, current_start = seg["speaker"], [seg["text"]], seg["start"]
    else:
        current_text.append(seg["text"])
if current_speaker:
    transcript_lines.append({"speaker": current_speaker, "start": current_start, "text": " ".join(current_text).strip()})

TRANSCRIPT_PATH = os.path.join(OUTPUT_DIR, "speaker_transcript.txt")
print("\n--- SPEAKER TRANSCRIPT PREVIEW ---")
with open(TRANSCRIPT_PATH, "w") as f:
    f.write("SPEAKER-WISE TRANSCRIPT\n" + "="*60 + "\n\n")
    for entry in transcript_lines:
        spk_num = int(entry["speaker"].split("_")[-1]) + 1
        mm = int(entry["start"] // 60)
        ss = int(entry["start"] % 60)
        line = f"[{mm:02d}:{ss:02d}] Speaker {spk_num}: {entry['text']}"
        print(line)
        f.write(line + "\n")

print(f"\nTranscript saved: {TRANSCRIPT_PATH}")

# ============================================================
# TASK 5: DER Accuracy Report
# ============================================================
print("\n" + "="*60)
print("TASK 5: Generating DER Accuracy Report")
print("="*60)

REF_RTTM = f"amicorpus/{SESSION}/reference/{SESSION}.rttm"

if os.path.exists(REF_RTTM):
    ref_loaded = load_rttm(REF_RTTM)
    hyp_loaded = load_rttm(RTTM_PATH)
    reference = list(ref_loaded.values())[0]
    hypothesis = list(hyp_loaded.values())[0]

    metric = DiarizationErrorRate()
    der_detail = metric(reference, hypothesis, detailed=True)
    total_der = abs(metric)
    missed = der_detail["missed detection"] / der_detail["total"] * 100
    false_alarm = der_detail["false alarm"] / der_detail["total"] * 100
    confusion = der_detail["confusion"] / der_detail["total"] * 100

    report = f"""
========================================================
       WEEK 3 - DIARIZATION ACCURACY REPORT
========================================================
Session    : {SESSION}
Model      : pyannote/speaker-diarization-3.1
Target DER : < 20%
========================================================
Missed Detection  : {missed:.2f}%
False Alarm       : {false_alarm:.2f}%
Speaker Confusion : {confusion:.2f}%
----------------------------------------
TOTAL DER         : {total_der:.2f}%
RESULT: {"✅ PASS - DER below 20%!" if total_der < 20 else "❌ FAIL - DER exceeds 20%. Try tuning."}
========================================================
"""
else:
    report = """
Reference RTTM not found. Download it with:
wget -P amicorpus/ES2016a/reference \\
  https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/ES2016a/annotations/ES2016a.rttm
Then re-run Task 5.
"""

print(report)
REPORT_PATH = os.path.join(OUTPUT_DIR, "accuracy_report.txt")
with open(REPORT_PATH, "w") as f:
    f.write(report)
print(f"Report saved: {REPORT_PATH}")

print("\n" + "="*60)
print("✅ ALL WEEK 3 TASKS COMPLETE!")
print(f"Outputs saved in: {OUTPUT_DIR}/")
print("="*60)