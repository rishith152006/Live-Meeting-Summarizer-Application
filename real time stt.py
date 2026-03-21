#!/usr/bin/env python3
"""
Real-Time Speech-to-Text using OpenAI Whisper
Week 2 Project: Live transcription from microphone with terminal display
"""

import os
import sys
import time
import queue
import threading
import wave
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Dependency check ─────────────────────────────────────────────────────────
try:
    import whisper
except ImportError:
    print("❌  whisper not found. Run:  pip install openai-whisper")
    sys.exit(1)

try:
    import sounddevice as sd
except ImportError:
    print("❌  sounddevice not found. Run:  pip install sounddevice")
    sys.exit(1)

try:
    import scipy.io.wavfile as wav_io
    from scipy.signal import resample
except ImportError:
    print("❌  scipy not found. Run:  pip install scipy")
    sys.exit(1)


# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000   # Whisper expects 16 kHz
CHANNELS           = 1
CHUNK_SECONDS      = 3       # Seconds of audio per transcription chunk
OVERLAP_SECONDS    = 0.5     # Overlap between chunks to avoid cut-off words
SILENCE_THRESHOLD  = 0.01    # RMS below this = silence (0.0–1.0)
MODEL_SIZE         = "base"  # tiny | base | small | medium | large
LOG_DIR            = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


# ── Colors for terminal ───────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"


# ── Transcription Log ─────────────────────────────────────────────────────────
class TranscriptionLog:
    def __init__(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.txt_path  = LOG_DIR / f"transcription_{ts}.txt"
        self.json_path = LOG_DIR / f"transcription_{ts}.json"
        self.entries   = []
        self.session_start = datetime.now()

    def add(self, text: str, confidence: float, latency_ms: float):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "confidence": round(confidence, 3),
            "latency_ms": round(latency_ms, 1),
        }
        self.entries.append(entry)

        # Append to plain-text log
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(f"[{entry['timestamp']}] {text}\n")

        # Rewrite JSON log
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump({
                "session_start": self.session_start.isoformat(),
                "model": MODEL_SIZE,
                "sample_rate": SAMPLE_RATE,
                "entries": self.entries,
            }, f, indent=2)

    def full_transcript(self) -> str:
        return " ".join(e["text"] for e in self.entries)


# ── Audio Buffer ──────────────────────────────────────────────────────────────
class AudioBuffer:
    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()

    def callback(self, indata, frames, time_info, status):
        """Called by sounddevice on every audio chunk."""
        if status:
            pass  # ignore overflow/underflow silently in live mode
        self.q.put(indata[:, 0].copy())

    def read_chunk(self, n_samples: int, overlap: int) -> np.ndarray | None:
        """Drain the queue, return a chunk when enough samples are ready."""
        while not self.q.empty():
            data = self.q.get_nowait()
            with self._lock:
                self._buffer = np.concatenate([self._buffer, data])

        with self._lock:
            if len(self._buffer) >= n_samples:
                chunk = self._buffer[:n_samples].copy()
                # Keep overlap for next chunk
                self._buffer = self._buffer[n_samples - overlap:]
                return chunk
        return None

    def rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio ** 2)))


# ── Real-Time STT Engine ──────────────────────────────────────────────────────
class RealtimeSTT:
    def __init__(self, model_size: str = MODEL_SIZE):
        self.model_size    = model_size
        self.model         = None
        self.audio_buf     = AudioBuffer()
        self.log           = TranscriptionLog()
        self.running       = False
        self.chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
        self.overlap_samp  = int(OVERLAP_SECONDS * SAMPLE_RATE)
        self.total_chunks  = 0
        self.total_words   = 0
        self.latencies     = []

    # ── Load model ────────────────────────────────────────────────────────────
    def load_model(self):
        print(f"\n{C.CYAN}{C.BOLD}⏳  Loading Whisper '{self.model_size}' model…{C.RESET}")
        t0 = time.time()
        self.model = whisper.load_model(self.model_size)
        elapsed = time.time() - t0
        print(f"{C.GREEN}✅  Model loaded in {elapsed:.1f}s{C.RESET}\n")

    # ── Transcribe one audio chunk ─────────────────────────────────────────
    def transcribe_chunk(self, audio: np.ndarray) -> tuple[str, float]:
        """Return (text, no_speech_prob)."""
        # Whisper wants float32 normalised to [-1, 1]
        audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        result = self.model.transcribe(
            audio,
            language="en",
            fp16=False,          # CPU-safe
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            temperature=0.0,     # Deterministic
        )
        text = result["text"].strip()
        # Average no-speech probability across segments as a "silence score"
        segs = result.get("segments", [])
        no_speech = np.mean([s.get("no_speech_prob", 0.0) for s in segs]) if segs else 0.5
        return text, no_speech

    # ── Main transcription loop ────────────────────────────────────────────
    def run(self):
        self.load_model()
        self._print_banner()

        self.running = True
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.1),   # 100ms blocks
            callback=self.audio_buf.callback,
        )

        try:
            with stream:
                print(f"{C.GREEN}{C.BOLD}🎙  Listening… (Ctrl+C to stop){C.RESET}\n")
                print(f"{'─' * 70}")

                while self.running:
                    chunk = self.audio_buf.read_chunk(
                        self.chunk_samples, self.overlap_samp
                    )

                    if chunk is None:
                        time.sleep(0.05)
                        continue

                    # Skip silent chunks
                    rms = self.audio_buf.rms(chunk)
                    if rms < SILENCE_THRESHOLD:
                        self._print_silence()
                        continue

                    # Transcribe
                    t0 = time.time()
                    text, no_speech = self.transcribe_chunk(chunk)
                    latency_ms = (time.time() - t0) * 1000

                    # Skip empty / no-speech results
                    if not text or no_speech > 0.8:
                        self._print_silence()
                        continue

                    # Log & display
                    confidence = 1.0 - no_speech
                    self.log.add(text, confidence, latency_ms)
                    self.total_chunks += 1
                    self.total_words  += len(text.split())
                    self.latencies.append(latency_ms)
                    self._print_result(text, confidence, latency_ms)

        except KeyboardInterrupt:
            print(f"\n\n{'─' * 70}")
            print(f"{C.YELLOW}⏹  Stopped by user.{C.RESET}")
        finally:
            self._print_summary()

    # ── Terminal display helpers ───────────────────────────────────────────
    def _print_banner(self):
        print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════════════════╗
║          🎤  Real-Time Speech-to-Text  ·  Whisper [{self.model_size:^6}]         ║
╚══════════════════════════════════════════════════════════════════════╝{C.RESET}
  {C.DIM}Sample rate : {SAMPLE_RATE} Hz    Chunk : {CHUNK_SECONDS}s    Overlap : {OVERLAP_SECONDS}s{C.RESET}
  {C.DIM}Logs saved  : {LOG_DIR.resolve()}{C.RESET}
""")

    def _print_result(self, text: str, confidence: float, latency_ms: float):
        conf_bar = self._conf_bar(confidence)
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"{C.DIM}[{ts}]{C.RESET} {C.GREEN}{C.BOLD}{text}{C.RESET}")
        print(f"       {conf_bar}  {C.DIM}conf={confidence:.0%}  latency={latency_ms:.0f}ms{C.RESET}\n")

    def _print_silence(self):
        # Overwrite same line — unobtrusive
        sys.stdout.write(f"\r{C.DIM}  · silence ·{C.RESET}   ")
        sys.stdout.flush()

    def _conf_bar(self, conf: float, width: int = 12) -> str:
        filled = int(conf * width)
        color  = C.GREEN if conf > 0.75 else C.YELLOW if conf > 0.5 else C.RED
        bar    = "█" * filled + "░" * (width - filled)
        return f"{color}[{bar}]{C.RESET}"

    def _print_summary(self):
        duration = (datetime.now() - self.log.session_start).total_seconds()
        avg_lat  = np.mean(self.latencies) if self.latencies else 0
        wpm      = (self.total_words / duration * 60) if duration > 0 else 0

        print(f"""
{C.CYAN}{C.BOLD}Session Summary{C.RESET}
{'─' * 40}
  Duration     : {duration:.1f}s
  Chunks       : {self.total_chunks}
  Words        : {self.total_words}
  Words/min    : {wpm:.1f}
  Avg latency  : {avg_lat:.0f} ms
  Log (text)   : {self.log.txt_path}
  Log (json)   : {self.log.json_path}
{'─' * 40}
{C.DIM}Run  python wer_report.py  to generate accuracy report.{C.RESET}
""")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Whisper speech-to-text")
    parser.add_argument(
        "--model", default=MODEL_SIZE,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--chunk", type=float, default=CHUNK_SECONDS,
        help="Seconds of audio per chunk (default: 3)",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available audio input devices and exit",
    )
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    CHUNK_SECONDS = args.chunk
    stt = RealtimeSTT(model_size=args.model)
    stt.run()