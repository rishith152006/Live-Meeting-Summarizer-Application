#!/usr/bin/env python3
"""
Setup & test script for Week 2 Real-Time STT
Run this first to verify your environment is ready.
"""

import subprocess
import sys


PACKAGES = [
    ("openai-whisper", "whisper"),
    ("sounddevice",    "sounddevice"),
    ("scipy",          "scipy"),
    ("numpy",          "numpy"),
]

def install(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])

def check_and_install():
    print("🔍  Checking dependencies…\n")
    for pip_name, import_name in PACKAGES:
        try:
            __import__(import_name)
            print(f"  ✅  {pip_name}")
        except ImportError:
            print(f"  📦  Installing {pip_name}…")
            try:
                install(pip_name)
                print(f"  ✅  {pip_name} installed")
            except Exception as e:
                print(f"  ❌  Failed to install {pip_name}: {e}")
                sys.exit(1)

    # ffmpeg check (needed by Whisper)
    try:
        subprocess.run(["ffmpeg", "-version"],
                       capture_output=True, check=True)
        print("  ✅  ffmpeg")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ⚠️   ffmpeg not found — install it for best Whisper performance:")
        print("       macOS : brew install ffmpeg")
        print("       Ubuntu: sudo apt install ffmpeg")
        print("       Windows: https://ffmpeg.org/download.html")

    print()


def test_microphone():
    import sounddevice as sd
    import numpy as np

    print("🎙  Testing microphone (2 seconds)…")
    try:
        audio = sd.rec(int(2 * 16000), samplerate=16000,
                       channels=1, dtype="float32")
        sd.wait()
        rms = float(np.sqrt(np.mean(audio**2)))
        print(f"  ✅  Mic OK — RMS level: {rms:.4f}")
        if rms < 0.001:
            print("  ⚠️   Very low signal — check mic is not muted")
    except Exception as e:
        print(f"  ❌  Microphone error: {e}")
        print("     List devices: python realtime_stt.py --list-devices")
        return False
    return True


def test_whisper():
    import whisper
    import numpy as np

    print("\n🤖  Testing Whisper model (tiny)…")
    try:
        model = whisper.load_model("tiny")
        dummy = np.zeros(16000, dtype=np.float32)  # 1s silence
        result = model.transcribe(dummy, language="en", fp16=False)
        print(f"  ✅  Whisper OK — result: '{result['text']}'")
    except Exception as e:
        print(f"  ❌  Whisper error: {e}")
        return False
    return True


if __name__ == "__main__":
    check_and_install()
    mic_ok     = test_microphone()
    whisper_ok = test_whisper()

    print()
    if mic_ok and whisper_ok:
        print("🚀  All checks passed! Run:  python realtime_stt.py")
        print()
        print("   Options:")
        print("   python realtime_stt.py --model tiny     # fastest")
        print("   python realtime_stt.py --model base     # balanced (default)")
        print("   python realtime_stt.py --model small    # more accurate")
        print("   python realtime_stt.py --chunk 5        # 5-second chunks")
        print()
        print("   After recording:")
        print("   python wer_report.py --all")
    else:
        print("⚠️   Some checks failed — see messages above.")