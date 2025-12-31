from whisper_tflite import WhisperModel
model = WhisperModel("./whisper-tiny-en.tflite")
segments, _ = model.transcribe("audio.mp3")
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
