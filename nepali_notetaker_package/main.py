from asr_model.model import NepaliASR

if __name__ == "__main__":
    audio_path = r"D:\nepali_notetaker_project\nepali_notetaker_package\DataSet\Nepali Speech To Text Dataset\audio_chunks\2079-11-21_80.wav"
    asr = NepaliASR()
    transcript = asr.transcribe(audio_path)
    print(transcript)
