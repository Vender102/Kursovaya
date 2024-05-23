import datetime
import whisperx
import os
import gc
import torch

def process_audio_files(data_dir, output_dir, batch_size=32, device='cuda', compute_type = "int8" ):
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)

    for filename in os.listdir(data_dir):
        if filename.endswith(".mp3"):  # or any other audio format you have
            path = os.path.join(data_dir, filename)
            audio = whisperx.load_audio(path)

            result = model.transcribe(audio, batch_size=batch_size)

            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

            diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)

            result = whisperx.assign_word_speakers(diarize_segments, result)
            segments = result["segments"]

            def time(secs):
                return datetime.timedelta(seconds=round(secs))

            with open(os.path.join(output_dir, f"{filename.split('.')[0]}.txt"), "w", encoding="utf-8") as f:
                for (i, segment) in enumerate(segments):
                    print(i, "  speak  ", segment)
                    segment_speaker = segment.get("speaker", "Unknown")
                    if i == 0 or segments[i - 1].get("speaker", "Unknown") != segment_speaker:
                        f.write("\n" + segment_speaker + ' ' + str(time(segment["start"])) + '\n')
                    f.write(segment["text"][0:] + ' ')

            # Clear memory
            del audio, result, model_a, metadata, diarize_segments, segments
            gc.collect()
            torch.cuda.empty_cache()

    del model, diarize_model
    gc.collect()
    torch.cuda.empty_cache()

process_audio_files("Record ATC", "processed")
