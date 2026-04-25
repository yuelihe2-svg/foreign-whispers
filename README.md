# Foreign Whispers Dubbing Pipeline


Foreign Whispers is an open-source video dubbing pipeline that takes an English YouTube video as input and produces a Spanish dubbed video as output. The system uses Whisper for speech transcription, pyannote for speaker diarization, Argos Translate for machine translation, Chatterbox TTS for local speech synthesis and voice cloning, and FFmpeg for final video/audio/caption stitching.


This project is designed to run with local GPU resources and does not rely on paid proprietary dubbing APIs.


---

## 1. Project Overview


The goal of this project is to reproduce the main functionality of commercial AI dubbing systems using open-source components. Given a YouTube URL, the pipeline downloads the source video, extracts or generates captions, transcribes the audio, translates the transcript into Spanish, optionally assigns speaker labels, synthesizes Spanish speech with Chatterbox TTS, and stitches the dubbed audio and WebVTT captions back onto the original video.


The final output is a dubbed MP4 file with translated WebVTT captions that can be viewed in the Foreign Whispers Dubbing Studio frontend.


---

## 2. Main Features


- Download YouTube videos and caption files using `yt-dlp`.
- Transcribe speech with Whisper.
- Compare Whisper segments with YouTube captions.
- Translate English transcript segments into Spanish using Argos Translate.
- Generate shorter translation candidates for timing-constrained dubbing.
- Run speaker diarization with pyannote and merge speaker labels into transcript segments.
- Select speaker-specific reference voices for Chatterbox TTS.
- Generate dubbed Spanish audio using local Chatterbox TTS.
- Align or schedule subtitles to the generated TTS timeline.
- Stitch the original video stream with dubbed audio and WebVTT captions using FFmpeg.
- Demonstrate the full workflow in a Next.js frontend at `http://localhost:8501`.


---

## 3. System Architecture


The system is organized into four main layers:


| Layer | Description | Runtime |
|---|---|---|
| Frontend | Next.js Dubbing Studio UI | Port `8501` |
| API Orchestrator | FastAPI backend that coordinates pipeline stages | Port `8080` |
| GPU Services | Whisper STT and Chatterbox TTS services | Ports `8000` and `8020` |
| Python Library | Alignment, diarization helpers, reranking, and evaluation logic | Local Python package |


The typical pipeline is:

    YouTube URL
      -> Download
      -> Transcribe
      -> Diarize
      -> Translate
      -> TTS
      -> Stitch
      -> Dubbed MP4 + WebVTT captions


---

## 4. Repository Structure


    api/
      src/
        routers/          FastAPI route handlers
        services/         Backend service and engine logic
        core/             Shared configuration and dependencies

    foreign_whispers/
      alignment.py        Duration prediction and alignment logic
      diarization.py      Speaker-label assignment logic
      evaluation.py       Dubbing quality scorecard
      reranking.py        Duration-aware translation candidates
      voice_resolution.py Speaker reference voice fallback logic

    frontend/
      src/                Next.js frontend source code

    notebooks/
      pipeline_end_to_end/
      download_integration/
      transcription_integration/
      translation_integration/
      diarization_integration/
      alignment_integration/
      tts_integration/
      stitch_integration/

    pipeline_data/
      api/                Cached runtime artifacts, generated locally


---

## 5. Requirements


The application is intended to run on a machine with NVIDIA GPU support and Docker installed.


Required tools:

- Docker and Docker Compose
- NVIDIA Container Toolkit
- Python environment with `uv`
- Git
- A HuggingFace token with access to `pyannote/speaker-diarization-3.1`


GPU notes:

- Whisper and Chatterbox TTS are GPU-oriented services.
- Chatterbox TTS generation can be slow or unstable on small or old GPUs.
- For the submitted demo artifacts, some heavy TTS generation was run on an HPC GPU machine and then copied back into `pipeline_data/api/`.
- This HPC workflow is not a required code dependency. A sufficiently capable local NVIDIA GPU can run the same pipeline directly through Docker.


---

## 6. Environment Setup


Create a `.env` file in the project root:

    FW_HF_TOKEN=hf_your_token_here
    LOGFIRE_TOKEN=your_logfire_token_optional


Before running diarization, make sure the HuggingFace account has accepted the license for `pyannote/speaker-diarization-3.1`.


Install the local Python package:

    uv sync


---

## 7. Running the Docker Stack


Start the full stack with NVIDIA GPU support:

    docker compose --profile nvidia up -d


Check the API health endpoint:

    curl http://localhost:8080/healthz

Expected response:

    {"status":"ok"}


Open the frontend:

    http://localhost:8501


---

## 8. Running the Pipeline from the Frontend


After the Docker stack is running, open the frontend at:

    http://localhost:8501


The frontend lets the user select or submit a video and run the pipeline stages:

1. Download
2. Transcribe
3. Diarize
4. Translate
5. TTS
6. Stitch


For cached videos, rerunning the pipeline is faster because intermediate artifacts are reused from `pipeline_data/api/`.


The final dubbed video can be played directly in the frontend with translated captions.


---

## 9. Running the Integration Notebooks


The project includes integration notebooks for the end-to-end workflow and for each individual pipeline stage:

    notebooks/pipeline_end_to_end/pipeline_end_to_end.ipynb
    notebooks/download_integration/download_integration.ipynb
    notebooks/transcription_integration/transcription_integration.ipynb
    notebooks/translation_integration/translation_integration.ipynb
    notebooks/diarization_integration/diarization_integration.ipynb
    notebooks/alignment_integration/alignment_integration.ipynb
    notebooks/tts_integration/tts_integration.ipynb
    notebooks/stitch_integration/stitch_integration.ipynb


To execute a notebook from the command line:

    python -m jupyter nbconvert \
      --to notebook \
      --execute \
      --inplace \
      --ExecutePreprocessor.timeout=1800 \
      notebooks/pipeline_end_to_end/pipeline_end_to_end.ipynb


The notebooks document and verify the major tasks:

- Download artifact inspection
- YouTube captions vs Whisper transcription comparison
- Duration-aware translation reranking
- Speaker diarization and speaker-label merging
- Improved duration prediction and global alignment
- Dubbing quality scorecard
- Speaker-aware Chatterbox TTS
- FFmpeg stitching and WebVTT caption generation


---

## 10. Output Artifacts


Runtime artifacts are stored under:

    pipeline_data/api/


Important output locations:

    pipeline_data/api/videos/
    pipeline_data/api/youtube_captions/
    pipeline_data/api/transcriptions/whisper/
    pipeline_data/api/diarizations/
    pipeline_data/api/translations/argos/
    pipeline_data/api/tts_audio/chatterbox/
    pipeline_data/api/dubbed_videos/
    pipeline_data/api/dubbed_captions/


The final dubbed videos are stored in:

    pipeline_data/api/dubbed_videos/{config}/


The translated WebVTT captions are stored in:

    pipeline_data/api/dubbed_captions/


Large runtime artifacts such as MP4 and WAV files are not intended to be committed to Git. They can be shared separately as sample input/output artifacts.


---

## 11. Implemented Technical Tasks


This project implements the required notebook tasks and several integration fixes.


### 11.1 Duration-Aware Translation Reranking


`foreign_whispers/reranking.py` implements `get_shorter_translations()`, which generates shorter Spanish candidates when a translated segment is too long for the available TTS time budget.


The implementation uses rule-based simplification, filler removal, phrase shortening, and fallback truncation at word boundaries. This is a lightweight local approach that does not require a paid LLM or external translation API.


### 11.2 Speaker Diarization


`foreign_whispers/diarization.py` implements `assign_speakers()`, which assigns speaker labels to transcript segments using maximum temporal overlap with pyannote diarization segments.


The API endpoint `POST /api/diarize/{video_id}` extracts audio, runs or loads cached pyannote diarization, writes diarization JSON, and merges speaker labels back into the Whisper transcription JSON.


### 11.3 Alignment and Evaluation


`foreign_whispers/alignment.py` improves TTS duration prediction and implements a global alignment strategy for choosing between natural fit, stretching, gap shifting, and shorter-translation requests.


`foreign_whispers/evaluation.py` implements a dubbing scorecard that combines timing accuracy, stretch risk, intelligibility proxy, semantic fidelity risk, and overall quality.


### 11.4 Speaker-Aware TTS


`foreign_whispers/voice_resolution.py` implements speaker reference voice resolution with a fallback chain:

    speaker-specific voice
      -> language default voice
      -> global default voice


The TTS API supports explicit `speaker_wav` selection and automatic per-speaker voice mapping when speaker labels are available.


### 11.5 TTS-Scheduled Captions


The stitch stage serves translated WebVTT captions synchronized to the assembled TTS timeline. Each cue shows only the current subtitle and is wrapped to at most two lines.


This avoids the earlier problem where captions could overlap or show the previous subtitle line below the current subtitle.


---

## 12. Demo Videos


The completed demo uses the following videos:

    1. Rob Reiner: The 60 Minutes Interview
    2. Strait of Hormuz disruption threatens to shake global economy
    3. Alysa Liu: The 60 Minutes Interview


The recommended primary demo video is:

    Rob Reiner: The 60 Minutes Interview


The Military Drones video was not used as the submitted demo because it contains more than 1300 translated segments and is computationally expensive to synthesize with Chatterbox TTS. The same pipeline supports it, but the final demo focuses on shorter completed clips for reproducibility.


---

## 13. Sample Input and Output


Sample input:

    YouTube URL for Rob Reiner: The 60 Minutes Interview


Sample output:

    pipeline_data/api/dubbed_videos/c-e5604bc/Rob Reiner: The 60 Minutes Interview.mp4


The output video preserves the original video stream, replaces the original audio with generated Spanish TTS audio, and serves translated WebVTT captions through the API/frontend.


---

## 14. Reproducing Heavy TTS Generation


The submitted code can run the full pipeline through the local Docker stack when the machine has a compatible NVIDIA GPU. In my submitted demo, the heaviest Chatterbox TTS generation jobs were run on an HPC GPU machine to improve stability and reduce waiting time. The generated WAV artifacts were then copied back into the project’s `pipeline_data/api/tts_audio/chatterbox/` directory and stitched locally.


This HPC step is not required by the application design. It is an optional acceleration workflow for long or unstable TTS runs. A machine with a sufficiently capable NVIDIA GPU can run the same TTS stage directly through the Chatterbox service on port `8020`.


For very long videos, segment-level caching is important. If a TTS run fails halfway, completed artifacts can be reused, and failed segments can be regenerated instead of restarting the entire pipeline.


---

## 15. Known Limitations


- Long videos with hundreds or thousands of short segments are expensive to synthesize with Chatterbox TTS.
- Chatterbox may occasionally fail during long continuous generation runs, so cached segment-level generation and repair workflows are useful.
- Very dense speech may require shortening translated text or relaxing the TTS schedule to avoid unnatural speed-up.
- The current duration-aware reranking is rule-based and can be improved with stronger semantic preservation methods.
- Speaker diarization quality depends on pyannote output and clean source audio.
- Voice cloning quality depends on the quality and cleanliness of reference WAV samples.


---

## 16. Reproducibility Notes


All major pipeline artifacts are cached under `pipeline_data/api/`. If an artifact already exists, the corresponding stage can reuse it instead of recomputing from scratch. This makes repeated frontend demos faster and more stable.


For a clean rerun, remove the relevant cached files under `pipeline_data/api/` and run the pipeline again from the frontend or notebooks.


---

## 17. Quick Start


    # 1. Clone the repository
    git clone https://github.com/yuelihe2-svg/foreign-whispers.git
    cd foreign-whispers

    # 2. Configure environment variables
    cat > .env <<'EOF'
    FW_HF_TOKEN=hf_your_token_here
    EOF

    # 3. Install Python dependencies
    uv sync

    # 4. Start the Docker stack
    docker compose --profile nvidia up -d

    # 5. Check API health
    curl http://localhost:8080/healthz

    # 6. Open the frontend
    # http://localhost:8501


---

## 18. Final Submission Materials


This submission includes:

1. The GitHub codebase with completed pipeline integration.
2. This README explaining how to run the application.
3. A technical report describing the architecture and implementation details.
4. A screen recording demo of the frontend application.
5. A sample dubbed output video generated by the pipeline.


---

## 19. Suggested Demo Flow


For the screen-recorded demo, the recommended flow is:

1. Start the Docker stack.
2. Open `http://localhost:8501`.
3. Select the Rob Reiner demo video.
4. Run or show the cached pipeline stages.
5. Confirm that Download, Transcribe, Diarize, Translate, TTS, and Stitch complete successfully.
6. Play the final dubbed video in the frontend.
7. Enable or show translated captions.
8. Briefly show another completed example, such as the Strait of Hormuz clip or the Alysa Liu clip.


The Rob Reiner clip is recommended as the primary demo because it has stable audio quality and clear frontend playback.


---

## 20. Notes for Evaluators


The codebase implements the complete open-source dubbing pipeline. The repository itself does not include large generated MP4 or WAV artifacts, because those files are too large for normal Git version control. They should be submitted separately as sample input/output artifacts.


If the evaluator wants to regenerate outputs from scratch, the machine should have a working NVIDIA Docker setup and enough GPU memory for Whisper, pyannote, and Chatterbox TTS. If cached artifacts are provided, the frontend can demonstrate the final stitched outputs much faster.


The notebooks have been executed successfully and document the implementation and verification process for each stage of the pipeline.
