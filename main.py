# IDK THIS IS PROB VERY TRASH CODE LOTS OF ISSUES also Only speedramping


import os
import cv2
import numpy as np
import librosa
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
import xgboost as xgb
import pickle
import logging
import time
import sys
import random

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CONFIG = {
    "clips_folder": "Clips",
    "songs_folder": "Songs",
    "output_file": "output_car_edit.mp4",
    "intensity": 2.0,
    "min_clip_duration": 0.1,
    "max_clip_duration": 1.0,
    "resolution": (1080, 1920),
    "min_beat_spacing": 0.3,
    "model_path": "trained_model.pkl",  # only uses if u have da model if u want a model train it on trainee.py
}


class VideoEditor:
    def __init__(self, config):
        self.config = config
        self.edit_type = "car"
        self.clips = self.load_clips()
        self.song = self.load_song()
        self.beats = []
        self.bass_energy = []
        self.final_clips = []
        self.clip_metadata = []
        self.frame_rate = 30
        self.current_resolution = self.config["resolution"]
        self.performance_metrics = {"analysis_time": 0, "render_time": 0}
        self.sr = 22050
        try:
            with open(self.config["model_path"], "rb") as f:
                self.model = pickle.load(f)
            logging.info(f"Loaded trained model from {self.config['model_path']}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}. Using default speed ramping.")
            self.model = None

    def load_clips(self):
        clips_path = self.config["clips_folder"]
        if not os.path.exists(clips_path):
            os.makedirs(clips_path)
            raise FileNotFoundError(f"Clips folder {clips_path} not found")
        clips = [
            VideoFileClip(os.path.join(clips_path, f)).resize(self.config["resolution"])
            for f in os.listdir(clips_path)
            if f.endswith(".mp4")
        ]
        if not clips:
            raise FileNotFoundError("No clips found in Clips folder")
        logging.info(f"Loaded {len(clips)} clips")
        return clips

    def load_song(self):
        songs_path = self.config["songs_folder"]
        if not os.path.exists(songs_path):
            os.makedirs(songs_path)
            raise FileNotFoundError(f"Songs folder {songs_path} not found")
        song_files = [f for f in os.listdir(songs_path) if f.endswith(".mp3")]
        if not song_files:
            raise FileNotFoundError("No song found in Songs folder")
        if len(song_files) > 1:
            logging.warning(f"Multiple .mp3 files found, using {song_files[0]}")
        song = AudioFileClip(os.path.join(songs_path, song_files[0]))
        logging.info(f"Loaded song: {song_files[0]}")
        return song

    def analyze_audio(self):
        logging.info("Analyzing audio for bass beat detection")
        y, sr = librosa.load(self.song.filename)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        o_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=o_env, sr=sr)
        S = np.abs(librosa.stft(y_percussive))
        freqs = librosa.fft_frequencies(sr=sr)
        bass_mask = (freqs >= 20) & (freqs <= 150)
        bass_energy = np.mean(S[bass_mask, :], axis=0)
        bass_energy = librosa.util.normalize(bass_energy)
        bass_energy_smooth = np.convolve(bass_energy, np.ones(20) / 20, mode="same")
        beats_time = librosa.frames_to_time(beats, sr=sr)
        strong_beats = []
        last_beat = -self.config["min_beat_spacing"]
        for beat in beats_time:
            energy_index = min(int(beat * sr / 512), len(bass_energy_smooth) - 1)
            if (
                bass_energy_smooth[energy_index] > np.percentile(bass_energy_smooth, 85)
                and beat - last_beat >= self.config["min_beat_spacing"]
            ):
                strong_beats.append(beat)
                last_beat = beat
        self.beats = strong_beats
        if not self.beats:
            self.beats = beats_time.tolist()
            last_beat = -self.config["min_beat_spacing"]
            filtered_beats = []
            for beat in self.beats:
                if beat - last_beat >= self.config["min_beat_spacing"]:
                    filtered_beats.append(beat)
                    last_beat = beat
            self.beats = filtered_beats
        logging.info(f"Detected {len(self.beats)} bass beats")
        self.bass_energy = bass_energy.tolist()

    def compute_motion_features(self, clip, start, duration):
        cap = cv2.VideoCapture(clip.filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logging.warning(f"Invalid FPS for {clip.filename}")
            cap.release()
            return 0
        start_frame = int(start * fps)
        end_frame = int((start + duration) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        motion_energy = 0
        frame_count = 0
        prev_frame = None
        frame_step = 5
        while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None and frame_count % frame_step == 0:
                diff = cv2.absdiff(prev_frame, gray)
                motion_energy += np.mean(diff)
            prev_frame = gray
            frame_count += 1
        cap.release()
        if frame_count == 0:
            logging.warning(f"No frames processed for {clip.filename} at {start}s")
            return 0
        return motion_energy / (frame_count // frame_step + 1)

    def extract_features_for_segment(self, clip, start, duration, beat_idx):
        motion_energy = self.compute_motion_features(clip, start, duration)
        start_frame = int(start * self.sr / 512)
        end_frame = int((start + duration) * self.sr / 512)
        bass_energy_segment = (
            np.mean(self.bass_energy[start_frame:end_frame])
            if end_frame <= len(self.bass_energy)
            else 0
        )
        bass_beat_count = sum(1 for b in self.beats if start <= b < start + duration)
        features = [
            motion_energy,
            bass_energy_segment,
            bass_beat_count,
            duration,
            start / clip.duration,
        ]
        return np.array(features)

    def analyze_clips(self):
        logging.info("Analyzing clips")
        start_time = time.time()
        self.clip_metadata = []
        for clip in self.clips:
            energy = random.uniform(0.3, 1.0)
            self.clip_metadata.append({"clip": clip, "energy": energy})
        self.clip_metadata.sort(key=lambda x: x["energy"], reverse=True)
        self.performance_metrics["analysis_time"] = time.time() - start_time
        logging.info(
            f"Clip analysis complete in {self.performance_metrics['analysis_time']} seconds"
        )

    def apply_speed_ramp(self, clip, start_speed, end_speed, duration):
        logging.info(
            f"Applying sigmoid speed ramp to clip of duration {clip.duration}s"
        )
        if clip.duration < duration or duration < 0.01:
            return clip.speedx(start_speed)

        def speed_function(t):
            progress = t / duration
            sigmoid = 1 / (1 + np.exp(-6 * (progress - 0.5)))
            speed = start_speed + (end_speed - start_speed) * sigmoid
            return max(speed, 0.2)

        ramped_clip = clip.fl_time(
            lambda t: t / speed_function(t), apply_to=["mask", "audio"]
        )
        avg_speed = (start_speed + end_speed) / 2
        ramped_clip = ramped_clip.set_duration(duration / avg_speed)
        if ramped_clip.duration > duration:
            ramped_clip = ramped_clip.subclip(0, duration)
        return ramped_clip

    def select_clips(self):
        logging.info("Starting clip selection")
        song_duration = self.song.duration
        current_time = 0
        beat_idx = 0
        available_clips = self.clip_metadata.copy()
        used_clips = []
        while current_time < song_duration:
            if beat_idx >= len(self.beats):
                beat_idx = 0
            if beat_idx >= len(self.beats) - 1:
                beat_duration = 0.5
            else:
                beat_duration = (
                    self.beats[beat_idx + 1] - self.beats[beat_idx]
                    if beat_idx + 1 < len(self.beats)
                    else 0.5
                )
            clip_duration = min(
                max(
                    beat_duration * self.config["intensity"],
                    self.config["min_clip_duration"],
                ),
                self.config["max_clip_duration"],
            )
            if current_time + clip_duration > song_duration:
                clip_duration = song_duration - current_time
            logging.info(f"Processing beat {beat_idx}/{len(self.beats)-1}")
            if not available_clips:
                if not used_clips:
                    logging.warning(
                        "No clips available or reusable. Stopping clip selection."
                    )
                    break
                available_clips = used_clips.copy()
                used_clips = []
                logging.info("Reusing clips")
            bass_energy_at_beat = 0.5
            if self.beats and self.bass_energy and beat_idx < len(self.beats):
                energy_index = min(
                    int(self.beats[beat_idx] * 22050 / 512), len(self.bass_energy) - 1
                )
                bass_energy_at_beat = (
                    self.bass_energy[energy_index]
                    if 0 <= energy_index < len(self.bass_energy)
                    else 0.5
                )
            clip_meta = random.choice(available_clips) if available_clips else None
            if clip_meta:
                clip = clip_meta["clip"]
                clip_duration = min(
                    clip_duration, clip.duration
                )  # Ensure clip_duration fits clip
                if beat_idx < len(self.beats):
                    nearest_beat = min(
                        self.beats,
                        key=lambda x: (
                            abs(x - current_time)
                            if x - current_time >= 0
                            and x - current_time < clip.duration
                            else float("inf")
                        ),
                    )
                    if (
                        nearest_beat - current_time < 0
                        or nearest_beat - current_time > clip.duration
                    ):
                        clip_start = np.random.uniform(0, clip.duration - clip_duration)
                    else:
                        clip_start = (
                            nearest_beat - current_time
                            if nearest_beat - current_time >= 0
                            else 0
                        )
                else:
                    clip_start = np.random.uniform(0, clip.duration - clip_duration)
                subclip = clip.subclip(clip_start, clip_start + clip_duration)
                features = self.extract_features_for_segment(
                    clip, clip_start, clip_duration, beat_idx
                )
                if self.model:
                    try:
                        dmatrix = xgb.DMatrix(features.reshape(1, -1))
                        prediction = self.model.predict(dmatrix)[0]
                        start_speed, end_speed = prediction
                        start_speed = max(0.2, min(start_speed, 1.0))
                        end_speed = max(0.5, min(end_speed, 2.5))
                    except Exception as e:
                        logging.warning(
                            f"Model prediction failed: {e}. Using default speeds."
                        )
                        start_speed = 0.2 + 0.1 * bass_energy_at_beat
                        end_speed = 0.5 + 2.0 * bass_energy_at_beat
                else:
                    start_speed = 0.2 + 0.1 * bass_energy_at_beat
                    end_speed = 0.5 + 2.0 * bass_energy_at_beat
                subclip = self.apply_speed_ramp(
                    subclip, start_speed, end_speed, clip_duration
                )
                self.final_clips.append(subclip)
                current_time += subclip.duration
                available_clips.remove(clip_meta)
                used_clips.append(clip_meta)
            beat_idx += 1
        logging.info(f"Clip selection complete. Total duration: {current_time} seconds")

    def render_video(self):
        logging.info("Rendering final video")
        start_time = time.time()
        final_video = concatenate_videoclips(self.final_clips, method="compose")
        final_video = final_video.subclip(0, self.song.duration)
        final_video = final_video.set_audio(self.song)
        final_video = final_video.resize(self.current_resolution)
        final_video.write_videofile(
            self.config["output_file"],
            codec="libx264",
            audio_codec="aac",
            fps=self.frame_rate,
            bitrate="10000k",  # hehehehehhe
            threads=8,
            preset="veryslow",
            audio_bitrate="320k",
        )
        self.performance_metrics["render_time"] = time.time() - start_time
        logging.info(
            f"Rendering complete in {self.performance_metrics['render_time']} seconds"
        )

    def run(self):
        logging.info("Starting editing pipeline")
        self.analyze_audio()
        self.analyze_clips()
        self.select_clips()
        self.render_video()
        logging.info("Editing complete")
        logging.info(f"Performance metrics: {self.performance_metrics}")


def main():
    editor = VideoEditor(CONFIG)
    editor.run()


if __name__ == "__main__":
    main()
