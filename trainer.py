import os
import cv2
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import logging
from scipy.signal import find_peaks
import concurrent.futures
import tempfile
import shutil
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

TRAINER_CONFIG = {
    "trainer_folder": "Trainer",
    "model_output": "trained_model.pkl",
    "feature_window": 0.5,
    "min_peak_height": 0.6,
    "n_threads": 8,
    "speed_ramp_focus": True,
}


class carrrrr:
    def __init__(self, config):
        self.config = config
        self.edits = []
        self.features = []
        self.labels = []
        self.sr = 22050
        self.temp_dir = tempfile.mkdtemp()
        self.lo_edits()

    def lo_edits(self):
        trainer_path = self.config["trainer_folder"]
        if not os.path.exists(trainer_path):
            os.makedirs(trainer_path)
            raise FileNotFoundError(
                f"Trainer folder '{trainer_path}' created but empty. Add .mp4 files."
            )

        for f in os.listdir(trainer_path):
            if f.endswith(".mp4"):
                file_path = os.path.join(trainer_path, f)
                try:
                    clip = VideoFileClip(file_path)
                    if clip.duration <= 0:
                        raise ValueError("Clip has zero or negative duration")
                    self.edits.append(clip)
                    logging.info(f"Successfully loaded {f}")
                except Exception as e:
                    logging.error(f"Failed to load {f}: {e}")

        if not self.edits:
            raise FileNotFoundError(f"No valid .mp4 files found in '{trainer_path}'.")
        logging.info(f"Loaded {len(self.edits)} valid car edits from '{trainer_path}'.")

    def compute_motion_features(self, clip, start, duration):
        cap = cv2.VideoCapture(clip.filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logging.warning(f"Invalid FPS for {clip.filename}")
            cap.release()
            return 0, 0

        start_frame = int(start * fps)
        end_frame = int((start + duration) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        motion_energy = 0
        motion_direction = 0
        frame_count = 0
        prev_frame = None
        prev_mean = 0

        while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_mean = np.mean(gray)
            if prev_frame is not None and frame_count % 5 == 0:
                diff = cv2.absdiff(prev_frame, gray)
                motion_energy += np.mean(diff)
                if current_mean > prev_mean:
                    motion_direction += 1
                elif current_mean < prev_mean:
                    motion_direction -= 1
            prev_frame = gray
            prev_mean = current_mean
            frame_count += 1

        cap.release()
        if frame_count == 0:
            logging.warning(f"No frames processed for {clip.filename} at {start}s")
            return 0, 0
        motion_energy /= frame_count // 5 + 1
        motion_direction = np.sign(motion_direction) if frame_count > 5 else 0
        return motion_energy, motion_direction

    def compute_color_dynamics(self, clip, start, duration):
        cap = cv2.VideoCapture(clip.filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            return 0

        start_frame = int(start * fps)
        end_frame = int((start + duration) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        color_variance = 0
        frame_count = 0
        prev_color = None

        while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            current_color = np.mean(hsv, axis=(0, 1))
            if prev_color is not None and frame_count % 5 == 0:
                color_variance += np.mean(np.abs(current_color - prev_color))
            prev_color = current_color
            frame_count += 1

        cap.release()
        return color_variance / (frame_count // 5 + 1) if frame_count > 0 else 0

    def detect_transitions(self, clip):
        cap = cv2.VideoCapture(clip.filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            return []

        prev_frame = None
        transition_times = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                diff_mean = np.mean(diff)
                if diff_mean > 25:
                    transition_times.append(frame_idx / fps)
            prev_frame = gray
            frame_idx += 1

        cap.release()
        return transition_times

    def extract_audio_features(self, clip, edit_id):
        try:
            y, sr = librosa.load(clip.filename, sr=self.sr, mono=True)
        except Exception as e:
            logging.warning(
                f"Audio extraction failed for {clip.filename}: {e}. Using silence."
            )
            y = np.zeros(int(clip.duration * self.sr))
            sr = self.sr

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beats = librosa.frames_to_time(beat_frames, sr=sr)
        rms = librosa.feature.rms(y=y)[0]
        rms_normalized = librosa.util.normalize(rms)
        peaks, _ = find_peaks(
            rms_normalized, height=self.config["min_peak_height"], distance=5
        )
        peak_times = librosa.frames_to_time(peaks, sr=sr)
        tempo_variation = (
            np.std(librosa.feature.tempo(y=y, sr=sr, aggregate=None))
            if len(beats) > 1
            else 0
        )

        return {
            "beats": beats,
            "rms": rms_normalized,
            "peaks": peak_times,
            "tempo_variation": tempo_variation,
        }

    def extract_segment_features(
        self, clip, start, duration, audio_features, transitions
    ):
        if start + duration > clip.duration:
            duration = clip.duration - start

        motion_energy, motion_direction = self.compute_motion_features(
            clip, start, duration
        )
        color_dynamics = self.compute_color_dynamics(clip, start, duration)
        transition_prob = (
            1.0 if any(abs(t - start) < 0.05 for t in transitions) else 0.0
        )

        start_frame = int(start * self.sr / 512)
        end_frame = int((start + duration) * self.sr / 512)
        rms_segment = (
            np.mean(audio_features["rms"][start_frame:end_frame])
            if end_frame <= len(audio_features["rms"])
            else 0
        )
        beat_intensity = (
            np.mean(
                [
                    audio_features["rms"][int(b * self.sr / 512)]
                    for b in audio_features["beats"]
                    if start <= b < start + duration
                ]
            )
            if any(start <= b < start + duration for b in audio_features["beats"])
            else 0
        )
        beat_count = sum(
            1 for b in audio_features["beats"] if start <= b < start + duration
        )
        peak_count = sum(
            1 for p in audio_features["peaks"] if start <= p < start + duration
        )

        features = [
            motion_energy,
            motion_direction,
            rms_segment,
            beat_intensity,
            beat_count,
            peak_count,
            duration,
            start / clip.duration,
            color_dynamics,
            transition_prob,
            audio_features["tempo_variation"],
        ]

        base_speed = 1.0
        start_speed = max(
            0.2, min(base_speed - (motion_energy * 0.3 + rms_segment * 0.2), 1.5)
        )
        end_speed = min(base_speed + (beat_intensity * 0.5 + peak_count * 0.3), 2.5)
        ramp_duration = min(duration, 0.3 + (beat_count + peak_count) * 0.1)
        smoothness_factor = max(
            0.5, min(1.0 - (np.abs(motion_direction) + color_dynamics) * 0.2, 1.0)
        )

        labels = [start_speed, end_speed, ramp_duration, smoothness_factor]
        return features, labels

    def process_edit(self, edit, edit_id):
        logging.info(f"Starting processing of {edit.filename}")
        audio_features = self.extract_audio_features(edit, edit_id)
        transitions = self.detect_transitions(edit)

        features = []
        labels = []
        current_time = 0

        while current_time < edit.duration:
            duration = min(self.config["feature_window"], edit.duration - current_time)
            feat, lab = self.extract_segment_features(
                edit, current_time, duration, audio_features, transitions
            )
            features.append(feat)
            labels.append(lab)
            current_time += duration
        logging.info(f"Completed processing {edit.filename}")
        return features, labels

    def extract_features(self):
        logging.info("Extracting features from car edits...")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config["n_threads"]
        ) as executor:
            future_to_edit = {
                executor.submit(self.process_edit, edit, i): edit
                for i, edit in enumerate(self.edits)
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_edit),
                total=len(self.edits),
                desc="Processing car edits",
            ):
                try:
                    feat, lab = future.result()
                    self.features.extend(feat)
                    self.labels.extend(lab)
                except Exception as e:
                    logging.error(
                        f"Error processing edit {future_to_edit[future].filename}: {e}"
                    )
                    raise

        if not self.features:
            raise ValueError("No features extracted from training data.")
        logging.info(f"Extracted {len(self.features)} feature sets.")

    def train_model(self):
        self.extract_features()
        X = np.array(self.features)
        y = np.array(self.labels)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Feature-label mismatch: X has {X.shape[0]} samples, y has {y.shape[0]} samples"
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "eta": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "seed": 42,
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        try:
            model = xgb.train(
                params,
                dtrain,
                500,
                [(dtrain, "train"), (dtest, "eval")],
                early_stopping_rounds=20,
                verbose_eval=False,
            )
        except TypeError:
            logging.info(
                "Falling back to basic XGBoost training without early stopping"
            )
            model = xgb.train(
                params,
                dtrain,
                500,
                [(dtrain, "train"), (dtest, "eval")],
                verbose_eval=False,
            )

        y_pred = model.predict(dtest)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Model performance - MSE: {mse}, R2: {r2}")

        try:
            importance = model.get_score(importance_type="gain")
            feature_names = [
                "motion_energy",
                "motion_direction",
                "rms_segment",
                "beat_intensity",
                "beat_count",
                "peak_count",
                "duration",
                "position",
                "color_dynamics",
                "transition_prob",
                "tempo_variation",
            ]
            logging.info("Feature importance:")
            for i, name in enumerate(feature_names):
                imp = importance.get(f"f{i}", 0)
                logging.info(f"{name}: {imp:.4f}")
        except:
            logging.warning("Could not get feature importance")

        try:
            with open(self.config["model_output"], "wb") as f:
                pickle.dump(model, f)
            logging.info(f"Model saved to '{self.config['model_output']}'")
        except Exception as e:
            logging.error(f"Failed to save model: {e}")

        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logging.error(f"Failed to clean up temp dir {self.temp_dir}: {e}")

    def run(self):
        try:
            self.train_model()
        except Exception as e:
            logging.error(f"Training pipeline failed: {e}")
            raise


def main():
    try:
        trainer = carrrrr(TRAINER_CONFIG)
        trainer.run()
    except Exception as e:
        logging.error(f"Main execution failed: {e}")


if __name__ == "__main__":
    main()
