import os
import csv
import time
import math
import random
import struct
import threading
import numpy as np
import serial
import pygame

PORT = "COM3"
BAUD = 921600

ANGLE_OFFSET_DEG = 270.0
ANGLE_INVERT = False

STALE_AFTER_SEC = 0.25

WIDTH, HEIGHT = 1920, 1080
CENTER = (WIDTH // 2, HEIGHT // 2)

SHIFT_X = 440
SHIFT_Y = -200

MIRROR_UI = True

RADIUS = 470
ARC_SPAN = math.radians(85)
THICKNESS = 112
RESOLUTION = 200

ARC_COLOR = (255, 255, 255)
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (15, 15, 20)

FONT_SIZE = 120

FAKE_SPEAKER = False
FAKE_ANGLE_DEG = 0
FAKE_MODE_LABEL = "SRP"

STEER_SPEED_DEG = 20.0

HOVER_FAST_STD_DEG = 4.5
HOVER_FAST_SPEED = 9.0
HOVER_SLOW_STD_DEG = 3.0
HOVER_SLOW_SPEED = 0.9
HOVER_MAX_DEV_DEG = 15.0

JUMP_RATE_PER_SEC = 0.25
JUMP_MIN_DEG = 30.0
JUMP_MAX_DEG = 150.0
JUMP_HOLD_MIN_SEC = 0.03
JUMP_HOLD_MAX_SEC = 0.05

SOUND_CLASS_ENABLED = True
SOUND_CLASS_USE_YAMNET = True
SOUND_CLASS_MANUAL_LABEL = "Speech"

YAMNET_MODEL = "https://tfhub.dev/google/yamnet/1"
TFHUB_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tfhub_cache")

AUDIO_RATE = 16000
AUDIO_GAIN = 4.0
PC_MIC_GAIN = 1.0
AUDIO_LEVEL_THRESHOLD = 20
PC_MIC_LEVEL_THRESHOLD = 150
CLASS_WINDOW_SEC = 0.975
CLASS_HOP_SEC = 0.25
CLASS_SMOOTHING = 0.6
CLASS_MIN_SCORE = 0.25
CLASS_HIDE = {"Silence"}
CLASS_HOLD_SEC = 1.0
CLASS_SHORT_NAMES = True
CLASS_SHOW_SCORE = False
CLASS_PRINT = True

CLASS_FONT_SIZE = 90
CLASS_FONT_MIN = 40
CLASS_BOX_Y_OFFSET = 0
CLASS_BOX_MAX_W = int(1.6 * (RADIUS - THICKNESS / 2))
CLASS_BOX_MIN_W = 320
CLASS_BOX_PAD_X = 36
CLASS_BOX_PAD_Y = 22
CLASS_BOX_FILL = (30, 30, 38, 230)
CLASS_BOX_BORDER = (255, 255, 255, 255)
CLASS_BOX_BORDER_W = 4
CLASS_BOX_RADIUS = 24

AUDIO_MAGIC = b"\xA5\x5A\x41\x55"
AUDIO_HDR_LEN = 8
AUDIO_MAX_SAMPLES = 4096
MAX_TEXT_LINE = 4096
SERIAL_RX_BUFFER = 1 << 20


class Hover:

    def __init__(self):
        self.fast = 0.0
        self.slow = 0.0

    @staticmethod
    def _step(x, std, speed, dt):
        return x - speed * x * dt + std * math.sqrt(2.0 * speed * dt) * random.gauss(0.0, 1.0)

    def update(self, dt):
        self.fast = self._step(self.fast, HOVER_FAST_STD_DEG, HOVER_FAST_SPEED, dt)
        self.slow = self._step(self.slow, HOVER_SLOW_STD_DEG, HOVER_SLOW_SPEED, dt)
        total = self.fast + self.slow
        return HOVER_MAX_DEV_DEG * math.tanh(total / HOVER_MAX_DEV_DEG)


class Glitch:

    def __init__(self):
        self.offset = 0.0
        self.remaining = 0.0

    def update(self, dt):
        if self.remaining > 0.0:
            self.remaining -= dt
            if self.remaining <= 0.0:
                self.offset = 0.0
        elif random.random() < JUMP_RATE_PER_SEC * dt:
            self.offset = random.choice((-1, 1)) * random.uniform(JUMP_MIN_DEG, JUMP_MAX_DEG)
            self.remaining = random.uniform(JUMP_HOLD_MIN_SEC, JUMP_HOLD_MAX_SEC)
        return self.offset


class SoundClassifier:

    def __init__(self, gain, level_threshold):
        self.gain = gain
        self.level_threshold = level_threshold
        self.window = int(round(CLASS_WINDOW_SEC * AUDIO_RATE))
        self.hop = int(round(CLASS_HOP_SEC * AUDIO_RATE))
        self.ring = np.zeros(self.window, dtype=np.float32)
        self.filled = 0
        self.fresh = 0
        self.label = None
        self.score = 0.0
        self.label_time = 0.0
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        self.wake.set()

    def push(self, pcm):
        x = np.asarray(pcm, dtype=np.float32)
        n = len(x)
        if n == 0:
            return
        with self.lock:
            if n >= self.window:
                self.ring[:] = x[-self.window:]
            else:
                self.ring[:-n] = self.ring[n:]
                self.ring[-n:] = x
            self.filled = min(self.filled + n, self.window)
            self.fresh += n
            if self.fresh >= self.hop and self.filled == self.window:
                self.wake.set()

    def current(self):
        with self.lock:
            if self.label is None or time.perf_counter() - self.label_time > CLASS_HOLD_SEC:
                return None, 0.0
            return self.label, self.score

    def _run(self):
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            os.environ.setdefault("TFHUB_CACHE_DIR", TFHUB_CACHE_DIR)
            import tensorflow as tf
            import tensorflow_hub as hub

            tf.config.threading.set_intra_op_parallelism_threads(2)
            print("YAMNet: loading", YAMNET_MODEL)
            model = hub.load(YAMNET_MODEL)
            with tf.io.gfile.GFile(model.class_map_path().numpy().decode("utf-8")) as f:
                names = [row["display_name"] for row in csv.DictReader(f)]
            model(np.zeros(self.window, dtype=np.float32))
            print(f"YAMNet: ready ({len(names)} classes)")
        except Exception as e:
            print("YAMNet unavailable, no class box:", e)
            return

        ema = None
        while not self.stop_flag.is_set():
            if not self.wake.wait(0.5):
                continue
            with self.lock:
                self.wake.clear()
                self.fresh = 0
                x = self.ring.copy()

            x -= x.mean()

            recent = x[-self.hop:]
            if np.abs(recent - recent.mean()).mean() < self.level_threshold:
                ema = None
                self._clear()
                continue

            x *= self.gain / 32768.0
            np.clip(x, -1.0, 1.0, out=x)

            scores, _, _ = model(x)
            s = scores.numpy().mean(axis=0)
            ema = s if ema is None else CLASS_SMOOTHING * ema + (1.0 - CLASS_SMOOTHING) * s

            top = int(np.argmax(ema))
            name, score = names[top], float(ema[top])
            if name in CLASS_HIDE:
                self._clear()
                continue
            if score < CLASS_MIN_SCORE:
                continue
            if CLASS_SHORT_NAMES:
                name = name.split(",")[0].strip()

            with self.lock:
                changed = name != self.label
                self.label, self.score = name, score
                self.label_time = time.perf_counter()

            if CLASS_PRINT and changed:
                best = np.argsort(ema)[::-1][:3]
                print("class:", " | ".join(f"{names[i]} {ema[i]:.2f}" for i in best))

    def _clear(self):
        with self.lock:
            changed = self.label is not None
            self.label = None
        if CLASS_PRINT and changed:
            print("class: (no sound)")


class ClassBox:

    def __init__(self):
        self.fonts = {}
        self.cache_key = None
        self.cache_img = None

    def _font(self, size):
        if size not in self.fonts:
            self.fonts[size] = pygame.font.Font(None, size)
        return self.fonts[size]

    def _render(self, text):
        if text != self.cache_key:
            size = CLASS_FONT_SIZE
            while True:
                img = self._font(size).render(text, True, TEXT_COLOR)
                if img.get_width() + 2 * CLASS_BOX_PAD_X <= CLASS_BOX_MAX_W or size <= CLASS_FONT_MIN:
                    break
                size -= 6
            self.cache_key, self.cache_img = text, img
        return self.cache_img

    def draw(self, surface, label, score):
        text = f"{label}  {score * 100:0.0f}%" if CLASS_SHOW_SCORE else label
        img = self._render(text)
        w = min(max(img.get_width() + 2 * CLASS_BOX_PAD_X, CLASS_BOX_MIN_W), CLASS_BOX_MAX_W)
        h = img.get_height() + 2 * CLASS_BOX_PAD_Y
        box = pygame.Rect(0, 0, w, h)
        box.center = (CENTER[0], CENTER[1] + CLASS_BOX_Y_OFFSET)
        pygame.draw.rect(surface, CLASS_BOX_FILL, box, border_radius=CLASS_BOX_RADIUS)
        pygame.draw.rect(surface, CLASS_BOX_BORDER, box, CLASS_BOX_BORDER_W, border_radius=CLASS_BOX_RADIUS)
        surface.blit(img, img.get_rect(center=box.center))


def draw_gradient_arc(surface, angle_rad):
    start = angle_rad - ARC_SPAN / 2
    end   = angle_rad + ARC_SPAN / 2

    r_in = RADIUS - THICKNESS / 2
    r_out = RADIUS + THICKNESS / 2
    color = (*ARC_COLOR, 255)

    for i in range(RESOLUTION):
        t1 = start + (end - start) * i / RESOLUTION
        t2 = start + (end - start) * (i + 1) / RESOLUTION
        c1, s1 = math.cos(t1), math.sin(t1)
        c2, s2 = math.cos(t2), math.sin(t2)

        pts = [
            (CENTER[0] + r_in * c1,  CENTER[1] - r_in * s1),
            (CENTER[0] + r_out * c1, CENTER[1] - r_out * s1),
            (CENTER[0] + r_out * c2, CENTER[1] - r_out * s2),
            (CENTER[0] + r_in * c2,  CENTER[1] - r_in * s2),
        ]
        pygame.draw.polygon(surface, color, pts)

    for t in (start, end):
        cx = CENTER[0] + RADIUS * math.cos(t)
        cy = CENTER[1] - RADIUS * math.sin(t)
        pygame.draw.circle(surface, color, (round(cx), round(cy)), THICKNESS // 2)


def pop_serial_items(buf: bytearray):
    items = []
    while buf:
        if buf.startswith(AUDIO_MAGIC):
            if len(buf) < AUDIO_HDR_LEN:
                break
            seq, n = struct.unpack_from("<HH", buf, 4)
            if n == 0 or n > AUDIO_MAX_SAMPLES:
                del buf[0]
                continue
            total = AUDIO_HDR_LEN + 2 * n + 2
            if len(buf) < total:
                break
            (chk,) = struct.unpack_from("<H", buf, total - 2)
            if (sum(buf[4:total - 2]) & 0xFFFF) != chk:
                del buf[0]
                continue
            pcm = np.frombuffer(bytes(buf[AUDIO_HDR_LEN:total - 2]), dtype="<i2")
            del buf[:total]
            items.append(("audio", (seq, pcm)))
            continue

        nl_at = buf.find(b"\n")
        magic_at = buf.find(AUDIO_MAGIC)
        if nl_at != -1 and (magic_at == -1 or nl_at < magic_at):
            items.append(("line", buf[:nl_at].decode("utf-8", errors="ignore").strip()))
            del buf[:nl_at + 1]
        elif magic_at > 0:
            del buf[:magic_at]
        else:
            if len(buf) > MAX_TEXT_LINE:
                del buf[:-(len(AUDIO_MAGIC) - 1)]
            break
    return items


def parse_line(line: str):
    if not line:
        return None, None

    tag = line[0].lower()
    if tag not in ("o", "g", "s"):
        return None, None

    mode = {"o": "OUT", "g": "GCC", "s": "SRP"}[tag]

    try:
        ang = float(line[1:].strip())
        if not math.isfinite(ang):
            return None, None
        ang %= 360.0
        return mode, ang
    except ValueError:
        return None, None


def map_angle_deg_to_arc_rad(angle_deg):
    deg = angle_deg
    if ANGLE_INVERT:
        deg = (-deg) % 360.0
    deg = (deg + ANGLE_OFFSET_DEG) % 360.0
    return math.radians(deg)


def main():
    ser = None
    if not FAKE_SPEAKER:
        ser = serial.Serial(
            port=PORT,
            baudrate=BAUD,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.0,
        )
        if hasattr(ser, "set_buffer_size"):
            ser.set_buffer_size(rx_size=SERIAL_RX_BUFFER)
        ser.reset_input_buffer()

    classifier = None
    mic_stream = None
    if SOUND_CLASS_ENABLED and SOUND_CLASS_USE_YAMNET and not FAKE_SPEAKER:
        classifier = SoundClassifier(AUDIO_GAIN, AUDIO_LEVEL_THRESHOLD)
        classifier.start()
    elif SOUND_CLASS_ENABLED and SOUND_CLASS_USE_YAMNET:
        try:
            import sounddevice as sd
            mic_classifier = SoundClassifier(PC_MIC_GAIN, PC_MIC_LEVEL_THRESHOLD)
            mic_stream = sd.InputStream(
                samplerate=AUDIO_RATE, channels=1, dtype="int16", blocksize=512,
                callback=lambda indata, frames, t, status: mic_classifier.push(indata[:, 0]),
            )
            mic_stream.start()
            mic_classifier.start()
            classifier = mic_classifier
        except Exception as e:
            print("PC microphone unavailable, no class box:", e)
            mic_stream = None

    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Arc UI + Serial Angle")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, FONT_SIZE)
    class_box = ClassBox()

    ui_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    latest_mode = "SRP"
    latest_angle_deg = 0.0
    last_update = time.perf_counter()
    last_sound = last_update

    hover = Hover()
    glitch = Glitch()
    steer_deg = FAKE_ANGLE_DEG
    steer_dir = -1
    rx_buf = bytearray()

    audio_seq = None
    audio_lost = 0
    audio_lost_report = time.perf_counter()

    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                steer_dir = -steer_dir

        if FAKE_SPEAKER:
            latest_mode = FAKE_MODE_LABEL
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                steer_deg += steer_dir * STEER_SPEED_DEG * dt
            steer_deg %= 360.0
            latest_angle_deg = (steer_deg + hover.update(dt) + glitch.update(dt)) % 360.0
            last_update = time.perf_counter()
        else:
            try:
                n = ser.in_waiting
                if n:
                    rx_buf += ser.read(n)

                    for kind, value in pop_serial_items(rx_buf):
                        if kind == "audio":
                            seq, pcm = value
                            if audio_seq is not None:
                                audio_lost += (seq - audio_seq - 1) & 0xFFFF
                            audio_seq = seq
                            if classifier is not None:
                                classifier.push(pcm)
                            continue

                        mode, ang = parse_line(value)
                        if ang is not None:
                            last_sound = time.perf_counter()
                        if mode == "OUT":
                            latest_angle_deg = ang
                            last_update = last_sound
            except serial.SerialException as e:
                print("Serial error:", e)

            if classifier is not None and audio_lost and time.perf_counter() - audio_lost_report > 2.0:
                print(f"audio: {audio_lost} packets lost (check BAUD / USB)")
                audio_lost = 0
                audio_lost_report = time.perf_counter()

        now = time.perf_counter()
        stale = (now - last_update) > STALE_AFTER_SEC

        screen.fill(BG_COLOR)

        ui_surface.fill((0, 0, 0, 0))

        if not stale:
            angle_rad = map_angle_deg_to_arc_rad(latest_angle_deg)
            draw_gradient_arc(ui_surface, angle_rad)

        sound_present = FAKE_SPEAKER or (now - last_sound) <= STALE_AFTER_SEC
        if not SOUND_CLASS_ENABLED or not sound_present:
            class_label, class_score = None, 0.0
        elif classifier is not None:
            class_label, class_score = classifier.current()
        elif SOUND_CLASS_USE_YAMNET:
            class_label, class_score = None, 0.0
        else:
            class_label, class_score = SOUND_CLASS_MANUAL_LABEL, 1.0
        if class_label:
            class_box.draw(ui_surface, class_label, class_score)

        label2 = f"{latest_angle_deg:0.1f}°"

        t2 = font.render(label2, True, TEXT_COLOR)

        ui_surface.blit(t2, t2.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 160)))

        if MIRROR_UI:
            ui_to_draw = pygame.transform.flip(ui_surface, True, False)
        else:
            ui_to_draw = ui_surface

        screen.blit(ui_to_draw, (SHIFT_X, SHIFT_Y))

        pygame.display.flip()

    if classifier is not None:
        classifier.stop()
    if mic_stream is not None:
        mic_stream.stop()
        mic_stream.close()
    pygame.quit()
    if ser is not None:
        ser.close()


if __name__ == "__main__":
    main()
