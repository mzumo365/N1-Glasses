#include <Arduino.h>
#include <stdint.h>
#include <math.h>
#include "driver/i2s_pdm.h"

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 64
#define SCREEN_HEIGHT 32
#define SDA 8
#define SCL 9
#define OLED_ADDR 0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire);

void displayDir(float mod) {
  display.clearDisplay();
  for (float angle = -23 + mod; angle <= 30 + mod; angle += 1) {
    int cx = 32, cy = 16, r = 14;
    int x = cx + r * cosf(angle * 3.14159f / 180.0f);
    int y = cy - r * sinf(angle * 3.14159f / 180.0f);
    display.drawPixel(x, y, SSD1306_WHITE);
  }
  display.display();
}

#define STAB_HISTORY      15
#define STAB_VAR_THRESH   90.0f
#define STAB_JUMP_THRESH  29.0f

static float angleHist[STAB_HISTORY];
static int   angleHistN = 0;
static int   angleHistIdx = 0;
static float lastStableAngle = 0.0f;
static bool  hasStableAngle = false;

#define SAMPLE_RATE_HZ        16000
#define SERIAL_BAUD           921600
#define FRAMES_PER_READ       512
#define LEVEL_THRESHOLD       20

#define ENABLE_AUDIO_STREAM   1
#define AUDIO_STREAM_MIC      1
#define ENABLE_BEAM_AUDIO     1
#define SERIAL_TX_BUF_SIZE    8192

#if ARDUINO_USB_CDC_ON_BOOT && !ARDUINO_USB_MODE
#define AUDIO_TX_BUFFERED     0
#else
#define AUDIO_TX_BUFFERED     1
#endif

static const float C_SOUND = 343.0f;

#define PRINT_BOTH_ESTIMATES  1

#define ENABLE_ANGLE_CORRECTION  1

#define SRP_COARSE_STEP_DEG   3
#define SRP_FINE_STEP_DEG     1
#define SRP_REFINE_SPAN_DEG   8

#define SRP_ELEV_STEPS        7
#define SRP_ELEV_STEP_DEG     10.0f

#define ENABLE_GCC_ESTIMATE   0

#define GCC_MODE              0
#define GCC_ALPHA             0.7f

#define GCC_BAND_LIMIT        1
#define GCC_FMIN_HZ           200.0f
#define GCC_FMAX_HZ           3500.0f

#define SRP_CORR_EMA          1
#define SRP_EMA_ALPHA         0.70f

#define SRP_MAX_LAG_CAP       32
#define SRP_MAX_LAGS          (2*SRP_MAX_LAG_CAP + 1)

#define ENABLE_BEAM_MASK      1
#define BEAM_MASK_BETA        0.15f
#define BEAM_MASK_MIN         0.85f
#define BEAM_MASK_MAX         1.15f
#define BEAM_MASK_SAMPLES     64

#define MIC_CLK   GPIO_NUM_4
#define MIC0_DIN  GPIO_NUM_5
#define MIC1_DIN  GPIO_NUM_6
#define MIC2_DIN  GPIO_NUM_7
#define MIC3_DIN  GPIO_NUM_15

static const int NUM_CH = 8;
static int16_t sampleBuffer[FRAMES_PER_READ * NUM_CH];
i2s_chan_handle_t rx_handle = nullptr;

static void checkErr(esp_err_t err, const char *msg) {
  if (err != ESP_OK) {
    Serial.print("[I2S ERROR] "); Serial.print(msg);
    Serial.print(" -> 0x"); Serial.println((uint32_t)err, HEX);
    while (true) delay(1000);
  }
}

enum {
  CH_L0 = 0, CH_R0 = 1,
  CH_L1 = 2, CH_R1 = 3,
  CH_L2 = 4, CH_R2 = 5,
  CH_L3 = 6, CH_R3 = 7
};

static const int M = 8;

static const int outIdx[M] = {
  CH_R0,
  CH_R1,
  CH_R2,
  CH_R3,
  CH_L3,
  CH_L2,
  CH_L1,
  CH_L0
};

static const float x_cm[M] = { 10.3f, 5.6f, 1.1f, 0.0f, 0.0f, 1.1f, 5.6f, 10.3f };
static const float y_cm[M] = { 16.0f,16.0f,16.0f,12.6f, 3.6f, 0.0f, 0.0f,  0.0f };

static inline float wrap360(float a) {
  while (a < 0.0f) a += 360.0f;
  while (a >= 360.0f) a -= 360.0f;
  return a;
}

static inline float fclamp(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static inline void fft_bit_reverse(float *re, float *im, int n) {
  int j = 0;
  for (int i = 1; i < n; i++) {
    int bit = n >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) {
      float tr = re[i]; re[i] = re[j]; re[j] = tr;
      float ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
}

static void fft_complex(float *re, float *im, int n, bool inverse) {
  fft_bit_reverse(re, im, n);

  for (int len = 2; len <= n; len <<= 1) {
    float ang = (inverse ? 2.0f : -2.0f) * 3.14159265358979323846f / (float)len;
    float wlen_cos = cosf(ang);
    float wlen_sin = sinf(ang);

    for (int i = 0; i < n; i += len) {
      float w_cos = 1.0f;
      float w_sin = 0.0f;

      int half = len >> 1;
      for (int j = 0; j < half; j++) {
        int u = i + j;
        int v = u + half;

        float vr = re[v] * w_cos - im[v] * w_sin;
        float vi = re[v] * w_sin + im[v] * w_cos;

        float ur = re[u];
        float ui = im[u];

        re[u] = ur + vr;
        im[u] = ui + vi;
        re[v] = ur - vr;
        im[v] = ui - vi;

        float next_cos = w_cos * wlen_cos - w_sin * wlen_sin;
        float next_sin = w_cos * wlen_sin + w_sin * wlen_cos;
        w_cos = next_cos;
        w_sin = next_sin;
      }
    }
  }

  if (inverse) {
    float invN = 1.0f / (float)n;
    for (int i = 0; i < n; i++) { re[i] *= invN; im[i] *= invN; }
  }
}

static float  hannWin[FRAMES_PER_READ];
static int16_t micInt[M][FRAMES_PER_READ];
static float  Xre[M][FRAMES_PER_READ];
static float  Xim[M][FRAMES_PER_READ];
static float  Cre[FRAMES_PER_READ];
static float  Cim[FRAMES_PER_READ];

#if ENABLE_AUDIO_STREAM
static const uint8_t AUDIO_MAGIC[4] = { 0xA5, 0x5A, 0x41, 0x55 };
static uint8_t  audioPkt[4 + 2 + 2 + 2*FRAMES_PER_READ + 2];
static uint16_t audioSeq = 0;

static void sendAudioPacket(const int16_t *pcm, int n) {
  int pos = 0;
  for (int i = 0; i < 4; ++i) audioPkt[pos++] = AUDIO_MAGIC[i];
  audioPkt[pos++] = (uint8_t)(audioSeq & 0xFF);
  audioPkt[pos++] = (uint8_t)(audioSeq >> 8);
  audioPkt[pos++] = (uint8_t)(n & 0xFF);
  audioPkt[pos++] = (uint8_t)(n >> 8);
  for (int i = 0; i < n; ++i) {
    uint16_t s = (uint16_t)pcm[i];
    audioPkt[pos++] = (uint8_t)(s & 0xFF);
    audioPkt[pos++] = (uint8_t)(s >> 8);
  }
  uint16_t sum = 0;
  for (int i = 4; i < pos; ++i) sum += audioPkt[i];
  audioPkt[pos++] = (uint8_t)(sum & 0xFF);
  audioPkt[pos++] = (uint8_t)(sum >> 8);

#if AUDIO_TX_BUFFERED
  if (Serial.availableForWrite() < pos) { audioSeq++; return; }
#endif
  Serial.write(audioPkt, pos);
  audioSeq++;
}

#if ENABLE_BEAM_AUDIO
#define BEAM_HIST 16
static int16_t beamHist[M][BEAM_HIST];
static int16_t beamOut[FRAMES_PER_READ];
static int     beamLag[M];

static void setAudioSteer(float steerDeg) {
  float phi = steerDeg * 3.14159265358979323846f / 180.0f;
  float ux = sinf(phi);
  float uy = cosf(phi);
  float dot[M];
  float maxDot = -1e9f;
  for (int k = 0; k < M; ++k) {
    dot[k] = (x_cm[k]*ux + y_cm[k]*uy) * 0.01f;
    if (dot[k] > maxDot) maxDot = dot[k];
  }
  for (int k = 0; k < M; ++k) {
    int lag = (int)lroundf((maxDot - dot[k]) * (float)SAMPLE_RATE_HZ / C_SOUND);
    if (lag > BEAM_HIST) lag = BEAM_HIST;
    beamLag[k] = lag;
  }
}

static const int16_t *beamformBlock(int n) {
  for (int i = 0; i < n; ++i) {
    int32_t acc = 0;
    for (int k = 0; k < M; ++k) {
      int j = i - beamLag[k];
      acc += (j >= 0) ? micInt[k][j] : beamHist[k][BEAM_HIST + j];
    }
    beamOut[i] = (int16_t)(acc / M);
  }
  for (int k = 0; k < M; ++k) {
    for (int h = 0; h < BEAM_HIST; ++h) {
      int j = n - BEAM_HIST + h;
      beamHist[k][h] = (j >= 0) ? micInt[k][j] : beamHist[k][h + n];
    }
  }
  return beamOut;
}
#endif
#endif

#if ENABLE_BEAM_MASK
static float BeamRe[FRAMES_PER_READ];
static float BeamIm[FRAMES_PER_READ];
static float BeamMagHalf[(FRAMES_PER_READ/2) + 1];
static float BeamMask[FRAMES_PER_READ];
static int   BeamDelaySamp[M];
static float g_lastSteerDeg = 0.0f;
#endif

static const int P = 12;
static const uint8_t pairA[P] = { 0, 7, 0, 7, 0, 3, 1, 2, 3, 4, 6, 5 };
static const uint8_t pairB[P] = { 4, 3, 3, 4, 7, 4, 6, 5, 4, 5, 1, 2 };

static float pDx[P], pDy[P], pDist2[P];
static int g_maxLag = 12;
static float elevShrink[SRP_ELEV_STEPS];

static float corrAbs[P][SRP_MAX_LAGS];
#if SRP_CORR_EMA
static float corrAbsEma[P][SRP_MAX_LAGS];
#endif

static void computeMaxLagAndBaselines() {
  float maxDist = 0.0f;
  for (int p = 0; p < P; ++p) {
    int a = pairA[p], b = pairB[p];
    float dx = (x_cm[b] - x_cm[a]) * 0.01f;
    float dy = (y_cm[b] - y_cm[a]) * 0.01f;
    pDx[p] = dx;
    pDy[p] = dy;
    float d2 = dx*dx + dy*dy;
    pDist2[p] = d2;
    float d = sqrtf(d2);
    if (d > maxDist) maxDist = d;
  }

  int lag = (int)ceilf(maxDist * (float)SAMPLE_RATE_HZ / C_SOUND) + 2;
  if (lag < 2) lag = 2;
  if (lag > SRP_MAX_LAG_CAP) lag = SRP_MAX_LAG_CAP;
  g_maxLag = lag;

  for (int e = 0; e < SRP_ELEV_STEPS; ++e)
    elevShrink[e] = cosf((float)e * SRP_ELEV_STEP_DEG * 3.14159265358979323846f / 180.0f);
}

static inline int lagToIndex(int lag, int N) {
  return (lag >= 0) ? lag : (N + lag);
}

static inline float corrAtLag(const float *win, int lag) {
  if (lag < -g_maxLag) lag = -g_maxLag;
  if (lag >  g_maxLag) lag =  g_maxLag;
  return win[lag + g_maxLag];
}

static float interpCorrAbsAtLag(const float *win, float lagPred) {
  if (lagPred < -(float)g_maxLag || lagPred > (float)g_maxLag) return 0.0f;

  int i1 = (int)floorf(lagPred);
  float t = lagPred - (float)i1;

  float p0 = corrAtLag(win, i1 - 1);
  float p1 = corrAtLag(win, i1);
  float p2 = corrAtLag(win, i1 + 1);
  float p3 = corrAtLag(win, i1 + 2);

  return 0.5f * (2.0f*p1 + (p2 - p0)*t
                 + (2.0f*p0 - 5.0f*p1 + 4.0f*p2 - p3)*t*t
                 + (3.0f*p1 - p0 - 3.0f*p2 + p3)*t*t*t);
}

#if ENABLE_BEAM_MASK
static void buildBeamMask(float steerDeg, int framesRead) {
  float phi = steerDeg * 3.14159265358979323846f / 180.0f;
  float ux = sinf(phi);
  float uy = cosf(phi);

  const int REF = 1;

  float refDot = (x_cm[REF]*0.01f)*ux + (y_cm[REF]*0.01f)*uy;

  for (int k = 0; k < M; ++k) {
    float dot = (x_cm[k]*0.01f)*ux + (y_cm[k]*0.01f)*uy;
    float dt  = (dot - refDot) / C_SOUND;
    int dSamp = (int)lroundf(dt * (float)SAMPLE_RATE_HZ);
    if (dSamp >  SRP_MAX_LAG_CAP) dSamp =  SRP_MAX_LAG_CAP;
    if (dSamp < -SRP_MAX_LAG_CAP) dSamp = -SRP_MAX_LAG_CAP;
    BeamDelaySamp[k] = dSamp;
  }

  for (int n = 0; n < FRAMES_PER_READ; ++n) {
    int32_t acc = 0;
    int32_t count = 0;

    for (int k = 0; k < M; ++k) {
      int idx = n + BeamDelaySamp[k];
      if (idx >= 0 && idx < framesRead) {
        acc += (int32_t)micInt[k][idx];
        count++;
      }
    }

    float bf = (count > 0) ? ((float)acc / (float)count) : 0.0f;
    BeamRe[n] = bf * hannWin[n];
    BeamIm[n] = 0.0f;
  }

  fft_complex(BeamRe, BeamIm, FRAMES_PER_READ, false);

  for (int k = 0; k <= (FRAMES_PER_READ/2); ++k) {
    float re = BeamRe[k], im = BeamIm[k];
    BeamMagHalf[k] = sqrtf(re*re + im*im);
  }

  float samp[BEAM_MASK_SAMPLES];
  int step = (FRAMES_PER_READ/2) / BEAM_MASK_SAMPLES;
  if (step < 1) step = 1;

  int si = 0;
  for (int k = step; k < (FRAMES_PER_READ/2) && si < BEAM_MASK_SAMPLES; k += step) {
    samp[si++] = BeamMagHalf[k];
  }
  while (si < BEAM_MASK_SAMPLES) samp[si++] = samp[si-1];

  for (int i = 1; i < BEAM_MASK_SAMPLES; ++i) {
    float key = samp[i];
    int j = i - 1;
    while (j >= 0 && samp[j] > key) { samp[j+1] = samp[j]; j--; }
    samp[j+1] = key;
  }
  float med = samp[BEAM_MASK_SAMPLES/2];
  const float eps = 1e-9f;
  if (med < eps) med = eps;

  for (int k = 0; k < FRAMES_PER_READ; ++k) {
    int km = (k <= (FRAMES_PER_READ/2)) ? k : (FRAMES_PER_READ - k);
    float mag = BeamMagHalf[km];
    float w = 1.0f + BEAM_MASK_BETA * ((mag - med) / (med + eps));
    BeamMask[k] = fclamp(w, BEAM_MASK_MIN, BEAM_MASK_MAX);
  }

  BeamMask[0] = 1.0f;
}
#endif

static void buildPairCorrAbsWindow(int p, int a, int b, int N) {
  const float eps = 1e-9f;
  const float fs  = (float)SAMPLE_RATE_HZ;

  for (int k = 0; k < N; ++k) {
#if GCC_BAND_LIMIT
    float fk = (k <= (N/2)) ? (fs * (float)k / (float)N)
                            : (fs * (float)(N - k) / (float)N);
    if (fk < GCC_FMIN_HZ || fk > GCC_FMAX_HZ || k == 0) {
      Cre[k] = 0.0f; Cim[k] = 0.0f;
      continue;
    }
#endif

    float Ar = Xre[a][k], Ai = Xim[a][k];
    float Br = Xre[b][k], Bi = Xim[b][k];

    float cr = Ar * Br + Ai * Bi;
    float ci = Ai * Br - Ar * Bi;

#if (GCC_MODE == 0)
    float denom = sqrtf(cr*cr + ci*ci) + eps;
#elif (GCC_MODE == 1)
    float denom = powf(sqrtf(cr*cr + ci*ci) + eps, GCC_ALPHA);
#endif

#if ENABLE_BEAM_MASK
    float w = BeamMask[k];
    Cre[k] = (w * cr) / denom;
    Cim[k] = (w * ci) / denom;
#else
    Cre[k] = cr / denom;
    Cim[k] = ci / denom;
#endif
  }

  fft_complex(Cre, Cim, N, true);

  for (int lag = -g_maxLag; lag <= g_maxLag; ++lag) {
    int idx = lagToIndex(lag, N);
    float v = fabsf(Cre[idx]);

#if SRP_CORR_EMA
    float prev = corrAbsEma[p][lag + g_maxLag];
    float sm = SRP_EMA_ALPHA * prev + (1.0f - SRP_EMA_ALPHA) * v;
    corrAbsEma[p][lag + g_maxLag] = sm;
    corrAbs[p][lag + g_maxLag] = sm;
#else
    corrAbs[p][lag + g_maxLag] = v;
#endif
  }
}

static float srpScore(float phiDeg, float shrink) {
  float phi = phiDeg * 3.14159265358979323846f / 180.0f;
  float ux = sinf(phi);
  float uy = cosf(phi);

  float score = 0.0f;
  for (int p = 0; p < P; ++p) {
    float dot = pDx[p]*ux + pDy[p]*uy;
    float lagPred = -(dot * (float)SAMPLE_RATE_HZ / C_SOUND) * shrink;
    score += pDist2[p] * interpCorrAbsAtLag(corrAbs[p], lagPred);
  }
  return score;
}

static float srpPhatEstimateDeg() {
  float bestScore = -1e30f;
  float bestPhi = 0.0f;
  float bestShrink = 1.0f;

  float lag0[P];
  for (int phiDeg = 0; phiDeg < 360; phiDeg += SRP_COARSE_STEP_DEG) {
    float phi = (float)phiDeg * 3.14159265358979323846f / 180.0f;
    float ux = sinf(phi);
    float uy = cosf(phi);
    for (int p = 0; p < P; ++p)
      lag0[p] = -((pDx[p]*ux + pDy[p]*uy) * (float)SAMPLE_RATE_HZ / C_SOUND);

    for (int e = 0; e < SRP_ELEV_STEPS; ++e) {
      float score = 0.0f;
      for (int p = 0; p < P; ++p)
        score += pDist2[p] * interpCorrAbsAtLag(corrAbs[p], lag0[p] * elevShrink[e]);

      if (score > bestScore) { bestScore = score; bestPhi = (float)phiDeg; bestShrink = elevShrink[e]; }
    }
  }

  const int nFine = 2 * SRP_REFINE_SPAN_DEG / SRP_FINE_STEP_DEG + 1;
  float fineScore[nFine];
  int j = 0;
  for (int i = 0; i < nFine; ++i) {
    float d = (float)(-SRP_REFINE_SPAN_DEG + i * SRP_FINE_STEP_DEG);
    fineScore[i] = srpScore(wrap360(bestPhi + d), bestShrink);
    if (fineScore[i] > fineScore[j]) j = i;
  }

  float delta = 0.0f;
  if (j > 0 && j < nFine - 1) {
    float y1 = fineScore[j - 1], y2 = fineScore[j], y3 = fineScore[j + 1];
    float denom = (y1 - 2.0f*y2 + y3);
    if (fabsf(denom) > 1e-12f) {
      delta = 0.5f * (y1 - y3) / denom;
      if (delta > 0.5f) delta = 0.5f;
      if (delta < -0.5f) delta = -0.5f;
    }
  }

  return wrap360(bestPhi + (float)(-SRP_REFINE_SPAN_DEG + j * SRP_FINE_STEP_DEG) + delta * (float)SRP_FINE_STEP_DEG);
}

#if ENABLE_GCC_ESTIMATE
static float peakLagFromCorr(const float *corrRe, int N, int maxLag, float *peakValOut) {
  int bestLag = 0;
  float bestVal = -1.0f;

  for (int lag = -maxLag; lag <= maxLag; ++lag) {
    int idx = (lag >= 0) ? lag : (N + lag);
    float v = fabsf(corrRe[idx]);
    if (v > bestVal) { bestVal = v; bestLag = lag; }
  }

  float lagF = (float)bestLag;

  if (bestLag > -maxLag && bestLag < maxLag) {
    int lagL = bestLag - 1;
    int lagR = bestLag + 1;
    int idxL = (lagL >= 0) ? lagL : (N + lagL);
    int idxC = (bestLag >= 0) ? bestLag : (N + bestLag);
    int idxR = (lagR >= 0) ? lagR : (N + lagR);

    float y1 = fabsf(corrRe[idxL]);
    float y2 = fabsf(corrRe[idxC]);
    float y3 = fabsf(corrRe[idxR]);

    float denom = (y1 - 2.0f*y2 + y3);
    if (fabsf(denom) > 1e-12f) {
      float delta = 0.5f * (y1 - y3) / denom;
      if (delta > 0.5f) delta = 0.5f;
      if (delta < -0.5f) delta = -0.5f;
      lagF += delta;
    }
  }

  if (peakValOut) *peakValOut = bestVal;
  return lagF;
}

static float gccPhatLagFromFFTs(int ref, int sig, int N, int maxLag, float *peakValOut) {
  const float eps = 1e-9f;
  const float fs  = (float)SAMPLE_RATE_HZ;

  for (int k = 0; k < N; k++) {
#if GCC_BAND_LIMIT
    float fk = (k <= (N/2)) ? (fs * (float)k / (float)N)
                            : (fs * (float)(N - k) / (float)N);
    if (fk < GCC_FMIN_HZ || fk > GCC_FMAX_HZ || k == 0) {
      Cre[k] = 0.0f; Cim[k] = 0.0f;
      continue;
    }
#endif
    float Rr = Xre[ref][k], Ri = Xim[ref][k];
    float Sr = Xre[sig][k], Si = Xim[sig][k];

    float cr = Rr*Sr + Ri*Si;
    float ci = Ri*Sr - Rr*Si;

#if (GCC_MODE == 0)
    float denom = sqrtf(cr*cr + ci*ci) + eps;
#elif (GCC_MODE == 1)
    float denom = powf(sqrtf(cr*cr + ci*ci) + eps, GCC_ALPHA);
#endif

#if ENABLE_BEAM_MASK
    float w = BeamMask[k];
    Cre[k] = (w * cr) / denom;
    Cim[k] = (w * ci) / denom;
#else
    Cre[k] = cr / denom;
    Cim[k] = ci / denom;
#endif
  }

  fft_complex(Cre, Cim, N, true);
  return peakLagFromCorr(Cre, N, maxLag, peakValOut);
}

static bool gccPhatDoaLeastSquares(int ref, float &degOut) {
  float A00 = 0, A01 = 0, A11 = 0;
  float b0  = 0, b1  = 0;

  for (int i = 0; i < M; ++i) {
    if (i == ref) continue;

    float dx = (x_cm[i] - x_cm[ref]) * 0.01f;
    float dy = (y_cm[i] - y_cm[ref]) * 0.01f;

    float d2 = dx*dx + dy*dy;
    if (d2 < 1e-9f) continue;

    float dist = sqrtf(d2);
    int maxLag = (int)ceilf(dist * (float)SAMPLE_RATE_HZ / C_SOUND) + 2;
    if (maxLag < 2) maxLag = 2;
    if (maxLag > 32) maxLag = 32;
    if (maxLag > (FRAMES_PER_READ - 1)) maxLag = FRAMES_PER_READ - 1;

    float peakVal = 0.0f;
    float lag = gccPhatLagFromFFTs(ref, i, FRAMES_PER_READ, maxLag, &peakVal);
    float dt  = lag / (float)SAMPLE_RATE_HZ;

    float rhs = -C_SOUND * dt;

    float w = d2 * (0.2f + peakVal);
    A00 += w * dx * dx;
    A01 += w * dx * dy;
    A11 += w * dy * dy;
    b0  += w * dx * rhs;
    b1  += w * dy * rhs;
  }

  float det = A00*A11 - A01*A01;
  if (fabsf(det) < 1e-9f) return false;

  float ux = ( b0*A11 - b1*A01) / det;
  float uy = ( A00*b1 - A01*b0) / det;

  float n = sqrtf(ux*ux + uy*uy);
  if (n < 1e-6f) return false;

  ux /= n; uy /= n;

  float phi = atan2f(ux, uy) * 180.0f / 3.14159265358979323846f;
  if (phi < 0) phi += 360.0f;
  degOut = phi;
  return true;
}
#endif

void setup() {
#if ENABLE_AUDIO_STREAM && AUDIO_TX_BUFFERED
  Serial.setTxBufferSize(SERIAL_TX_BUF_SIZE);
#endif
  Serial.begin(SERIAL_BAUD);
  delay(500);

  for (int i = 0; i < FRAMES_PER_READ; i++) {
    float a = 2.0f * 3.14159265358979323846f * (float)i / (float)(FRAMES_PER_READ - 1);
    hannWin[i] = 0.5f * (1.0f - cosf(a));
  }

  computeMaxLagAndBaselines();

#if SRP_CORR_EMA
  for (int p = 0; p < P; ++p)
    for (int i = 0; i < SRP_MAX_LAGS; ++i)
      corrAbsEma[p][i] = 0.0f;
#endif

#if ENABLE_BEAM_MASK
  for (int k = 0; k < FRAMES_PER_READ; ++k) BeamMask[k] = 1.0f;
  g_lastSteerDeg = 0.0f;
#endif

  Wire.begin(SDA, SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println(F("SSD1306 allocation failed"));
    for (;;) delay(1000);
  }
  display.clearDisplay();
  display.drawRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, SSD1306_WHITE);
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(2, 10);
  display.print("OLED OK");
  display.display();

  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  checkErr(i2s_new_channel(&chan_cfg, NULL, &rx_handle), "i2s_new_channel");

  i2s_pdm_rx_config_t pdm_rx_cfg = {
    .clk_cfg  = I2S_PDM_RX_CLK_DEFAULT_CONFIG(SAMPLE_RATE_HZ),
    .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
    .gpio_cfg = {
      .clk  = MIC_CLK,
      .dins = { MIC0_DIN, MIC1_DIN, MIC2_DIN, MIC3_DIN },
      .invert_flags = { .clk_inv = false },
    },
  };

  pdm_rx_cfg.slot_cfg.slot_mask =
    (i2s_pdm_slot_mask_t)(
      I2S_PDM_RX_LINE0_SLOT_LEFT  | I2S_PDM_RX_LINE0_SLOT_RIGHT |
      I2S_PDM_RX_LINE1_SLOT_LEFT  | I2S_PDM_RX_LINE1_SLOT_RIGHT |
      I2S_PDM_RX_LINE2_SLOT_LEFT  | I2S_PDM_RX_LINE2_SLOT_RIGHT |
      I2S_PDM_RX_LINE3_SLOT_LEFT  | I2S_PDM_RX_LINE3_SLOT_RIGHT
    );

  checkErr(i2s_channel_init_pdm_rx_mode(rx_handle, &pdm_rx_cfg), "i2s_channel_init_pdm_rx_mode");
  checkErr(i2s_channel_enable(rx_handle), "i2s_channel_enable");

  Serial.print("Ready. SRP pairs="); Serial.print(P);
  Serial.print(" maxLag="); Serial.println(g_maxLag);
#if ENABLE_BEAM_MASK
  Serial.println("BeamMask: ON (tiny weighting)");
#endif
}

static void angleHistStats(float &meanOut, float &stdOut) {
  int n = (angleHistN < STAB_HISTORY) ? angleHistN : STAB_HISTORY;
  if (n == 0) { meanOut = 0; stdOut = 999; return; }

  float sx = 0, sy = 0;
  for (int i = 0; i < n; i++) {
    float rad = angleHist[i] * 3.14159265f / 180.0f;
    sx += cosf(rad);
    sy += sinf(rad);
  }
  float mx = sx / n;
  float my = sy / n;
  float meanRad = atan2f(my, mx);
  meanOut = meanRad * 180.0f / 3.14159265f;
  if (meanOut < 0) meanOut += 360.0f;

  float R = sqrtf(mx * mx + my * my);
  if (R > 0.9999f) R = 0.9999f;
  if (R < 0.0001f) { stdOut = 999; return; }
  stdOut = sqrtf(-2.0f * logf(R)) * (180.0f / 3.14159265f);
}

static float angleDiffAbs(float a, float b) {
  float d = fabsf(a - b);
  if (d > 180.0f) d = 360.0f - d;
  return d;
}

void loop() {
  size_t bytesRead = 0;
  esp_err_t err = i2s_channel_read(
    rx_handle, (void*)sampleBuffer, sizeof(sampleBuffer), &bytesRead,
    100 / portTICK_PERIOD_MS
  );
  if (err != ESP_OK || bytesRead == 0) return;

  int framesRead = bytesRead / (sizeof(int16_t) * NUM_CH);
  if (framesRead <= 0) return;
  if (framesRead > FRAMES_PER_READ) framesRead = FRAMES_PER_READ;

  for (int k = 0; k < M; ++k)
    for (int i = 0; i < FRAMES_PER_READ; ++i)
      micInt[k][i] = 0;

  int32_t levelAcc = 0;
  const int LEVEL_MIC = 1;
  for (int i = 0; i < framesRead; ++i) {
    int base = i * NUM_CH;
    for (int k = 0; k < M; ++k) {
      micInt[k][i] = sampleBuffer[base + outIdx[k]];
    }
    int16_t s = micInt[LEVEL_MIC][i];
    levelAcc += (s >= 0) ? s : -(int32_t)s;
  }

#if ENABLE_AUDIO_STREAM
#if ENABLE_BEAM_AUDIO
  sendAudioPacket(beamformBlock(framesRead), framesRead);
#else
  sendAudioPacket(micInt[AUDIO_STREAM_MIC], framesRead);
#endif
#endif

  int32_t meanAbsGate = levelAcc / framesRead;
  if (meanAbsGate < LEVEL_THRESHOLD) return;

  for (int k = 0; k < M; ++k) {
    int32_t sum = 0;
    for (int i = 0; i < framesRead; ++i) sum += micInt[k][i];
    int32_t mean = sum / framesRead;
    for (int i = 0; i < framesRead; ++i) {
      int32_t v = (int32_t)micInt[k][i] - mean;
      if (v >  32767) v =  32767;
      if (v < -32768) v = -32768;
      micInt[k][i] = (int16_t)v;
    }
  }

  for (int k = 0; k < M; ++k) {
    for (int i = 0; i < FRAMES_PER_READ; ++i) {
      float s = (float)micInt[k][i];
      Xre[k][i] = s * hannWin[i];
      Xim[k][i] = 0.0f;
    }
    fft_complex(Xre[k], Xim[k], FRAMES_PER_READ, false);
  }

#if ENABLE_BEAM_MASK
  buildBeamMask(g_lastSteerDeg, framesRead);
#endif

  for (int p = 0; p < P; ++p) {
    buildPairCorrAbsWindow(p, pairA[p], pairB[p], FRAMES_PER_READ);
  }

  float srpDeg = wrap360(srpPhatEstimateDeg());

#if ENABLE_AUDIO_STREAM && ENABLE_BEAM_AUDIO
  setAudioSteer(srpDeg);
#endif

#if ENABLE_GCC_ESTIMATE
  const int GCC_REF = 1;
  float gccDeg = NAN;
  bool gccOk = gccPhatDoaLeastSquares(GCC_REF, gccDeg);
  if (gccOk) gccDeg = wrap360(gccDeg);
#endif

#if ENABLE_ANGLE_CORRECTION
#endif

#if PRINT_BOTH_ESTIMATES
  Serial.print("s"); Serial.println(srpDeg, 1);
#if ENABLE_GCC_ESTIMATE
  Serial.print("g");
  if (gccOk) Serial.println(gccDeg, 1);
  else       Serial.println("nan");
#endif
#endif

  float outDeg = srpDeg;

#if ENABLE_BEAM_MASK
  g_lastSteerDeg = outDeg;
#endif

  angleHist[angleHistIdx] = outDeg;
  angleHistIdx = (angleHistIdx + 1) % STAB_HISTORY;
  if (angleHistN < STAB_HISTORY) angleHistN++;

  float histMean, histStd;
  angleHistStats(histMean, histStd);

  float filteredDeg;
  bool suppress = false;

  if (histStd > STAB_VAR_THRESH) {
    suppress = true;
  } else if (hasStableAngle && angleDiffAbs(outDeg, histMean) > STAB_JUMP_THRESH) {
    filteredDeg = histMean;
  } else {
    filteredDeg = outDeg;
  }

  if (!suppress) {
    lastStableAngle = filteredDeg;
    hasStableAngle = true;

    Serial.print("o"); Serial.println(filteredDeg, 1);
    displayDir(filteredDeg);
  }
}