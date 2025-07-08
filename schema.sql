CREATE TABLE IF NOT EXISTS batches (
    id              bigserial PRIMARY KEY,
    batch_id        integer UNIQUE NOT NULL,
    device_name     text    NOT NULL,
    captured_at     timestamptz NOT NULL DEFAULT now(),
    sample_rate_hz  integer NOT NULL,
    voltages        real[]  NOT NULL
);

CREATE TABLE IF NOT EXISTS batch_features (
    batch_id integer PRIMARY KEY REFERENCES batches(batch_id) ON DELETE CASCADE,
    rms      real,
    peak_db  real,
    fft_peaks real[],
    label    text
);

-- 5-second windows for training (1 000 000 samples each)
CREATE TABLE IF NOT EXISTS windows (
    id             bigserial PRIMARY KEY,
    start_batch_id integer      NOT NULL,
    captured_at    timestamptz NOT NULL DEFAULT now(),
    voltages       real[]       NOT NULL,   -- 5-second raw voltage window
    label_normal   boolean      NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_windows_captured_at
    ON windows (captured_at);