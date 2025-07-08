use crossbeam_channel::{Receiver};
use postgres::{Client, NoTls};
use crate::DataBatch;
use std::sync::{Arc, Mutex};

pub fn start_db_writer(rx: Receiver<DataBatch>, conn_str: String) {
    std::thread::spawn(move || {
        let mut client = match Client::connect(&conn_str, NoTls) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("DB connect failed: {}", e);
                return;
            }
        };
        // Clear any previous run's data
        let _ = client.execute("TRUNCATE TABLE batches CASCADE", &[]);

        let mut insert_count = 0u32;

        for batch in rx.iter() {
            let _ = client.execute(
                "INSERT INTO batches(batch_id, device_name, sample_rate_hz, voltages)
                 VALUES ($1,$2,$3,$4) ON CONFLICT (batch_id) DO NOTHING",
                &[&(batch.batch_id as i32), &"Pico", &200000i32, &batch.voltages],
            );

            insert_count += 1;
            if insert_count % 100 == 0 { // roughly once per second at 60kHz
                let _ = client.execute(
                    "DELETE FROM batches WHERE captured_at < now() - interval '20 seconds'",
                    &[],
                );
            }
        }
    });
}

pub fn truncate_windows(conn_str: &str) {
    if let Ok(mut c) = Client::connect(conn_str, NoTls) {
        let _ = c.execute("TRUNCATE TABLE windows", &[]);
    }
}

pub fn start_window_writer(rx: Receiver<DataBatch>, conn_str: String, collect_flag: Arc<Mutex<bool>>) {
    std::thread::spawn(move || {
        let mut client = match Client::connect(&conn_str, NoTls) {
            Ok(c) => c,
            Err(e) => { eprintln!("Window writer DB connect failed: {}", e); return; }
        };

        let mut buffer: Vec<Vec<f32>> = Vec::with_capacity(500);
        let mut start_id: i32 = 0;
        loop {
            match rx.recv() {
                Ok(batch) => {
                    if !*collect_flag.lock().unwrap() {
                        buffer.clear();
                        continue;
                    }
                    if buffer.is_empty() {
                        start_id = batch.batch_id as i32;
                    }
                    buffer.push(batch.voltages);
                    if buffer.len() == 500 {
                        // concatenate
                        let mut window: Vec<f32> = Vec::with_capacity(1_000_000);
                        for v in &buffer { window.extend_from_slice(v); }
                        let _ = client.execute(
                            "INSERT INTO windows(start_batch_id, voltages) VALUES ($1,$2)",
                            &[&start_id, &window],
                        );
                        buffer.clear();

                        static MAX_WINDOWS: u32 = 180;
                        let mut inserted = client.query_one("SELECT count(*) FROM windows", &[]).unwrap();
                        let total: i64 = inserted.get(0);
                        if total as u32 >= MAX_WINDOWS {
                            *collect_flag.lock().unwrap() = false;
                            println!("Reached {} windows – stopping training collection", MAX_WINDOWS);
                        }
                    }
                }
                Err(_) => break,
            }
        }
    });
} 