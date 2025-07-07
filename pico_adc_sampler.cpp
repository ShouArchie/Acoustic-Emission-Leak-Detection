#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>  // memcpy
#include "pico/stdlib.h"
#include "pico/stdio_usb.h"
#include "hardware/adc.h"
#include "hardware/dma.h"
#include "hardware/irq.h"
#include "hardware/clocks.h"
#include "pico/stdio.h"
#include "tusb.h"   // TinyUSB direct API

#define ADC_PIN_NUM 26      // GPIO 26 is ADC 0
#define ADC_CHANNEL 0       // ADC channel for GPIO 26
#define SAMPLE_RATE 200000   // 200kHz sampling rate
#define SAMPLES_PER_BATCH 2000    // 0.01 second worth of samples at 200kHz

volatile uint32_t sample_index = 0;
static uint16_t raw_buffers[2][SAMPLES_PER_BATCH];      // Double buffer for ping-pong DMA
static volatile uint32_t batch_number = 0;

// Pico ADC reference voltage (used only for host-side conversion)
#define ADC_REF_VOLTAGE 3.3f
#define ADC_MAX_VALUE 4095.0f

// Binary packet format:
// [0]   0xA5
// [1]   0x5A
// [2..5] uint32_t batch_id (little-endian)
// [6..]  3000 * uint16_t raw ADC samples
static void send_batch_binary(const uint16_t *samples, uint32_t id) {
    // Build a contiguous packet
    static uint8_t tx_buffer[2 + 4 + SAMPLES_PER_BATCH * 2];
    tx_buffer[0] = 0xA5;
    tx_buffer[1] = 0x5A;
    memcpy(&tx_buffer[2], &id, sizeof(id));
    memcpy(&tx_buffer[6], samples, SAMPLES_PER_BATCH * sizeof(uint16_t));

    // Send via TinyUSB CDC directly (bypass stdio buffering)
    size_t sent = 0;
    while (sent < sizeof(tx_buffer)) {
        // Attempt to write remaining bytes
        uint32_t n = tud_cdc_write(tx_buffer + sent, sizeof(tx_buffer) - sent);
        if (n == 0) {
            // Buffer full or not ready – service USB and retry
            tud_task();
            sleep_us(50);
            continue;
        }
        sent += n;
        tud_cdc_write_flush();
        tud_task();
    }
}

int main() {
    // Initialise USB (stdio+TinyUSB)
    stdio_init_all();

    // Wait until the USB stack is mounted and the host opens the CDC port
    while (!tud_cdc_connected()) {
        tud_task();        // service USB events while waiting
        sleep_ms(10);
    }
    
    printf("=== PICO FAST SAMPLER v4.0 ===\n");
    printf("Config: 200kHz, 2k samples per batch, 0.01s intervals\n");
    printf("GPIO: 26 (ADC0)\n");
    printf("Optimized for speed\n");
    fflush(stdout);
    
    // ---------- ADC CONFIGURATION ----------
    adc_init();
    adc_gpio_init(ADC_PIN_NUM);
    adc_select_input(ADC_CHANNEL);

    // Set ADC clock so that sampling rate is exactly SAMPLE_RATE
    // ADC clock source is 48 MHz; divisor = 48 000 000 / SAMPLE_RATE
    adc_set_clkdiv(48'000'000.0f / SAMPLE_RATE);

    // Configure FIFO: enable, DREQ on 1 sample, no ERR bit, no shift
    adc_fifo_setup(true, true, 1, false, false);
    adc_run(true);

    // ---------- DMA CONFIGURATION ----------
    int dma_chan = dma_claim_unused_channel(true);
    dma_channel_config cfg = dma_channel_get_default_config(dma_chan);
    channel_config_set_transfer_data_size(&cfg, DMA_SIZE_16);
    channel_config_set_read_increment(&cfg, false);   // Always read from ADC FIFO
    channel_config_set_write_increment(&cfg, true);   // Increment destination buffer
    channel_config_set_dreq(&cfg, DREQ_ADC);          // Pace on ADC ready

    printf("Status: Ready for fast sampling...\n");
    fflush(stdout);
    
    // ------------------ DOUBLE-BUFFERED LOOP ------------------
    int fill_idx = 0;   // buffer being filled by DMA
    int send_idx = 1;   // buffer we will transmit next

    // Prime the first DMA transfer so that we always have a buffer to send
    dma_channel_configure(
        dma_chan,
        &cfg,
        raw_buffers[fill_idx],
        &adc_hw->fifo,
        SAMPLES_PER_BATCH,
        true  // start
    );

    while (true) {
        // Wait until the current DMA transfer into raw_buffers[fill_idx] completes
        dma_channel_wait_for_finish_blocking(dma_chan);

        // Immediately kick DMA to the OTHER buffer so sampling continues while we transmit
        dma_channel_configure(
            dma_chan,
            &cfg,
            raw_buffers[send_idx],   // next fill buffer
            &adc_hw->fifo,
            SAMPLES_PER_BATCH,
            true  // start
        );

        // Now transmit the buffer that just finished (fill_idx)
        send_batch_binary(raw_buffers[fill_idx], batch_number++);

        // Swap roles for next iteration
        fill_idx ^= 1;
        send_idx ^= 1;
    }
    
    return 0;
} 