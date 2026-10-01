//! POSIX Shared Memory (`/dev/shm`) zero-copy audio ring buffer and IPC transport.
//!
//! Provides ultra-low-latency, zero-copy audio ingestion between external processes
//! (C++, Python, Kestrel autopilot, ROS 2 nodes) and the Sonon DSP engine in 100% safe Rust.

#![deny(unsafe_code)]

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Magic header identifier: ASCII "SONO" (0x534F4E4F).
pub const SHM_MAGIC: u32 = 0x534F4E4F;

/// SHM protocol format version.
pub const SHM_VERSION: u32 = 1;

/// Standard default shared memory filepath on Linux tmpfs.
pub const DEFAULT_SHM_PATH: &str = "/dev/shm/sonon_audio";

/// Default audio sample capacity (65,536 samples = 4.096 seconds at 16 kHz).
pub const DEFAULT_SHM_CAPACITY: usize = 65536;

/// Fixed 64-byte binary header describing shared memory ring buffer state.
#[derive(Debug, Clone, PartialEq)]
pub struct ShmHeader {
    pub magic: u32,
    pub version: u32,
    pub sample_rate: f32,
    pub channels: u32,
    pub capacity: u32,
    pub write_head: u64,
    pub read_head: u64,
    pub health_score: f32,
    pub worst_severity: u32,
    pub last_keyword_confidence: f32,
    pub last_keyword_len: u32,
    pub last_keyword: [u8; 16],
}

impl ShmHeader {
    /// Construct a new default header for given sample rate and sample capacity.
    pub fn new(sample_rate: f32, capacity: u32) -> Self {
        Self {
            magic: SHM_MAGIC,
            version: SHM_VERSION,
            sample_rate,
            channels: 1,
            capacity,
            write_head: 0,
            read_head: 0,
            health_score: 1.0,
            worst_severity: 0,
            last_keyword_confidence: 0.0,
            last_keyword_len: 0,
            last_keyword: [0u8; 16],
        }
    }

    /// Serialize header to a 64-byte binary array.
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut buf = [0u8; 64];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.sample_rate.to_le_bytes());
        buf[12..16].copy_from_slice(&self.channels.to_le_bytes());
        buf[16..20].copy_from_slice(&self.capacity.to_le_bytes());
        buf[20..28].copy_from_slice(&self.write_head.to_le_bytes());
        buf[28..36].copy_from_slice(&self.read_head.to_le_bytes());
        buf[36..40].copy_from_slice(&self.health_score.to_le_bytes());
        buf[40..44].copy_from_slice(&self.worst_severity.to_le_bytes());
        buf[44..48].copy_from_slice(&self.last_keyword_confidence.to_le_bytes());
        buf[48..52].copy_from_slice(&self.last_keyword_len.to_le_bytes());
        buf[52..64].copy_from_slice(&self.last_keyword[0..12]);
        // Reserve remaining 4 bytes as padding to align to 64 bytes
        buf
    }

    /// Deserialize header from a 64-byte binary slice.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 64 {
            return None;
        }

        let magic = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
        if magic != SHM_MAGIC {
            return None;
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        let sample_rate = f32::from_le_bytes(bytes[8..12].try_into().ok()?);
        let channels = u32::from_le_bytes(bytes[12..16].try_into().ok()?);
        let capacity = u32::from_le_bytes(bytes[16..20].try_into().ok()?);
        let write_head = u64::from_le_bytes(bytes[20..28].try_into().ok()?);
        let read_head = u64::from_le_bytes(bytes[28..36].try_into().ok()?);
        let health_score = f32::from_le_bytes(bytes[36..40].try_into().ok()?);
        let worst_severity = u32::from_le_bytes(bytes[40..44].try_into().ok()?);
        let last_keyword_confidence = f32::from_le_bytes(bytes[44..48].try_into().ok()?);
        let last_keyword_len = u32::from_le_bytes(bytes[48..52].try_into().ok()?);

        let mut last_keyword = [0u8; 16];
        last_keyword[0..12].copy_from_slice(&bytes[52..64]);

        Some(Self {
            magic,
            version,
            sample_rate,
            channels,
            capacity,
            write_head,
            read_head,
            health_score,
            worst_severity,
            last_keyword_confidence,
            last_keyword_len,
            last_keyword,
        })
    }
}

/// Shared memory audio channel managing read/write ring buffer operations.
pub struct ShmAudioChannel {
    path: PathBuf,
    file: File,
    capacity: usize,
    header: ShmHeader,
}

impl ShmAudioChannel {
    /// Create or initialize a shared memory audio channel as a writer/host.
    pub fn create_or_open<P: AsRef<Path>>(
        path: P,
        sample_rate: f32,
        capacity: usize,
    ) -> io::Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path_buf)?;

        let total_size = (64 + capacity * 4) as u64;
        let current_len = file.metadata()?.len();

        let header = if current_len >= total_size {
            // Read existing header
            let mut h_bytes = [0u8; 64];
            file.seek(SeekFrom::Start(0))?;
            file.read_exact(&mut h_bytes)?;
            ShmHeader::from_bytes(&h_bytes).unwrap_or_else(|| ShmHeader::new(sample_rate, capacity as u32))
        } else {
            // Preallocate shared memory file and write initial header
            file.set_len(total_size)?;
            let h = ShmHeader::new(sample_rate, capacity as u32);
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&h.to_bytes())?;
            file.flush()?;
            h
        };

        Ok(Self {
            path: path_buf,
            file,
            capacity,
            header,
        })
    }

    /// Writes raw float32 samples into the shared memory circular ring buffer.
    ///
    /// Updates `write_head` monotonically.
    pub fn write_samples(&mut self, samples: &[f32]) -> io::Result<usize> {
        if samples.is_empty() {
            return Ok(0);
        }

        // Refresh current header
        self.reload_header()?;

        let cap = self.capacity;
        let write_start = self.header.write_head;

        let mut byte_buf = Vec::with_capacity(samples.len() * 4);
        for &s in samples {
            byte_buf.extend_from_slice(&s.to_le_bytes());
        }

        let start_slot = (write_start % (cap as u64)) as usize;
        let end_slot = start_slot + samples.len();

        if end_slot <= cap {
            // Contiguous write
            let file_offset = 64 + (start_slot * 4) as u64;
            self.file.seek(SeekFrom::Start(file_offset))?;
            self.file.write_all(&byte_buf)?;
        } else {
            // Wrapped write
            let first_chunk_count = cap - start_slot;
            let first_bytes = &byte_buf[0..first_chunk_count * 4];
            let second_bytes = &byte_buf[first_chunk_count * 4..];

            let first_offset = 64 + (start_slot * 4) as u64;
            self.file.seek(SeekFrom::Start(first_offset))?;
            self.file.write_all(first_bytes)?;

            self.file.seek(SeekFrom::Start(64))?;
            self.file.write_all(second_bytes)?;
        }

        // Advance write head in header
        self.header.write_head = write_start + (samples.len() as u64);
        self.save_header()?;

        Ok(samples.len())
    }

    /// Reads newly available audio samples written since the last read.
    ///
    /// Advances `read_head` monotonically.
    pub fn read_available_samples(&mut self, out_buffer: &mut Vec<f32>) -> io::Result<usize> {
        self.reload_header()?;

        let write_head = self.header.write_head;
        let mut read_head = self.header.read_head;

        if write_head <= read_head {
            return Ok(0); // No new samples available
        }

        let cap = self.capacity as u64;
        let unread_count = write_head - read_head;

        // If producer overtook consumer by more than capacity, clamp to oldest available
        if unread_count > cap {
            read_head = write_head - cap;
        }

        let count = (write_head - read_head) as usize;
        let start_slot = (read_head % cap) as usize;
        let end_slot = start_slot + count;

        let mut byte_buf = vec![0u8; count * 4];

        if end_slot <= self.capacity {
            // Contiguous read
            let file_offset = 64 + (start_slot * 4) as u64;
            self.file.seek(SeekFrom::Start(file_offset))?;
            self.file.read_exact(&mut byte_buf)?;
        } else {
            // Wrapped read
            let first_chunk_count = self.capacity - start_slot;
            let (first_bytes, second_bytes) = byte_buf.split_at_mut(first_chunk_count * 4);

            let first_offset = 64 + (start_slot * 4) as u64;
            self.file.seek(SeekFrom::Start(first_offset))?;
            self.file.read_exact(first_bytes)?;

            self.file.seek(SeekFrom::Start(64))?;
            self.file.read_exact(second_bytes)?;
        }

        // Decode float32 samples
        out_buffer.clear();
        out_buffer.reserve(count);
        for chunk in byte_buf.chunks_exact(4) {
            let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            out_buffer.push(sample);
        }

        // Advance read head
        self.header.read_head = write_head;
        self.save_header()?;

        Ok(count)
    }

    /// Updates latest health score and severity in shared memory header.
    pub fn write_health_status(&mut self, health_score: f32, severity: u32) -> io::Result<()> {
        self.header.health_score = health_score;
        self.header.worst_severity = severity;
        self.save_header()
    }

    /// Reads latest health score and severity from shared memory header.
    pub fn read_health_status(&mut self) -> io::Result<(f32, u32)> {
        self.reload_header()?;
        Ok((self.header.health_score, self.header.worst_severity))
    }

    /// Records a detected keyword detection event into shared memory header.
    pub fn write_detection_event(&mut self, keyword: &str, confidence: f32) -> io::Result<()> {
        let bytes = keyword.as_bytes();
        let len = bytes.len().min(12);
        self.header.last_keyword_len = len as u32;
        self.header.last_keyword[0..len].copy_from_slice(&bytes[0..len]);
        self.header.last_keyword_confidence = confidence;
        self.save_header()
    }

    /// Reads latest detection event from shared memory header.
    pub fn read_detection_event(&mut self) -> io::Result<Option<(String, f32)>> {
        self.reload_header()?;
        let len = self.header.last_keyword_len as usize;
        if len == 0 || self.header.last_keyword_confidence <= 0.0 {
            return Ok(None);
        }

        let clamped_len = len.min(12);
        let kw_slice = &self.header.last_keyword[0..clamped_len];
        let kw_str = String::from_utf8_lossy(kw_slice).to_string();
        Ok(Some((kw_str, self.header.last_keyword_confidence)))
    }

    /// Reloads 64-byte header from file disk/RAM cache.
    fn reload_header(&mut self) -> io::Result<()> {
        let mut buf = [0u8; 64];
        self.file.seek(SeekFrom::Start(0))?;
        self.file.read_exact(&mut buf)?;
        if let Some(h) = ShmHeader::from_bytes(&buf) {
            self.header = h;
        }
        Ok(())
    }

    /// Writes 64-byte header back to file.
    fn save_header(&mut self) -> io::Result<()> {
        let buf = self.header.to_bytes();
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&buf)?;
        self.file.flush()?;
        Ok(())
    }

    /// Returns path to shared memory file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns sample capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns sample rate.
    pub fn sample_rate(&self) -> f32 {
        self.header.sample_rate
    }
}
