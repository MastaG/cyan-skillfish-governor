use super::{DrmTempStrategy, TempStrategy, init_device_handle};
use crate::app_error::{AppError, Result};
use log::warn;
use std::{
    io::Error as IoError,
    path::{Path, PathBuf},
};

/// Anything outside this range is treated as an unusable reading rather than
/// passed on -- a bogus value here would feed straight into thermal throttling.
const TEMP_MIN_MILLIDEGREES: i64 = -40_000;
const TEMP_MAX_MILLIDEGREES: i64 = 150_000;

/// How many reads in a row have to fail before this strategy gives up.
///
/// One failure means nothing: the value only drives thermal throttling, and a
/// reading a few hundred milliseconds stale serves that just as well. Giving up
/// is the expensive outcome, so the strategy then falls back to the DRM ioctl.
const MAX_CONSECUTIVE_FAILURES: u32 = 5;

/// Convert a validated hwmon reading to whole degrees.
///
/// The clamp matters because `read_hwmon_millidegrees` deliberately accepts
/// readings down to `TEMP_MIN_MILLIDEGREES`, and `as u32` on a negative i64
/// wraps: -1000 millidegrees would become 4294967295 rather than 0, which
/// reads as permanently over the throttling threshold. Cyan Skillfish cannot
/// actually report a sub-zero edge temperature -- the firmware field is a
/// `uint16_t` in centi-Celsius -- so this is about the validator and the
/// conversion agreeing, not about a fault seen in the field.
fn millidegrees_to_celsius(millidegrees: i64) -> u32 {
    (millidegrees / 1000).max(0) as u32
}

/// Reads the GPU temperature from the amdgpu hwmon `temp1_input` attribute.
///
/// Opens the attribute per read, so no DRM client is held while sysfs is
/// working. Carries the last good reading so a transient failure can be ridden
/// out, and the render path plus a lazily built `DrmTempStrategy` so it can
/// hand over to the ioctl if sysfs stops working altogether.
pub(super) struct SysfsTempStrategy {
    path: PathBuf,
    drm_render_path: PathBuf,
    fallback: Option<Box<dyn TempStrategy + Send>>,
    last_good: i64,
    failures: u32,
}

impl SysfsTempStrategy {
    /// Locate and probe the attribute, returning an error if it is unusable.
    ///
    /// The hwmon attribute is plain upstream amdgpu -- `temp1_input` is
    /// registered for every ASIC except multi-AID parts, and Cyan Skillfish
    /// already implements `AMDGPU_PP_SENSOR_EDGE_TEMP` in a stock kernel -- so
    /// this normally succeeds. It is probed once anyway rather than trusted, so
    /// a kernel that does not expose it costs a warning instead of a governor
    /// that will not start.
    ///
    /// The probe read is kept rather than discarded: it seeds `last_good`, so a
    /// failure on the very first control cycle already has something sensible
    /// to fall back on.
    pub(super) fn probe(sysfs_path: &Path, drm_render_path: PathBuf) -> Result<Self> {
        let path = find_hwmon_temp_input(sysfs_path)?;
        let last_good = read_hwmon_millidegrees(&path)?;
        Ok(Self {
            path,
            drm_render_path,
            fallback: None,
            last_good,
            failures: 0,
        })
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    /// Build a strategy straight from a known attribute path, skipping the
    /// hwmon lookup, and optionally supply the strategy it hands over to.
    ///
    /// Tests use this to point at a scratch file and to stand in for the DRM
    /// ioctl, which cannot be opened in a unit test.
    #[cfg(test)]
    fn probe_from_path(path: PathBuf, fallback: Option<Box<dyn TempStrategy + Send>>) -> Self {
        let last_good = read_hwmon_millidegrees(&path).expect("test file must be readable");
        Self {
            path,
            drm_render_path: PathBuf::new(),
            fallback,
            last_good,
            failures: 0,
        }
    }
}

impl TempStrategy for SysfsTempStrategy {
    /// A failed read reuses the last good value and is reported only as a
    /// warning. Once `MAX_CONSECUTIVE_FAILURES` reads in a row have failed the
    /// strategy hands over to the DRM ioctl for the rest of the run. A
    /// successful read resets the count, so isolated errors never accumulate
    /// towards that.
    fn read_temperature(&mut self) -> Result<u32> {
        if let Some(fallback) = &mut self.fallback {
            return fallback.read_temperature();
        }

        match read_hwmon_millidegrees(&self.path) {
            Ok(millidegrees) => {
                self.last_good = millidegrees;
                self.failures = 0;
                Ok(millidegrees_to_celsius(millidegrees))
            }
            Err(e) => {
                self.failures += 1;
                if self.failures < MAX_CONSECUTIVE_FAILURES {
                    warn!(
                        "gpu-usage.temp-read = \"sysfs\": {e}; reusing the last good reading ({} failure(s) in a row, giving up at {MAX_CONSECUTIVE_FAILURES})",
                        self.failures
                    );
                    return Ok(millidegrees_to_celsius(self.last_good));
                }
                warn!(
                    "gpu-usage.temp-read = \"sysfs\": {e}; falling back to the DRM ioctl for the rest of this run"
                );
                let mut fallback: Box<dyn TempStrategy + Send> = Box::new(DrmTempStrategy {
                    dev_handle: init_device_handle(self.drm_render_path.clone())?,
                });
                // Kept whatever the first read through it returns: dropping it
                // on a failed read would reopen the render node every cycle and
                // make the warning above untrue.
                let temperature = fallback.read_temperature();
                self.fallback = Some(fallback);
                temperature
            }
        }
    }
}

/// Read one temperature from an amdgpu hwmon `temp1_input`, rejecting anything
/// that is missing, unreadable, not an integer, or out of plausible range.
fn read_hwmon_millidegrees(path: &Path) -> Result<i64> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| IoError::other(format!("{}: {e}", path.display())))?;
    let millidegrees: i64 = raw.trim().parse().map_err(|_| {
        IoError::other(format!(
            "{} did not contain an integer: {:?}",
            path.display(),
            raw.trim()
        ))
    })?;

    if !(TEMP_MIN_MILLIDEGREES..=TEMP_MAX_MILLIDEGREES).contains(&millidegrees) {
        return Err(IoError::other(format!(
            "{} reported an implausible temperature: {} millidegrees C",
            path.display(),
            millidegrees
        ))
        .into());
    }

    Ok(millidegrees)
}

/// Locate the amdgpu hwmon temperature input for this device.
///
/// Resolved once rather than per read: the hwmon index is stable for the life
/// of the bound device, and a readdir on every control loop would be wasteful.
fn find_hwmon_temp_input(sysfs_path: &Path) -> Result<PathBuf> {
    let hwmon_root = sysfs_path.join("hwmon");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&hwmon_root)
        .map_err(|e| IoError::other(format!("{}: {e}", hwmon_root.display())))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path().join("temp1_input"))
        .filter(|path| path.is_file())
        .collect();
    candidates.sort();

    candidates.into_iter().next().ok_or_else(|| {
        AppError::from(format!(
            "no hwmon temp1_input under {}",
            hwmon_root.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_CONSECUTIVE_FAILURES, TEMP_MAX_MILLIDEGREES, TEMP_MIN_MILLIDEGREES,
        millidegrees_to_celsius, read_hwmon_millidegrees,
    };
    use crate::app_error::Result;
    use crate::gpu::TempStrategy;
    use std::path::PathBuf;

    /// Stands in for `DrmTempStrategy`, which cannot be opened in a unit test.
    struct StubStrategy {
        celsius: u32,
    }

    impl TempStrategy for StubStrategy {
        fn read_temperature(&mut self) -> Result<u32> {
            Ok(self.celsius)
        }
    }

    /// Write `contents` to a uniquely named file and hand back its path.
    ///
    /// The crate has no dev-dependencies and this is the only test that needs a
    /// file on disk, so a counter plus the pid is cheaper than pulling in a
    /// temporary-file crate.
    fn temp_file_with(contents: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);

        let path = std::env::temp_dir().join(format!(
            "cs-governor-hwmon-test-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, contents).expect("failed to write test file");
        path
    }

    #[test]
    fn read_hwmon_accepts_a_plausible_reading() {
        let path = temp_file_with("45000\n");

        assert_eq!(read_hwmon_millidegrees(&path).unwrap(), 45000);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_hwmon_rejects_an_implausible_reading() {
        let path = temp_file_with(&format!("{}\n", TEMP_MAX_MILLIDEGREES + 1));

        assert!(read_hwmon_millidegrees(&path).is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_hwmon_rejects_a_non_integer_reading() {
        let path = temp_file_with("not a number\n");

        assert!(read_hwmon_millidegrees(&path).is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_hwmon_rejects_a_missing_file() {
        let path = std::env::temp_dir().join("cs-governor-hwmon-test-does-not-exist");
        let _ = std::fs::remove_file(&path);

        assert!(read_hwmon_millidegrees(&path).is_err());
    }

    /// A transient failure must not end the strategy: it reuses the last good
    /// reading rather than propagating the error.
    ///
    /// Only the first `MAX_CONSECUTIVE_FAILURES - 1` failures are exercised.
    /// The hand-over itself opens a real DRM render node, which a unit test
    /// cannot do; `reads_delegate_to_the_fallback_once_one_is_installed`
    /// covers what happens afterwards.
    #[test]
    fn a_transient_failure_reuses_the_last_good_reading() {
        let path = temp_file_with("45000\n");
        let mut strategy = super::SysfsTempStrategy::probe_from_path(path.clone(), None);

        assert_eq!(strategy.read_temperature().unwrap(), 45);

        std::fs::write(&path, "garbage\n").unwrap();
        for _ in 1..MAX_CONSECUTIVE_FAILURES {
            assert_eq!(
                strategy.read_temperature().unwrap(),
                45,
                "a failed read should reuse the last good value"
            );
        }

        let _ = std::fs::remove_file(&path);
    }

    /// Once a fallback is installed, reads go through it and the attribute is
    /// not consulted again -- the state after a hand-over.
    #[test]
    fn reads_delegate_to_the_fallback_once_one_is_installed() {
        let path = temp_file_with("45000\n");
        let mut strategy = super::SysfsTempStrategy::probe_from_path(
            path.clone(),
            Some(Box::new(StubStrategy { celsius: 61 })),
        );

        // Remove the attribute: if anything still read it, this would fail.
        std::fs::remove_file(&path).unwrap();

        for _ in 0..(MAX_CONSECUTIVE_FAILURES * 2) {
            assert_eq!(strategy.read_temperature().unwrap(), 61);
        }
    }

    /// The conversion must clamp. `read_hwmon_millidegrees` accepts readings
    /// down to TEMP_MIN_MILLIDEGREES, and a bare `as u32` on a negative i64
    /// wraps to roughly u32::MAX, which reads as permanently over the
    /// throttling threshold.
    #[test]
    fn a_sub_zero_reading_clamps_instead_of_wrapping() {
        assert_eq!(millidegrees_to_celsius(TEMP_MIN_MILLIDEGREES), 0);
        assert_eq!(millidegrees_to_celsius(-1000), 0);
        assert_eq!(millidegrees_to_celsius(-1), 0);
        assert_eq!(millidegrees_to_celsius(0), 0);
        assert_eq!(millidegrees_to_celsius(45_000), 45);
        assert_eq!(millidegrees_to_celsius(TEMP_MAX_MILLIDEGREES), 150);
    }

    /// The validator and the conversion must agree on what is acceptable: a
    /// reading the validator lets through must survive the conversion.
    #[test]
    fn every_accepted_reading_converts_to_a_sane_temperature() {
        for millidegrees in [TEMP_MIN_MILLIDEGREES, -1, 0, 45_000, TEMP_MAX_MILLIDEGREES] {
            let celsius = millidegrees_to_celsius(millidegrees);
            assert!(
                celsius <= 150,
                "{millidegrees} millidegrees converted to {celsius} C"
            );
        }
    }

    /// A good read in between must reset the count, so isolated errors never
    /// accumulate towards giving up.
    #[test]
    fn a_good_read_resets_the_failure_count() {
        let path = temp_file_with("45000\n");
        let mut strategy = super::SysfsTempStrategy::probe_from_path(path.clone(), None);

        for _ in 0..(MAX_CONSECUTIVE_FAILURES * 3) {
            std::fs::write(&path, "garbage\n").unwrap();
            assert!(strategy.read_temperature().is_ok());
            std::fs::write(&path, "46000\n").unwrap();
            assert_eq!(strategy.read_temperature().unwrap(), 46);
        }

        let _ = std::fs::remove_file(&path);
    }
}
