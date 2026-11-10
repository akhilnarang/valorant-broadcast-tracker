//! A debug dump of what the readers see, as JSON lines: from a directory of raw
//! 1280x720 RGB24 frames (`--rgb DIR`) or from a video decoded through the
//! tracker's ffmpeg arguments (`--video PATH`).
//!
//! With the `scoreboard` feature on, the `strip` field is the production read: the
//! CTC lexicon read over the template reader, not the template read alone.

use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use clap::Parser;
use serde_json::json;

use vbt::stream::{DECODE_VF, read_frame};
use vbt_vision::frame::Frame;
use vbt_vision::recognizer::Recognizer;
use vbt_vision::templates::Templates;
use vbt_vision::{regions, scene};

#[derive(Parser, Debug)]
#[command(name = "vbt-probe", about = "Dumps what the readers see in each frame")]
struct Args {
    /// Directory of raw RGB24 `.rgb` frames with matching `.json` size files.
    #[arg(long)]
    rgb: Option<PathBuf>,
    /// A video decoded through the same ffmpeg arguments as the tracker.
    #[arg(long)]
    video: Option<PathBuf>,
    /// Expected team codes, comma-separated; adds the `plates` block, the OCR's read of
    /// both tags (needs the `scoreboard` feature).
    #[arg(long)]
    teams: Option<String>,
}

fn main() {
    let args = Args::parse();
    if let Err(message) = run(args) {
        eprintln!("vbt-probe: {message}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), String> {
    let templates = Templates::load_embedded().map_err(|e| e.to_string())?;
    let recognizer = vbt::tracker_recognizer(Recognizer::new().map_err(|e| e.to_string())?, 1);
    let tags = vbt::tag_reader(1);
    let codes: Option<Vec<String>> = args
        .teams
        .as_ref()
        .map(|teams| teams.split(',').map(|code| code.to_string()).collect());

    if let Some(dir) = &args.rgb {
        let mut frames: Vec<(String, PathBuf)> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("rgb"))
            .map(|path| {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                (name, path)
            })
            .collect();
        frames.sort();
        for (name, path) in frames {
            let frame = load_rgb(&path)?;
            print_line(
                &name,
                0,
                &frame,
                &templates,
                &*recognizer,
                tags.as_deref(),
                codes.as_deref(),
            )?;
        }
        return Ok(());
    }

    let Some(video) = &args.video else {
        return Err("--rgb or --video is required".to_string());
    };
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            &video.display().to_string(),
            "-an",
            "-vf",
            DECODE_VF,
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| "ffmpeg stdout unavailable".to_string())?;
    let name = video
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut second = 0i64;
    while let Ok(frame) = read_frame(&mut stdout) {
        print_line(
            &name,
            second,
            &frame,
            &templates,
            &*recognizer,
            tags.as_deref(),
            codes.as_deref(),
        )?;
        second += 1;
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("ffmpeg exited with {status}"));
    }
    Ok(())
}

fn load_rgb(path: &std::path::Path) -> Result<Frame, String> {
    let size_path = path.with_extension("json");
    let size: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&size_path).map_err(|e| format!("{}: {e}", size_path.display()))?,
    )
    .map_err(|e| format!("{}: {e}", size_path.display()))?;
    let w = size["w"].as_u64().ok_or("size json has no w")? as usize;
    let h = size["h"].as_u64().ok_or("size json has no h")? as usize;
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut pix = Vec::new();
    file.read_to_end(&mut pix)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Frame::from_rgb(w, h, pix).map_err(|e| e.to_string())
}

fn print_line(
    name: &str,
    second: i64,
    frame: &Frame,
    templates: &Templates,
    recognizer: &dyn vbt::tracker::Recognizer,
    tags: Option<&(dyn vbt::tracker::TagReader + Send)>,
    codes: Option<&[String]>,
) -> Result<(), String> {
    let tmpl_left = templates.read_score(&frame.crop(regions::SCORE_LEFT));
    let tmpl_right = templates.read_score(&frame.crop(regions::SCORE_RIGHT));
    let lr_left = recognizer.score(&frame.crop(regions::SCORE_LEFT));
    let lr_right = recognizer.score(&frame.crop(regions::SCORE_RIGHT));
    let strip = vbt::tracker::Recognizer::map(recognizer, &frame.crop_raw(regions::MAP_STRIP));
    let (pause, pause_ok) = vbt::tracker::Recognizer::pause(
        recognizer,
        &frame.crop(regions::PAUSE_HEADER),
        &frame.crop(regions::PAUSE_CARD),
    );
    let mut line = json!({
        "frame": name,
        "second": second,
        "scene": scene::scene_of(frame).as_str(),
        "score_visible": scene::score_visible(frame),
        "tmpl_left": {"value": tmpl_left.value, "confidence": tmpl_left.confidence},
        "tmpl_right": {"value": tmpl_right.value, "confidence": tmpl_right.confidence},
        "lr_left": {"value": lr_left.unwrap_or(0), "ok": lr_left.is_some()},
        "lr_right": {"value": lr_right.unwrap_or(0), "ok": lr_right.is_some()},
        "strip": strip,
        "pause": {"kind": pause.kind, "reason": pause.reason, "score": pause.score, "ok": pause_ok},
    });
    if let (Some(tags), Some([a, b])) = (tags, codes) {
        let [left, right] = tags.read(frame, [a, b]);
        line["plates"] = json!({"left": left, "right": right});
    }
    println!(
        "{}",
        serde_json::to_string(&line).map_err(|e| e.to_string())?
    );
    Ok(())
}
