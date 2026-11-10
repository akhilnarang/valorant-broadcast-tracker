//! CLI and `--serve` daemon for the native scoreboard reader.
//!
//! One-shot: `vbt-scoreboard --context ctx.json [--threads 2] [--bench N] frame.png...`
//! Daemon:   `vbt-scoreboard --context ctx.json --serve`
//!
//! The daemon reads frames on stdin and writes results on stdout. A request is a 4-byte
//! big-endian length, then PNG or JPEG bytes. A reply is a 4-byte big-endian length, then
//! the canonical JSON, or `null` when there is no table or the frame is unusable.

use std::error::Error as StdError;
use std::io::{self, Read, Write};
use std::process::ExitCode;
use std::time::Instant;

use vbt_scoreboard::reader::Reader;
use vbt_scoreboard::schema::{Context, Scoreboard};
use vbt_vision::frame::Frame;

/// Upper bound on a request's payload.
const MAX_PAYLOAD: usize = 64 << 20;

struct Args {
    context: Option<String>,
    threads: usize,
    bench: usize,
    serve: bool,
    images: Vec<String>,
}

fn usage() -> &'static str {
    "usage: vbt-scoreboard --context ctx.json [--threads N] [--bench N] [-i] frame.png...\n\
     \x20      vbt-scoreboard --context ctx.json --serve"
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        context: None,
        threads: 2,
        bench: 0,
        serve: false,
        images: Vec::new(),
    };
    let mut argv = std::env::args().skip(1);
    while let Some(arg) = argv.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((flag, value)) => (flag.to_string(), Some(value.to_string())),
            None => (arg.clone(), None),
        };
        let mut value = |name: &str| -> Result<String, String> {
            if let Some(v) = inline.clone() {
                return Ok(v);
            }
            argv.next()
                .ok_or_else(|| format!("{name} needs a value\n{}", usage()))
        };
        match flag.as_str() {
            "-c" | "--context" => args.context = Some(value("--context")?),
            "-i" | "--image" => args.images.push(value("--image")?),
            "--threads" => {
                args.threads = value("--threads")?
                    .parse()
                    .map_err(|_| "--threads needs a positive integer".to_string())?;
                if args.threads == 0 {
                    return Err("--threads needs a positive integer".into());
                }
            }
            "--bench" => {
                args.bench = value("--bench")?
                    .parse()
                    .map_err(|_| "--bench needs a non-negative integer".to_string())?
            }
            "--serve" => args.serve = true,
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other if other.starts_with('-') && other != "-" => {
                return Err(format!("unknown option {other}\n{}", usage()));
            }
            _ => args.images.push(arg),
        }
    }
    if args.context.is_none() {
        return Err(format!("--context is required\n{}", usage()));
    }
    Ok(args)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("vbt-scoreboard: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn StdError>> {
    let args = parse_args()?;
    let context_bytes = std::fs::read(args.context.as_deref().unwrap_or_default())?;
    let ctx: Context = serde_json::from_slice(&context_bytes).map_err(|e| {
        format!(
            "context {}: {e}",
            args.context.as_deref().unwrap_or_default()
        )
    })?;

    let init = Instant::now();
    let mut reader = Reader::new(args.threads)?;
    let init_ms = init.elapsed().as_secs_f64() * 1000.0;

    if args.serve {
        return serve(&mut reader, &ctx, args.threads);
    }
    if args.images.is_empty() {
        return Err(format!("no frames given\n{}", usage()).into());
    }
    for path in &args.images {
        let bytes = std::fs::read(path)?;
        let decode = Instant::now();
        let image = image::load_from_memory(&bytes)?;
        drop(bytes);
        let frame = Frame::from_rgb_image_owned(image.into_rgb8());
        let decode_ms = decode.elapsed().as_secs_f64() * 1000.0;

        let scoreboard = reader.analyze(&frame, &ctx)?;
        println!("{}", serde_json::to_string(&scoreboard)?);
        if args.bench > 0 {
            bench(
                &mut reader,
                &ctx,
                &frame,
                path,
                args.bench,
                init_ms,
                decode_ms,
            )?;
        }
    }
    Ok(())
}

/// Times `analyze` after 3 warm-ups and reports p50/p95/max plus decode and init separately.
fn bench(
    reader: &mut Reader,
    ctx: &Context,
    frame: &Frame,
    path: &str,
    n: usize,
    init_ms: f64,
    decode_ms: f64,
) -> Result<(), Box<dyn StdError>> {
    for _ in 0..3 {
        reader.analyze(frame, ctx)?;
    }
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let start = Instant::now();
        reader.analyze(frame, ctx)?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(f64::total_cmp);
    let percentile = |q: f64| -> f64 {
        let index = ((q * samples.len() as f64).ceil() as usize).saturating_sub(1);
        samples[index.min(samples.len() - 1)]
    };
    eprintln!(
        "{path}: init {init_ms:.1} ms, decode {decode_ms:.1} ms, read p50 {:.1} ms, p95 {:.1} ms, max {:.1} ms (n={n})",
        percentile(0.50),
        percentile(0.95),
        samples[samples.len() - 1],
    );
    Ok(())
}

/// Serves length-prefixed requests and replies on stdio, as described at the top.
fn serve(reader: &mut Reader, ctx: &Context, threads: usize) -> Result<(), Box<dyn StdError>> {
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());
    eprintln!("vbt-scoreboard ready ({threads} threads)");
    loop {
        let mut header = [0u8; 4];
        match input.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        }
        let size = u32::from_be_bytes(header) as usize;
        if size > MAX_PAYLOAD {
            return Err(format!("frame too large: {size} bytes").into());
        }
        let mut payload = vec![0u8; size];
        input.read_exact(&mut payload)?;
        let body = match analyze_bytes(reader, ctx, &payload) {
            Ok(body) => body,
            Err(e) => {
                eprintln!("vbt-scoreboard: frame dropped: {e}");
                b"null".to_vec()
            }
        };
        output.write_all(&(body.len() as u32).to_be_bytes())?;
        output.write_all(&body)?;
        output.flush()?;
    }
}

fn analyze_bytes(
    reader: &mut Reader,
    ctx: &Context,
    payload: &[u8],
) -> Result<Vec<u8>, Box<dyn StdError>> {
    let image = image::load_from_memory(payload)?;
    let frame = Frame::from_rgb_image_owned(image.into_rgb8());
    let scoreboard: Option<Scoreboard> = reader.analyze(&frame, ctx)?;
    Ok(serde_json::to_vec(&scoreboard)?)
}
