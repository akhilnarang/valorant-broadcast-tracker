//! Model loading: the ORT session options and the pruned model's class list.

use std::collections::HashMap;

use ort::ep::CPU;
use ort::session::Session;
use ort::value::TensorRef;

use crate::ctc::BLANK_CHAR;
use crate::tensor::TENSOR_H;
use crate::{Error, MODEL_BYTES};

/// The model's input and output tensor names.
const INPUT: &str = "x";
const OUTPUT: &str = "fetch_name_0";

/// Runs `n` planar crops of width `w` and returns the softmax output with its timestep and
/// class counts.
pub(crate) fn run(
    session: &mut Session,
    n: usize,
    w: usize,
    data: &[f32],
) -> Result<(Vec<f32>, usize, usize), Error> {
    let input = TensorRef::from_array_view(([n, 3, TENSOR_H, w], data))?;
    let outputs = session.run(ort::inputs![INPUT => input])?;
    let (shape, probs) = outputs[OUTPUT].try_extract_tensor::<f32>()?;
    Ok((probs.to_vec(), shape[1] as usize, shape[2] as usize))
}

/// A recognition session with the options §3.1 measures: `threads` intra-op threads, one
/// inter-op thread, no spinning, ORT's memory pattern on, and the CPU arena off.
pub fn session(threads: usize) -> Result<Session, Error> {
    let session = Session::builder()?
        .with_intra_threads(threads.max(1))?
        .with_inter_threads(1)?
        .with_intra_op_spinning(false)?
        .with_memory_pattern(true)?
        .with_execution_providers([CPU::default().with_arena_allocator(false).build()])?
        .commit_from_memory(MODEL_BYTES)?;
    Ok(session)
}

/// The classes blank, the model's `character` dict in order, then space, plus the
/// character lookup. The pruned model ships only the scoreboard's alphabet (printable
/// ASCII plus the Latin-1 letters), so class index 0 is blank, the dict lines are
/// 1..=n and space is n+1. See `tools/prune_head.py`.
///
/// A dry run checks the class count against the model's own output width, so a model
/// swapped under the pruned one fails at load instead of scoring garbage.
pub fn classes(session: &mut Session) -> Result<(Vec<char>, HashMap<char, usize>), Error> {
    let character = session
        .metadata()?
        .custom("character")
        .ok_or_else(|| Error::Model("model metadata has no 'character' list".into()))?;
    let mut classes: Vec<char> = vec![BLANK_CHAR];
    let mut index = HashMap::new();
    for line in character.split('\n') {
        let mut chars = line.chars();
        let first = chars.next().unwrap_or(BLANK_CHAR);
        classes.push(first);
        if chars.next().is_none() && first != ' ' {
            index.insert(first, classes.len() - 1);
        }
    }
    // Space is the last class, after the metadata list.
    classes.push(' ');
    index.insert(' ', classes.len() - 1);

    let (_, _, model_classes) = run(session, 1, 64, &[0f32; 3 * TENSOR_H * 64])?;
    if model_classes != classes.len() {
        return Err(Error::Model(format!(
            "model outputs {model_classes} classes, metadata lists {}",
            classes.len()
        )));
    }
    Ok((classes, index))
}
