// Offline reference adapter only. All tokenization/normalization is upstream.
use std::{
    env, fs,
    io::{BufWriter, Write},
};
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        return Err("TOKENIZER_JSON CORPUS_JSON NEW_OUTPUT_JSONL".into());
    }
    let tokenizer = Tokenizer::from_file(&args[1])?;
    let corpus: Vec<String> = serde_json::from_slice(&fs::read(&args[2])?)?;
    let mut output = BufWriter::new(fs::File::create(&args[3])?);
    writeln!(output, "{{\"implementation\":\"huggingface/tokenizers\",\"version\":\"0.23.2\",\"add_special_tokens\":false,\"skip_special_tokens\":false}}")?;
    for text in corpus {
        let encoded = tokenizer.encode(text.as_str(), false)?;
        let decoded = tokenizer.decode(encoded.get_ids(), false)?;
        serde_json::to_writer(
            &mut output,
            &serde_json::json!({
                "text": text, "ids": encoded.get_ids(), "decoded": decoded,
            }),
        )?;
        writeln!(output)?;
    }
    output.flush()?;
    Ok(())
}
