fn main() {
    let args: Vec<String> = std::env::args().collect();
    match lean_prover::run_cli_args(&args) {
        Ok((code, output)) => {
            if !output.is_empty() {
                println!("{output}");
            }
            std::process::exit(code);
        }
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    }
}
