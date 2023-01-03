use bril_ish;

fn main() {
    env_logger::try_init().ok();
    let prog = if let Some(arg) = std::env::args().skip(1).next() {
        let f = std::fs::File::open(arg).unwrap();
        bril_rs::load_program_from_read(f)
    } else {
        bril_rs::load_program()
    };
    let prog = bril_ish::perform_dead_code_elimination(prog, true, true);
    bril_rs::output_program(&prog);
}
