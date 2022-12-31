use bril_ish;


fn main() {
    let prog = bril_rs::load_program();
    let prog = bril_ish::perform_dead_code_elimination(prog);
    bril_rs::output_program(&prog);
}
