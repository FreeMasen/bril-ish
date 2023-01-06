fn main() {
    env_logger::try_init().ok();
    let prog = if let Some(arg) = std::env::args().skip(1).next() {
        let f = std::fs::File::open(arg).unwrap();
        bril_rs::load_program_from_read(f)
    } else {
        bril_rs::load_program()
    };
    for func in &prog.functions {
        println!("digraph {} {{", func.name);
        let blocks = bril_ish::instrs_to_blocks(&func.instrs);
        let cfg = bril_ish::blocks_to_cfg(blocks);
        for (name, node) in cfg {
            let body: Vec<String> = node.block.body.iter().map(|i| i.to_string()).collect();
            let formatted_name = name.replace(".", "");
            println!(r#"  {} [shape=box, xlabel="{}", label="{}\l"];"#, formatted_name, formatted_name, body.join(r"\l"));
            for succ in node.succ_names {
                println!("    {} -> {};", formatted_name, succ.replace(".", ""));
            }
        }
        println!("}}");
    }
}
