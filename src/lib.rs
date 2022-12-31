use std::collections::{HashSet};

use bril_rs::{Program, Function, Instruction, Code, EffectOps};



pub fn perform_dead_code_elimination(mut prog: Program) -> Program {
    loop {
        let ct = prog.functions.iter().fold(0,|acc, f| {
            acc + f.instrs.iter().count()
        });
        prog = dead_code_elimination_pass(prog);
        let ct2 = prog.functions.iter().fold(0,|acc, f| {
            acc + f.instrs.iter().count()
        }); 
        if ct == ct2 {
            return prog
        }
    }
}

pub fn dead_code_elimination_pass(prog: Program) -> Program {
    Program {
        functions: prog.functions.into_iter().map(elim_dead_code).collect()
    }
}

pub fn elim_dead_code(mut func: Function) -> Function {
    let mut blocks = instrs_to_blocks(&func.instrs);
    let mut used = HashSet::new();
    for block in &blocks {
        for instr in &block.body {
            match instr {
                Instruction::Value { args,.. } |
                Instruction::Effect { args, .. } => {
                    for arg in args {
                        used.insert(arg.clone());
                    }
                },
                _ => {},
            }
        }
    }
    blocks = blocks.into_iter().map(|block| {
        let body = block.body.into_iter().filter(|instr| {
            match instr {
                Instruction::Constant { dest, .. } |
                Instruction::Value { dest, .. } => {
                    used.contains(dest)
                },
                _ => true
            }
        }).collect();
        BasicBlock {
            label: block.label,
            body,
        }
    }).collect();
    func.instrs = blocks_to_codes(blocks);
    func
}

pub fn instrs_to_blocks(instrs: &[Code]) -> Vec<BasicBlock> {
    let mut ret = Vec::new();
    let mut block = BasicBlock::default();
    for code in instrs {
        match code {
            Code::Label { label } => {
                if !block.label.is_none() || !block.body.is_empty() {
                    ret.push(block);
                    block = BasicBlock::default()
                }
                block.label = Some(label.clone());
            }
            Code::Instruction(instr) => {
                block.body.push(instr.clone());
                let Instruction::Effect { op, .. } = instr else {
                    continue;
                };
                if matches!(op, EffectOps::Branch | EffectOps::Jump | EffectOps::Return) {
                    ret.push(block);
                    block = BasicBlock::default();
                }
            }
        }
    }
    if block.label.is_some() || !block.body.is_empty() {
        ret.push(block);
    }
    ret
}

fn blocks_to_codes(blocks: Vec<BasicBlock>) -> Vec<Code> {
    let mut ret = Vec::new();
    for block in blocks {
        if let Some(label) = block.label {
            ret.push(Code::Label { label });
        }
        for instr in block.body {
            ret.push(Code::Instruction(instr));
        }
    }
    ret
}

#[derive(Default, Debug, Clone)]
pub struct BasicBlock {
    pub label: Option<String>,
    pub body: Vec<Instruction>,
}
