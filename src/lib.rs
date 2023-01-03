use std::collections::{HashMap, HashSet, BTreeMap};

use bril_rs::{Code, ConstOps, EffectOps, Function, Instruction, Literal, Program, Type, ValueOps};

pub fn perform_dead_code_elimination(prog: Program, perform_numbering: bool, perform_tdce: bool) -> Program {
    log::trace!("perform_dead_code_elimination: {prog:#?}");
    let mut ret_funcs = Vec::new();
    for func in prog.functions {
        
        let mut blocks = instrs_to_blocks(&func.instrs);
        if perform_numbering {
            blocks = number_blocks(blocks);
        }
        while perform_tdce {
            let before = blocks.len();
            blocks = elim_dead_code(blocks);
            
            if before == blocks.len() {
                break;
            }
        }
        ret_funcs.push(Function {
            args: func.args,
            instrs: blocks_to_codes(blocks),
            name: func.name,
            return_type: func.return_type,
        });
    }
    Program {
        functions: ret_funcs,
    }
}

pub fn elim_dead_code(blocks: Vec<BasicBlock>) -> Vec<BasicBlock> {
    log::trace!("elim_dead_code: {blocks:#?}");
    let mut used = HashSet::new();
    for block in &blocks {
        for instr in &block.body {
            match instr {
                Instruction::Value { args, .. } | Instruction::Effect { args, .. } => {
                    for arg in args {
                        used.insert(arg.clone());
                    }
                }
                _ => {}
            }
        }
    }
    blocks
        .into_iter()
        .map(|block| {
            let body = block
                .body
                .into_iter()
                .filter(|instr| match instr {
                    Instruction::Constant { dest, .. } | Instruction::Value { dest, .. } => {
                        used.contains(dest)
                    }
                    _ => true,
                })
                .collect();
            BasicBlock {
                label: block.label,
                body,
            }
        })
        .collect()
}

pub fn instrs_to_blocks(instrs: &[Code]) -> Vec<BasicBlock> {
    log::trace!("instrs_to_blocks");
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
    log::trace!("blocks_to_codes");
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

fn number_blocks(blocks: Vec<BasicBlock>) -> Vec<BasicBlock> {
    blocks.into_iter().map(number_block).collect()
}

fn number_block(block: BasicBlock) -> BasicBlock {
    let first_reads = gen_read_first(&block.body);

    let mut t = LocalInstrNumbering::new(first_reads.into_iter());
    let mut body = Vec::with_capacity(block.body.len());
    let last_writes = gen_last_writes(&block.body);
    for (instr, last_write) in block.body.iter().cloned().zip(last_writes.into_iter()) {
        body.push(t.insert(instr.clone(), last_write));
    }
    BasicBlock {
        label: block.label,
        body,
    }
}

fn gen_last_writes(block: &[Instruction]) -> Vec<bool> {
    let mut ret = vec![false; block.len()];
    let mut set = HashSet::new();
    for (idx, instr) in block.iter().enumerate().rev() {
        match instr {
            Instruction::Constant { dest, .. } | Instruction::Value { dest, .. } => {
                if !set.contains(&dest) {
                    ret[idx] = true;
                    set.insert(dest);
                }
            }
            _ => {}
        }
    }
    ret
}

fn gen_read_first(block: &[Instruction]) -> HashSet<String> {
    let mut ret = HashSet::new();
    let mut writes = HashSet::new();
    for instr in block {
        match instr {
            Instruction::Constant { dest, .. } => {
                writes.insert(dest);
            },
            Instruction::Value { args, dest, .. } => {
                let mut arg_set = HashSet::with_capacity(args.len());
                arg_set.extend(args.iter());
                ret.extend(arg_set.difference(&writes));
                writes.insert(dest);
            },
            Instruction::Effect { args, .. } => {
                let mut arg_set = HashSet::with_capacity(args.len());
                arg_set.extend(args.iter());
                ret.extend(arg_set.difference(&writes));
            },
        }
    }
    ret.into_iter().cloned().collect()
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct BasicBlock {
    pub label: Option<String>,
    pub body: Vec<Instruction>,
}

#[derive(Clone, Debug, Default)]
struct LocalInstrNumbering {
    pub env: HashMap<String, usize>,
    pub instrs: BTreeMap<usize, Instr>,
    pub names: BTreeMap<usize, String>,
    pub consts: BTreeMap<usize, Literal>,
    next_idx: usize,
}


impl LocalInstrNumbering {
    pub fn new(read_first: impl Iterator<Item = String>) -> Self {
        let mut ret = Self::default();
        for val in read_first {
            let idx = ret.get_next_idx();
            ret.env.insert(val.clone(), idx);
            ret.instrs.insert(ret.next_idx, Instr::Effect);
            ret.names.insert(ret.next_idx, val);
        }
        ret
    }

    fn get_next_idx(&mut self) -> usize {
        let ret = self.next_idx;
        self.next_idx += 1;
        ret
    }

    fn get(&self, instr: &Instr) -> Option<(usize, Instr)> {
        for (idx, i) in self.instrs.iter() {
            if instr == i {
                log::debug!("found instr:\n{instr:?}\n{i:?}");
                return Some((*idx, i.clone()));
            }
        }
        None
    }

    pub fn insert(&mut self, instr: Instruction, last_write: bool) -> Instruction {
        log::trace!("inserting {instr:#?} ({last_write})");
        if matches!(instr, Instruction::Effect { .. }) {
            return instr;
        }
        let (converted, dest) = self.convert(instr);
        if let Some((idx, instr)) = self.get(&converted) {
            log::info!("instr exists, returning early");
            self.env.insert(dest.clone(), idx);
            if let Some(value) = self.consts.get(&idx) {
                let const_type = ty_for(value);
                return Instruction::Constant { dest, op: ConstOps::Const, const_type, value: value.clone() }
            }
            return self.revert(dest, instr);
        }
        let mut name = dest.clone();
        if !last_write {
            name = format!("{}.{}", name, self.next_idx);    
        }
        let idx = self.get_next_idx();
        let converted = self.maybe_fold(converted);
        if let Instr::Const { value, .. } = &converted {
            log::debug!("adding constant: {idx} {value:#?}");
            self.consts.insert(idx, value.clone());
        }
        self.instrs.insert(idx, converted.clone());
        log::debug!("adding {name} as {idx}");
        self.names.insert(idx, name.clone());
        assert_eq!(self.instrs.len(), self.names.len());
        self.env.insert(dest.clone(), idx);
        self.revert(name, converted)
    }

    fn convert(&self, instr: Instruction) -> (Instr, String) {
        match instr {
            Instruction::Constant {
                dest,
                const_type,
                value,
                ..
            } => {
                let instr = Instr::r#const(const_type, value);
                (instr, dest.clone())
            }
            Instruction::Value {
                dest,
                args,
                funcs,
                labels,
                op,
                op_type,
            } => {
                let mut numbered_args = Vec::new();
                for arg in args {
                    let idx = self.env[&arg];
                    log::debug!("mapping {arg} to {idx}");
                    numbered_args.push(idx);
                }
                if !matches!(op, ValueOps::Call) {
                    numbered_args.sort();
                }
                let instr = Instr::value(op, op_type, numbered_args, funcs, labels);
                (instr, dest.clone())
            }
            Instruction::Effect { .. } => (Instr::Effect, String::new()),
        }
    }

    fn revert(&self, dest: String, instr: Instr) -> Instruction {
        match instr {
            Instr::Const { const_type, value } => Instruction::Constant {
                dest,
                op: ConstOps::Const,
                const_type,
                value,
            },
            Instr::Value {
                op,
                op_type,
                args,
                funcs,
                labels,
            } => {
                let args = args
                    .into_iter()
                    .map(|idx| self.names.get(&idx).unwrap().clone())
                    .collect();
                Instruction::Value {
                    op,
                    op_type,
                    args,
                    funcs,
                    labels,
                    dest,
                }
            }
            Instr::Effect => unreachable!(),
        }
    }

    fn maybe_fold(&self, instr: Instr) -> Instr {
        let Instr::Value { op, args, .. } = &instr else {
            return instr
        };
        let value = match *op {
            ValueOps::Add => {
                log::debug!("Adding {args:#?}");
                let value = self.compute_args_int(
                    &args,
                    |l,r| {
                        l+r
                    })
                    .expect("failed to add");
                log::debug!("result: {value}");
                Literal::Int(value)
            },
            ValueOps::Sub => {
                let value = self.compute_args_int(
                    &args,
                    |l,r| {
                        l-r
                    })
                    .expect("failed to sub");
                
                Literal::Int(value)
            },
            ValueOps::Mul => {
                let value = self.compute_args_int(
                    &args,
                    |l,r| {
                        l*r
                    })
                    .expect("failed to mul");
                
                Literal::Int(value)
            },
            ValueOps::Div => {
                let value = self.compute_args_int(
                    &args,
                    |l,r| {
                        l/r
                    })
                    .expect(" to div");
                
                 Literal::Int(value)
            },
            ValueOps::Eq => {
                let (lhs, rhs) = self.get_two_ints(&args).unwrap();
                Literal::Bool(lhs.eq(&rhs))
            },
            ValueOps::Lt => {
                let (lhs, rhs) = self.get_two_ints(&args).unwrap();
                Literal::Bool(lhs.lt(&rhs))
            },
            ValueOps::Gt => {
                let (lhs, rhs) = self.get_two_ints(&args).unwrap();
                Literal::Bool(lhs.gt(&rhs))
            },
            ValueOps::Le => {
                let (lhs, rhs) = self.get_two_ints(&args).unwrap();
                Literal::Bool(lhs.le(&rhs))
            },
            ValueOps::Ge => {
                let (lhs, rhs) = self.get_two_ints(&args).unwrap();
                Literal::Bool(lhs.ge(&rhs))
            },
            ValueOps::Not => {
                let v = !self.get_one_bool(&args).unwrap();
                Literal::Bool(v)
            },
            ValueOps::And => {
                let v = self.compute_two_bools(&args, |lhs, rhs| lhs && rhs).unwrap();
                Literal::Bool(v)
                
            },
            ValueOps::Or => {
                let v = self.compute_two_bools(&args, |lhs, rhs| lhs || rhs).unwrap();
                Literal::Bool(v)
            },
            ValueOps::Call => {
                return instr
            },
            ValueOps::Id => {
                let lit = self.consts.get(&args.first().unwrap()).unwrap();
                lit.clone()
            },
            #[cfg(feature = "float")]
            ValueOps::Fadd => {
                let (lhs, rhs) = self.get_two_floats(args).unwrap();
                Literal::Float(lhs + rhs)
            },
            #[cfg(feature = "float")]
            ValueOps::Fsub => {
                let (lhs, rhs) = self.get_two_floats(args).unwrap();
                Literal::Float(lhs - rhs)
            },
            #[cfg(feature = "float")]
            ValueOps::Fmul => {
                let (lhs, rhs) = self.get_two_floats(args).unwrap();
                Literal::Float(lhs * rhs)
            },
            #[cfg(feature = "float")]
            ValueOps::Fdiv => {
                let (lhs, rhs) = self.get_two_floats(args).unwrap();
                Literal::Float(lhs / rhs)
            },
            #[cfg(feature = "float")]
            ValueOps::Feq => {
                let (lhs, rhs) = self.get_two_floats(args).unwrap();
                Literal::Bool(lhs == rhs)
            },
            #[cfg(feature = "float")]
            ValueOps::Flt => {
                let (lhs, rhs) = self.get_two_floats(args).unwrap();
                Literal::Bool(lhs < rhs)
            },
            #[cfg(feature = "float")]
            ValueOps::Fgt => {
                let (lhs, rhs) = self.get_two_floats(args).unwrap();
                Literal::Bool(lhs > rhs)
            },
            #[cfg(feature = "float")]
            ValueOps::Fle => {
                let (lhs, rhs) = self.get_two_floats(args).unwrap();
                Literal::Bool(lhs <= rhs)
            },
            #[cfg(feature = "float")]
            ValueOps::Fge => {
                let (lhs, rhs) = self.get_two_floats(args).unwrap();
                Literal::Bool(lhs >= rhs)
            },

        };
        let const_type = ty_for(&value);
        Instr::Const { const_type, value }
    }

    fn compute_args_int(&self, args: &[usize], f: impl FnOnce(i64, i64) -> i64) -> Result<i64, &'static str> {
        let (lhs, rhs) = self.get_two_ints(args)?;
        Ok(f(lhs, rhs))
    }

    fn compute_two_bools(&self, args: &[usize], f: impl FnOnce(bool, bool) -> bool) -> Result<bool, &'static str> {
        let (lhs, rhs) = self.get_two_bools(args)?;
        Ok(f(lhs, rhs))
    }

    fn get_two_ints(&self, args: &[usize]) -> Result<(i64, i64), &'static str> {
        let lhs = self.consts.get(args.first().unwrap()).unwrap();
        let rhs = self.consts.get(args.last().unwrap()).unwrap();
        let (Literal::Int(lhs), Literal::Int(rhs)) = (lhs, rhs) else {
            return Err("bad comparison")
        };
        Ok((*lhs, *rhs))
    }

    fn get_one_bool(&self, args: &[usize]) -> Result<bool, &'static str> {
        let v = self.consts.get(args.first().unwrap()).unwrap();
        let Literal::Bool(v) = *v else {
            return Err("bad unary bool op")
        };
        Ok(v)
    }
    fn get_two_bools(&self, args: &[usize]) -> Result<(bool, bool), &'static str> {
        let lhs = self.consts.get(args.first().unwrap()).unwrap();
        let rhs = self.consts.get(args.last().unwrap()).unwrap();
        let (Literal::Bool(lhs), Literal::Bool(rhs)) = (lhs, rhs) else {
            return Err("bad comparison")
        };
        Ok((*lhs, *rhs))
    }

    #[cfg(feature = "float")]
    fn get_one_float(&self, args: &[usize]) -> Result<f64, &'static str> {
        let flt = self.consts.get(args.first().unwrap());
        let Some(Literal::Float(ret)) = flt else {
            return Err("expected float");
        };
        Ok(*ret)
    }
    
    #[cfg(feature = "float")]
    fn get_two_floats(&self, args: &[usize]) -> Result<(f64, f64), &'static str> {
        let one = self.get_one_float(args)?;
        let Some(tail) = args.get(1..) else{
            return Err("args too short");
        };
        let two = self.get_one_float(tail)?;
        Ok((one, two))
    }

}

fn ty_for(lit: &Literal) -> Type {
    match lit {
        Literal::Int(_) => Type::Int,
        Literal::Bool(_) => Type::Bool,
        #[cfg(feature = "float")]
        Literal::Float(_) => Type::Float,
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Instr {
    Const {
        const_type: Type,
        value: Literal,
    },
    Value {
        op: ValueOps,
        op_type: Type,
        args: Vec<usize>,
        funcs: Vec<String>,
        labels: Vec<String>,
    },
    Effect,
}

impl Instr {
    pub fn r#const(const_type: Type, value: Literal) -> Self {
        Self::Const { const_type, value }
    }
    pub fn value(
        op: ValueOps,
        op_type: Type,
        args: Vec<usize>,
        funcs: Vec<String>,
        labels: Vec<String>,
    ) -> Self {
        Self::Value {
            op,
            op_type,
            args,
            funcs,
            labels,
        }
    }
}
