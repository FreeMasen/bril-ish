use std::collections::{BTreeMap, HashMap, HashSet};

use bril_rs::{Code, ConstOps, EffectOps, Function, Instruction, Literal, Program, Type, ValueOps};

pub fn perform_dead_code_elimination(prog: Program) -> Program {
    log::trace!("perform_dead_code_elimination: {prog:#?}");
    let mut ret_funcs = Vec::new();
    for func in prog.functions {
        let mut blocks = instrs_to_blocks(&func.instrs);
        loop {
            let before: usize = blocks.iter().map(|b| b.body.len()).sum();
            blocks = elim_dead_code(blocks);
            blocks = drop_all_local_killed(blocks);
            if before == blocks.iter().map(|b| b.body.len()).sum::<usize>() {
                break;
            }
        }
        ret_funcs.push(Function {
            args: func.args,
            instrs: blocks_to_codes(blocks, false),
            name: func.name,
            return_type: func.return_type,
        });
    }
    Program {
        functions: ret_funcs,
    }
}

pub fn perform_lvn(program: Program) -> Program {
    let mut ret_funcs = Vec::new();
    for func in program.functions {
        let blocks = instrs_to_blocks(&func.instrs);
        let numbered = number_blocks(blocks);
        ret_funcs.push(Function {
            instrs: blocks_to_codes(numbered, true),
            ..func
        })
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

pub fn drop_all_local_killed(blocks: Vec<BasicBlock>) -> Vec<BasicBlock> {
    blocks.into_iter().map(drop_local_killed).collect()
}

pub fn drop_local_killed(block: BasicBlock) -> BasicBlock {
    let mut set: BTreeMap<usize, Instruction> = block.body.into_iter().enumerate().collect();
    let mut to_remove = HashSet::new();
    let mut last_def: HashMap<&str, usize> = HashMap::new();
    for (idx, instr) in &set {
        match instr {
            Instruction::Value { args, .. } |
            Instruction::Effect { args, .. } => {
                for arg in args {
                    last_def.retain(|&v, _| v != arg);
                }
            },
            _ => {},
        }
        match instr {
            Instruction::Constant { dest, .. } |
            Instruction::Value { dest, ..} => {
                if let Some(idx) = last_def.get(dest.as_str()) {
                    to_remove.insert(*idx);
                }
                last_def.insert(dest.as_str(), *idx);
            },
            _ => {},
          }
    }

    for id in to_remove {
        set.remove(&id);
    }
    BasicBlock {
        label: block.label,
        body: set.into_values().collect()
    }
}

pub fn instrs_to_blocks(instrs: &[Code]) -> Vec<BasicBlock> {
    log::trace!("instrs_to_blocks");
    let mut ret = Vec::new();
    let mut iter = instrs.iter();
    let (label, instrs) = match iter.next() {
        Some(code) => match code {
            Code::Label { label } => (label.clone(), Vec::new()),
            Code::Instruction(instr) => ("b1".to_string(), vec![instr.clone()]),
        },
        None => return ret,
    };
    let mut block = BasicBlock::new(label, instrs);
    for code in iter {
        match code {
            Code::Label { label } => {
                ret.push(block);
                block = BasicBlock::new(label.clone(), Vec::new())
            }
            Code::Instruction(instr) => {
                block.body.push(instr.clone());
                // let Instruction::Effect { op, .. } = instr else {
                //     continue;
                // };
                // if matches!(op, EffectOps::Branch | EffectOps::Jump | EffectOps::Return) {
                //     ret.push(block);
                //     block = BasicBlock::default();
                // }
            }
        }
    }
    ret.push(block);
    ret
}

fn blocks_to_codes(blocks: Vec<BasicBlock>, include_first_label: bool) -> Vec<Code> {
    log::trace!("blocks_to_codes");
    let mut ret = Vec::new();
    for block in blocks {
        if block.label == "b1" {
            if include_first_label {
                ret.push(Code::Label { label: block.label });
            }
        } else {
            ret.push(Code::Label { label: block.label });
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
    log::trace!("gen_read_first");
    let mut ret = HashSet::new();
    let mut writes = HashSet::new();
    for instr in block {
        match instr {
            Instruction::Constant { dest, .. } => {
                writes.insert(dest);
            }
            Instruction::Value { args, dest, .. } => {
                let mut arg_set = HashSet::with_capacity(args.len());
                arg_set.extend(args.iter());
                ret.extend(arg_set.difference(&writes));
                writes.insert(dest);
            }
            Instruction::Effect { args, .. } => {
                let mut arg_set = HashSet::with_capacity(args.len());
                arg_set.extend(args.iter());
                ret.extend(arg_set.difference(&writes));
            }
        }
    }
    dbg!(ret.into_iter().cloned().collect())
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct BasicBlock {
    pub label: String,
    pub body: Vec<Instruction>,
}

impl BasicBlock {
    pub fn new(label: String, body: Vec<Instruction>) -> Self {
        Self { label, body }
    }
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
            ret.instrs.insert(
                ret.next_idx,
                Instr::Const {
                    const_type: Type::Int,
                    value: Literal::Int(0),
                },
            );
            ret.names.insert(idx, val);
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
            if instr.compare(i) {
                log::debug!("found instr:\n{instr:?}\n{i:?}");
                return Some((*idx, i.clone()));
            }
        }
        None
    }

    pub fn insert(&mut self, instr: Instruction, last_write: bool) -> Instruction {
        if matches!(instr, Instruction::Effect { .. }) {
            return self.insert_effect(instr);
        }
        log::trace!("inserting {instr:#?} ({last_write})");
        let (mut converted, dest) = self.convert(instr);
        if let Some((idx, instr)) = self.get(&converted) {
            log::info!("instr exists, returning early");
            self.env.insert(dest.clone(), idx);
            if let Some(value) = self.consts.get(&idx) {
                let const_type = ty_for(value);
                return Instruction::Constant {
                    dest,
                    op: ConstOps::Const,
                    const_type,
                    value: value.clone(),
                };
            }

            return self.revert(dest, instr);
        }
        let mut name = dest.clone();
        if !last_write {
            name = format!("{}.{}", name, self.next_idx);
        }
        let idx = self.get_next_idx();
        match self.maybe_fold(&converted) {
            Ok(instr) => {
                converted = instr;
            }
            Err(err) => {
                log::warn!("Error folding: {err}");
            }
        }
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

    fn insert_effect(&mut self, instr: Instruction) -> Instruction {
        let Instruction::Effect { args, funcs, labels, op } = instr else {
            unreachable!();
        };
        let args = args
            .iter()
            .map(|a| {
                let ret = self.env[a];
                if self.names.get(&ret).is_none() {
                    println!(
                        "Found non-existent arg {:#?} vs {:#?}",
                        self.names, self.env
                    )
                }
                ret
            })
            .map(|n| self.names[&n].to_string())
            .collect();
        Instruction::Effect {
            args,
            funcs,
            labels,
            op,
        }
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
            Instruction::Effect {
                args,
                funcs,
                labels,
                op,
            } => {
                let mut numbered_args = Vec::new();
                for arg in args {
                    let idx = self.env[&arg];
                    log::debug!("mapping {arg} to {idx}");
                    numbered_args.push(idx);
                }
                let instr = Instr::effect(numbered_args, op, funcs, labels);
                (instr, String::new())
            }
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
            Instr::Effect {
                args,
                funcs,
                labels,
                op,
            } => {
                let args = args
                    .iter()
                    .map(|idx| {
                        self.names
                            .get(idx)
                            .unwrap_or_else(|| {
                                panic!("Failed to revert {op:?} {args:?}");
                            })
                            .clone()
                    })
                    .collect();
                Instruction::Effect {
                    args,
                    funcs,
                    labels,
                    op,
                }
            }
        }
    }

    fn maybe_fold(&self, instr: &Instr) -> Result<Instr, &'static str> {
        let Instr::Value { op, args, .. } = instr else {
            return Err("can oly fold a value");
        };
        let value = match *op {
            ValueOps::Add => {
                log::debug!("Adding {args:#?}");
                let value = self.compute_args_int(&args, |l, r| l + r)?;
                log::debug!("result: {value}");
                Literal::Int(value)
            }
            ValueOps::Sub => {
                let value = self.compute_args_int(&args, |l, r| l - r)?;
                Literal::Int(value)
            }
            ValueOps::Mul => {
                let value = self.compute_args_int(&args, |l, r| l * r)?;

                Literal::Int(value)
            }
            ValueOps::Div => {
                let (lhs, rhs) = self.get_two_ints(&args)?;
                if rhs == 0 {
                    return Err("div by zero");
                }
                let value = lhs / rhs;

                Literal::Int(value)
            }
            ValueOps::Eq => {
                let (lhs, rhs) = self.get_two_ints(&args)?;
                Literal::Bool(lhs.eq(&rhs))
            }
            ValueOps::Lt => {
                let (lhs, rhs) = self.get_two_ints(&args)?;
                Literal::Bool(lhs.lt(&rhs))
            }
            ValueOps::Gt => {
                let (lhs, rhs) = self.get_two_ints(&args)?;
                Literal::Bool(lhs.gt(&rhs))
            }
            ValueOps::Le => {
                let (lhs, rhs) = self.get_two_ints(&args)?;
                Literal::Bool(lhs.le(&rhs))
            }
            ValueOps::Ge => {
                let (lhs, rhs) = self.get_two_ints(&args)?;
                Literal::Bool(lhs.ge(&rhs))
            }
            ValueOps::Not => {
                let v = !self.get_one_bool(&args)?;
                Literal::Bool(v)
            }
            ValueOps::And => {
                let v = self.compute_two_bools(&args, |lhs, rhs| lhs && rhs)?;
                Literal::Bool(v)
            }
            ValueOps::Or => {
                let v = self.compute_two_bools(&args, |lhs, rhs| lhs || rhs)?;
                Literal::Bool(v)
            }
            ValueOps::Call => return Err("can't fold a call"),
            ValueOps::Id => {
                let Some(arg) = args.first() else {
                    return Err("too few args");
                };
                let Some(lit) = self.consts.get(arg) else {
                    return Err("non constant arg");
                };
                lit.clone()
            }
            #[cfg(feature = "float")]
            ValueOps::Fadd => {
                let (lhs, rhs) = self.get_two_floats(args)?;
                Literal::Float(lhs + rhs)
            }
            #[cfg(feature = "float")]
            ValueOps::Fsub => {
                let (lhs, rhs) = self.get_two_floats(args)?;
                Literal::Float(lhs - rhs)
            }
            #[cfg(feature = "float")]
            ValueOps::Fmul => {
                let (lhs, rhs) = self.get_two_floats(args)?;
                Literal::Float(lhs * rhs)
            }
            #[cfg(feature = "float")]
            ValueOps::Fdiv => {
                let (lhs, rhs) = self.get_two_floats(args)?;
                if rhs == 0.0 {
                    return Err("fdiv by zero");
                }
                log::debug!("folding fdiv {lhs} / {rhs}");
                Literal::Float(lhs / rhs)
            }
            #[cfg(feature = "float")]
            ValueOps::Feq => {
                let (lhs, rhs) = self.get_two_floats(args)?;
                Literal::Bool(lhs == rhs)
            }
            #[cfg(feature = "float")]
            ValueOps::Flt => {
                let (lhs, rhs) = self.get_two_floats(args)?;
                Literal::Bool(lhs < rhs)
            }
            #[cfg(feature = "float")]
            ValueOps::Fgt => {
                let (lhs, rhs) = self.get_two_floats(args)?;
                Literal::Bool(lhs > rhs)
            }
            #[cfg(feature = "float")]
            ValueOps::Fle => {
                let (lhs, rhs) = self.get_two_floats(args)?;
                Literal::Bool(lhs <= rhs)
            }
            #[cfg(feature = "float")]
            ValueOps::Fge => {
                let (lhs, rhs) = self.get_two_floats(args)?;
                Literal::Bool(lhs >= rhs)
            }
        };
        let const_type = ty_for(&value);
        Ok(Instr::Const { const_type, value })
    }

    fn compute_args_int(
        &self,
        args: &[usize],
        f: impl FnOnce(i64, i64) -> i64,
    ) -> Result<i64, &'static str> {
        let (lhs, rhs) = self.get_two_ints(args)?;
        Ok(f(lhs, rhs))
    }

    fn compute_two_bools(
        &self,
        args: &[usize],
        f: impl FnOnce(bool, bool) -> bool,
    ) -> Result<bool, &'static str> {
        let (lhs, rhs) = self.get_two_bools(args)?;
        Ok(f(lhs, rhs))
    }

    fn get_two_ints(&self, args: &[usize]) -> Result<(i64, i64), &'static str> {
        let lhs = self
            .consts
            .get(args.first().ok_or_else(|| "args too short")?)
            .ok_or_else(|| "not a const")?;
        let rhs = self
            .consts
            .get(args.last().ok_or_else(|| "args too short")?)
            .ok_or_else(|| "not a const")?;
        let (Literal::Int(lhs), Literal::Int(rhs)) = (lhs, rhs) else {
            return Err("bad comparison")
        };
        Ok((*lhs, *rhs))
    }

    fn get_one_bool(&self, args: &[usize]) -> Result<bool, &'static str> {
        let v = self
            .consts
            .get(args.first().ok_or_else(|| "args too short")?)
            .ok_or_else(|| "not a const")?;
        let Literal::Bool(v) = *v else {
            return Err("bad unary bool op")
        };
        Ok(v)
    }
    fn get_two_bools(&self, args: &[usize]) -> Result<(bool, bool), &'static str> {
        let lhs = self
            .consts
            .get(args.first().ok_or_else(|| "args too short")?)
            .unwrap();
        let rhs = self
            .consts
            .get(args.last().ok_or_else(|| "args too short")?)
            .unwrap();
        let (Literal::Bool(lhs), Literal::Bool(rhs)) = (lhs, rhs) else {
            return Err("bad comparison")
        };
        Ok((*lhs, *rhs))
    }

    #[cfg(feature = "float")]
    fn get_one_float(&self, args: &[usize]) -> Result<f64, &'static str> {
        let flt = self
            .consts
            .get(args.first().ok_or_else(|| "args too short")?);
        let Some(Literal::Float(ret)) = flt else {
            return Err("expected float");
        };
        Ok(*ret)
    }

    #[cfg(feature = "float")]
    fn get_two_floats(&self, args: &[usize]) -> Result<(f64, f64), &'static str> {
        let one = self.get_one_float(args)?;
        let Some(tail) = args.get(1..) else {
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
    Effect {
        op: EffectOps,
        funcs: Vec<String>,
        labels: Vec<String>,
        args: Vec<usize>,
    },
}

impl Instr {
    pub fn r#const(const_type: Type, value: Literal) -> Self {
        Self::Const { const_type, value }
    }
    pub fn effect(
        args: Vec<usize>,
        op: EffectOps,
        funcs: Vec<String>,
        labels: Vec<String>,
    ) -> Self {
        Self::Effect {
            args,
            op,
            funcs,
            labels,
        }
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
    pub fn compare(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Const {
                    const_type: t1,
                    value: v1,
                },
                Self::Const {
                    const_type: t2,
                    value: v2,
                },
            ) => {
                if t1 != t2 {
                    return false;
                }
                #[cfg(feature = "float")]
                if let (Literal::Float(f1), Literal::Float(f2)) = (v1, v2) {
                    return compare_floats(*f1, *f2);
                }
                v1 == v2
            }
            _ => false,
        }
    }
}

#[cfg(feature = "float")]
fn compare_floats(f1: f64, f2: f64) -> bool {
    f1.is_sign_negative() == f2.is_sign_negative()
        && f1.is_finite() == f2.is_finite()
        && f1.is_nan() == f2.is_nan()
        && f1 == f2
}

pub struct CfgNode {
    pub block: BasicBlock,
    pub pred_names: Vec<String>,
    pub succ_names: Vec<String>,
}

pub fn blocks_to_cfg(blocks: Vec<BasicBlock>) -> BTreeMap<String, CfgNode> {
    let mut ret = BTreeMap::new();
    let names: Vec<String> = blocks.iter().map(|b| b.label.clone()).collect();
    for (i, block) in blocks.iter().enumerate() {
        let mut succs = Vec::new();
        if let Some(Instruction::Effect { labels, op, .. }) = block.body.last() {
            match op {
                EffectOps::Branch | EffectOps::Jump => {
                    succs = labels.clone();
                }
                EffectOps::Return => {
                    succs = vec!["___RET___".to_string()];
                }
                _ => {}
            }
        } else {
            if let Some(next) = blocks.get(i + 1) {
                succs = vec![next.label.clone()];
            } else {
                succs = vec!["___RET___".to_string()];
            }
        }
        ret.insert(
            block.label.clone(),
            CfgNode {
                block: block.clone(),
                pred_names: Vec::new(),
                succ_names: succs,
            },
        );
    }
    for name in names {
        if let Some(add_to) = ret.get(&name).map(|n| n.succ_names.clone()) {
            for suc in add_to {
                if let Some(node) = ret.get_mut(&suc) {
                    node.pred_names.push(name.clone());
                }
            }
        }
    }
    ret
}
