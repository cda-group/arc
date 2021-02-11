use crate::compiler::dfg::DFG;
use crate::compiler::info::diags::Error;
use crate::compiler::info::Info;
use petgraph::algo;

impl DFG {
    fn verify(&mut self, info: &mut Info) {
        if algo::is_cyclic_directed(&self.graph) {
            info.diags.intern(Error::CycleDetected)
        }
    }
}
