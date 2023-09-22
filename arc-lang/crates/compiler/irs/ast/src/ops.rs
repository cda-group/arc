use crate::BinopKind;
use crate::BinopKind::*;
use crate::UnopKind;
use crate::UnopKind::*;

#[macro_export]
macro_rules! binop {
    {+} => {"_add_"};
    {and} => {"_and_"};
    {/} => {"_div_"};
    {==} => {"_eq_"};
    {>=} => {"_geq_"};
    {>} => {"_gt_"};
    {<=} => {"_leq_"};
    {<} => {"_lt_"};
    {*} => {"_mul_"};
    {!=} => {"_neq_"};
    {or} => {"_or_"};
    {-} => {"_sub_"};
    {..} => {"_rexc_"};
    {..=} => {"_rinc_"};
}

#[macro_export]
macro_rules! unop {
    {+} => {"_pos_"};
    {-} => {"_neg_"};
    {!} => {"_not_"};
}

pub const fn unop(kind: UnopKind) -> &'static str {
    match kind {
        UPos => unop!(+),
        UNeg => unop!(-),
        UNot => unop!(!),
    }
}

pub const fn binop(kind: BinopKind) -> &'static str {
    match kind {
        BAdd => binop!(+),
        BAnd => binop!(and),
        BDiv => binop!(/),
        BEq => binop!(==),
        BGeq => binop!(>=),
        BGt => binop!(>),
        BLeq => binop!(<=),
        BLt => binop!(<),
        BMul => binop!(*),
        BNeq => binop!(!=),
        BOr => binop!(or),
        BSub => binop!(-),
        BRExc => binop!(..),
        BRInc => binop!(..=),
    }
}
