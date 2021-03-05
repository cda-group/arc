// compile-flags: --error-format=human
mod script {
    arc_script::include!("non-existent-file.arc")
}

fn main() {

}
