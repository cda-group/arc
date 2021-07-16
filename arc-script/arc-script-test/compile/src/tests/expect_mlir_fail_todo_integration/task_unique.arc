extern type Set() {
    fun contains(x: i32): bool;
    fun add(x: i32): unit;
}

task Unique(): ~i32 by i32 -> ~i32 by i32 {
    val set = crate::Set();
    on event by key => {
        if not set.contains(event) {
            set.add(event);
            emit event by key
        }
    };
}
