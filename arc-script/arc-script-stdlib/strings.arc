# A persistent string. Each mutable operation on the string produces a new string.
extern type String() {
    # Concatenates this `String` with another, producing a new one.
    fun concat(other: String): String;

    # Appends a character to this string.
    fun append(other: char): String;

    # Returns `true` if this string contains a `substring`, else `false`.
    fun contains(substring: String): bool;

    # Truncates this string to the specified length, producing a new one.
    # For example:
    # ```
    # val a = "foo";
    # val b = a.truncate(1);
    # assert(b == "fo");
    # ```
    fun truncate(new_len: size): String;
}
