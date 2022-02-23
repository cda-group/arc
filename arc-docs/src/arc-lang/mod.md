# Language Reference

The core concepts of Arc are:

* [**Names**](Name.md)
* [**Paths**](Path.md)
* [**Values**](Value.md)
* [**Types**](Type.md)
* [**Expressions**](Expr.md)
* [**Patterns**](Pattern.md)
* [**Items**](Item.md)

In the following sections, we will explain each concept by presenting its syntax and discussing its semantics through examples. Syntax will be explained using a Regex-based variation of the BNF grammar where:

* `+` and `*` denote repetition.
* `?` is for optional rules.
* `(` `)` indicates grouping.
* `|` is for alternation.
* `[` `]` for is character-alternation (e.g., `abc`).
* `-` is for ranges (e.g., `a-zA-Z`).
* `.` is for matching any character.
* `\` is for escaping characters.
* Non-terminals are written as uppercase (e.g., `Expr`).
* Terminals are written in blue text (e.g., <code><for></code>)

Some code examples do not yet compile. These are highlighted with an orange background.
