(stmt_def name: (name) @definition.function)
(stmt_type name: (name) @definition.type)
(stmt_enum name: (name) @definition.type)
(pattern_name (name) @definition.var)

(expr_name (name) @reference)
(type_name (name) @reference)
(expr_call function: (expr_name (name) @reference))
(expr_method_call name: ((name) @reference))

[
  (program)
  ((_) @x (#has-parent? @x stmt_def))
  ((_) @x (#has-parent? @x stmt_type))
  ((_) @x (#has-parent? @x stmt_enum))
  (block)
] @scope
