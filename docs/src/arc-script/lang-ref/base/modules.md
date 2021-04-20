# Modules

A **module** is a named unit of encapsulation which may contain items and other modules.

```text
Module ::= 'mod' Name '{' (Item | Module)+ '}'
```

Modules form a hierarchy through nesting. This hierarchy is in addition tied to the file system hierarchy (similar to Rust):

```text
my-project/
  src/
    main.arc     # :: (root module)
    foo/
      mod.arc    # ::foo
      bar/
        mod.arc  # ::foo::bar
    baz/
      mod.arc    # ::foo::baz
```
