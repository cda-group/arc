# Items

An **item** is a named top-level definition which can be referenced by path. The ordering of how items are defined in the source code insignificant. Items can be prefixed by annotations for configuration.

```grammar
Item ::=
  | [Annots]? [Global]      # Global immutable value
  | [Annots]? [Def]         # Function definition
  | [Annots]? [Task]        # Task definition
  | [Annots]? [TypeAlias]   # Type alias
  | [Annots]? [ExternDef]   # Extern function declaration
  | [Annots]? [ExternType]  # Extern type declaration
  | [Annots]? [Enum]        # Disjoint union
  | [Annots]? [Use]         # Import
  | [Annots]? [Module]      # Module
```
