# Arc-Runtime Reference

Arc-Runtime is a runtime library, for executing Arc-Lang programs, which focuses on flexibility, portability, native performance, and scalability. Its API provides lower-level abstractions than Arc-Lang that are more general but less safe. Programs written directly in Arc-Runtime can for example deadlock if the programmer is not careful, while Arc-Lang's parser and type system prevents such behavior. Arc-MLIR and Arc-Runtime share different responsibilities in making Arc-Lang programs execute efficiently.

Arc-MLIR supports ahead-of-time standard compiler optimisations as well as logical and physical optimisations of streaming-relational operators. However, Arc-MLIR makes no assumptions of where programs will execute and what the input data will be. That is, all Arc-MLIR optimisations are based on the source code itself.

Arc-Runtime supports runtime optimisations which might rely on information that is only known during execution. Streaming programs are expected to run for an indefinite duration and must therefore be able to adapt to changes in their environment. Among its responsibilities, Arc-Runtime must therefore be able to take care of specialisation, scheduling, and scaling decisions.
