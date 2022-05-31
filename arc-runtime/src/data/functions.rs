#[macro_export]
macro_rules! declare {
    {
        functions: [$($function_id:ident),* $(,)?],
        tasks: [$($task_id:ident ($state_id:ident)),* $(,)?]
    } => {
        #[derive(Send, Sync, Unpin, NoTrace)]
        pub struct Function<I: 'static, O: 'static> {
            pub ptr: fn(I, Context<Task>) -> O,
            pub tag: FunctionTag,
        }
        use serde_derive::Deserialize;
        use serde_derive::Serialize;
        use serde::Serialize;
        use serde::Deserialize;
        #[derive(Copy, Clone, Send, Debug, Serialize, Deserialize)]
        #[allow(non_camel_case_types)]
        pub enum FunctionTag {
            $($function_id,)*
            Unreachable,
        }
        impl<I, O> Copy for Function<I, O> {}
        impl<I, O> Clone for Function<I, O> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<I, O> std::fmt::Debug for Function<I, O> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "Function<{:?}>", self.tag)
            }
        }
        impl<I, O> SerializeState<SerdeState> for Function<I, O> {
            fn serialize_state<S: Serializer>(&self, s: S, ctx: &SerdeState) -> Result<S::Ok, S::Error> {
                self.tag.serialize(s)
            }
        }
        impl<'de, I, O> DeserializeState<'de, SerdeState> for Function<I, O> {
            fn deserialize_state<D: Deserializer<'de>>(ctx: &mut SerdeState, d: D) -> Result<Self, D::Error> {
                unsafe {
                    match FunctionTag::deserialize(d)? {
                        $(FunctionTag::$function_id => Ok(Function {
                            ptr: std::mem::transmute($function_id as usize),
                            tag: FunctionTag::$function_id
                        }),)*
                        FunctionTag::Unreachable => unreachable!()
                    }
                }
            }
        }
        #[derive(Copy, Clone, Debug, Send, Sync, Unpin, Trace)]
        #[serde_state]
        #[allow(non_camel_case_types)]
        pub struct Task {
            tag: TaskTag,
        }
        #[derive(Copy, Clone, Debug, Send, Sync, Unpin, Trace, From)]
        #[serde_state]
        #[allow(non_camel_case_types)]
        pub enum TaskTag {
            $($task_id($state_id),)*
            Unreachable
        }
        impl Execute for Task {
            fn execute(self, ctx: Context<Self>) -> Pin<Box<dyn Future<Output = ()> + Send>> {
                match self.tag {
                    $(TaskTag::$task_id(state) => Box::pin($task_id((state,), ctx)),)*
                    TaskTag::Unreachable => unreachable!()
                }
            }
        }
    };
}

#[macro_export]
macro_rules! function {
    // Create a function value
    ($fun:ident) => {
        Function {
            ptr: $fun,
            tag: FunctionTag::$fun,
        }
    };
    // Create a function type
    (($($input:ty),* $(,)?) -> $output:ty) => {
        Function<($($input,)*), $output>
    };
}
