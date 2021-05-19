use proc_macro as pm;
use proc_macro2 as pm2;
use quote::quote;
use syn::*;

use crate::extract;

pub(crate) fn generate(ast: extract::TaskComponents) -> pm2::TokenStream {
    let extract::TaskComponents {
        mod_name,
        task_name,
        iport_enum,
        oport_enum,
        iport_enum_name,
        oport_enum_name,
        iport_name,
        iport_type,
        oport_name,
        oport_type,
        state_name,
        state_type,
        handler_name,
    } = ast;

    let mut poll_outer = Vec::new();
    let mut poll_inner = Vec::new();

    for (i, port) in iport_name.iter().enumerate() {
        let skip = i;
        let handler = quote!({
            match event {
                DataEvent::Data(time, data) => self.handle_data(#port(data), time),
                DataEvent::Watermark(time) => todo!(),
                DataEvent::End => todo!(),
            }
        });
        poll_outer.push(poll_port(&handler, quote!(#skip), quote!(#skip), port));
        poll_inner.push(poll_port(&handler, quote!(#skip), quote!(max_events), port));
    }
    //     for (i, port) in oport_name.iter().enumerate() {
    //         let skip = i + iport_name.len();
    //         handlers_outer.push(handle_port(quote!(handle_oport(event)), quote!(#skip), quote!(#skip), port));
    //         handlers_inner.push(handle_port(quote!(handle_oport(event)), quote!(#skip), quote!(max_events), port));
    //     }

    quote!(
        use #mod_name::#task_name;
        use #mod_name::#iport_enum_name::*;
        use #mod_name::#oport_enum_name::*;
        use #mod_name::*;
        #[allow(clippy::all)]
        #[allow(non_snake_case)]
        mod #mod_name {
            use arctime::prelude::*;
            use super::*;

            pub struct #task_name {
                pub ctx: ComponentContext<Self>,
                pub event_time: DateTime,
                #(pub #iport_name: ProvidedPort<DataPort<#iport_type>>,)*
                #(pub #oport_name: RequiredPort<DataPort<#oport_type>>,)*
                #(pub #state_name: #state_type,)*
            }

            pub #iport_enum

            pub #oport_enum

            use #iport_enum_name::*;
            use #oport_enum_name::*;

            type TaskFn = Box<dyn FnOnce(#(Stream<#iport_type>,)*) -> (#(Stream<#oport_type>,)*)>;

            #[allow(deprecated)]
            pub fn #task_name(#(#state_name: #state_type,)*) -> TaskFn {
                Box::new(#task_name {
                    ctx: ComponentContext::uninitialised(),
                    event_time: DateTime::unix_epoch(),
                    #(#iport_name: ProvidedPort::uninitialised(),)*
                    #(#oport_name: RequiredPort::uninitialised(),)*
                    #(#state_name,)*
                }) as TaskFn
            }

            impl #task_name {
                #[inline(always)]
                fn handle_data(&mut self, data: #iport_enum_name, time: DateTime) -> Handled {
                    self.event_time = time;
                    self.#handler_name(data);
                    Handled::Ok
                }

                // Fix this
                #[inline(always)]
                fn handle_oport<T>(&mut self, data: T) -> Handled {
                    todo!()
                }

                pub fn emit(&mut self, data: #oport_enum_name) {
                    match data {
                        #(#oport_enum_name::#oport_name(data) => {
                            self.#oport_name.trigger(DataEvent::Data(self.event_time, data))
                        },)*
                    }
                }
            }

            // Create a module here to avoid name clashes between function parameters and enum
            // variants.
            mod hygiene {
                use arctime::prelude::*;
                use super::#task_name;

                type Input = (#(Stream<#iport_type>,)*);
                // This code makes it possible to use an instantiated task as a function over streams
                impl FnOnce<Input> for #task_name {
                    type Output = (#(Stream<#oport_type>,)*);
                    extern "rust-call" fn call_once(self, (#(#iport_name,)*): Input) -> Self::Output {
                        // Step 1. Initialise the task
                        // TODO: We currently assume all tasks at least take one stream as input
                        let (stream, ..) = (#(&#iport_name,)*);
                        let task = stream.client.system().create(|| self);
                        // Step 2. Connect the input streams to each of the task's input ports
                        task.on_definition(|producer| {
                            #((#iport_name.connector)(&mut producer.#iport_name);)*
                        });
                        // Step 3. Setup so that the task will be initialised eventually
                        {
                            let client = stream.client.clone();
                            let task = task.clone();
                            stream
                                .start_fns
                                .borrow_mut()
                                .push(Box::new(move || client.system().start(&task)));
                        }
                        // Step 4. Create a stream for each of the task's output ports
                        (#({
                            let producer = task.clone();
                            let connector: Arc<ConnectFn<_>>
                                = Arc::new(move |iport| {
                                producer.on_definition(|producer| {
                                    iport.connect(producer.#oport_name.share());
                                    producer.#oport_name.connect(iport.share());
                                });
                            });
                            let client = stream.client.clone();
                            let start_fns = stream.start_fns.clone();
                            Stream { client, connector, start_fns }
                        },)*)
                    }
                }
            }

            impl ComponentDefinition for #task_name {
                fn setup(&mut self, self_component: Arc<Component<Self>>) {
                    self.ctx.initialise(self_component.clone());
                    #(self.#iport_name.set_parent(self_component.clone());)*
                    #(self.#oport_name.set_parent(self_component.clone());)*
                }

                fn execute(&mut self, max_events: usize, skip: usize) -> ExecuteResult {
                    let mut count: usize = 0;
                    let mut done_work = true;
                    #(#poll_outer)*
                    while done_work {
                        done_work = false;
                        #(#poll_inner)*
                    }
                    ExecuteResult::new(false, count, 0)
                }

                fn ctx_mut(&mut self) -> &mut ComponentContext<Self> {
                    &mut self.ctx
                }

                fn ctx(&self) -> &ComponentContext<Self> {
                    &self.ctx
                }

                fn type_name() -> &'static str {
                    stringify!(#task_name)
                }
            }

            impl Actor for #task_name {
                type Message = Never;

                fn receive_local(&mut self, _: Self::Message) -> Handled {
                    todo!()
                }

                fn receive_network(&mut self, _: NetMessage) -> Handled {
                    todo!()
                }
            }

            impl ComponentLifecycle for #task_name {
                fn on_start(&mut self) -> Handled {
                    Handled::Ok
                }

                fn on_stop(&mut self) -> Handled {
                    Handled::Ok
                }

                fn on_kill(&mut self) -> Handled {
                    Handled::Ok
                }
            }

            impl DynamicPortAccess for #task_name {
                fn get_provided_port_as_any(&mut self, port_id: TypeId) -> Option<&mut dyn Any> {
                    unreachable!();
                }
                fn get_required_port_as_any(&mut self, port_id: TypeId) -> Option<&mut dyn Any> {
                    unreachable!();
                }
            }
        }
    )
}

fn poll_port(
    handler: &pm2::TokenStream,
    treshold: pm2::TokenStream,
    skip: pm2::TokenStream,
    port: &syn::Ident,
) -> pm2::TokenStream {
    quote! {
        if skip <= #treshold {
            if count >= max_events {
                return ExecuteResult::new(false, count, #skip);
            }
        }
        if let Some(event) = self.#port.dequeue() {
            let res = #handler;
            count += 1;
            done_work = true;
            if let Handled::BlockOn(blocking_future) = res {
                self.ctx_mut().set_blocking(blocking_future);
                return ExecuteResult::new(true, count, #skip);
            }
        }
    }
}

fn future(future: Option<syn::Type>) -> pm2::TokenStream {
    if let Some(ty) = future {
        quote! {}
    } else {
        quote! {}
    }
}
