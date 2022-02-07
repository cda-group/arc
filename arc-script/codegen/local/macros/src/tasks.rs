#![allow(unused)]

use crate::new_id;

use proc_macro as pm;
use proc_macro2 as pm2;
use quote::quote;
use syn::parse::*;
use syn::punctuated::Punctuated;
use syn::token::Comma;
use syn::*;

struct TaskComponents {
    mod_name: syn::Ident,
    task_name: syn::Ident,
    iport_enum: syn::ItemEnum,
    oport_enum: syn::ItemEnum,
    abstract_iport_enum_name: syn::Ident,
    abstract_oport_enum_name: syn::Ident,
    concrete_iport_enum_name: syn::Ident,
    concrete_oport_enum_name: syn::Ident,
    iport_name_var: Vec<syn::Ident>,
    iport_name: Vec<syn::Ident>,
    oport_name: Vec<syn::Ident>,
    iport_type: Vec<syn::Type>,
    oport_type: Vec<syn::Type>,
    param_name: Vec<syn::Ident>,
    param_type: Vec<syn::Type>,
    state_name: Vec<syn::Ident>,
    state_type: Vec<syn::Type>,
    on_event_name: syn::Ident,
    on_start_name: syn::Ident,
}

fn extract(attr: AttributeArgs, module: ItemMod) -> TaskComponents {
    let mod_name = module.ident.clone();

    let items = module.content.expect("Expected module to contain items").1;

    let task = items
        .iter()
        .find_map(|item| match item {
            Item::Struct(item) => Some(item),
            _ => None,
        })
        .expect("Expected a task-struct in module");

    let task_name = task.ident.clone();

    let (states, params): (Vec<_>, Vec<_>) = task
        .fields
        .clone()
        .into_iter()
        .partition(|f| has_attr_key("state", &f.attrs));

    let (state_name, state_type) = get_name_type(&states);
    let (param_name, param_type) = get_name_type(&params);

    let mut enums = items.iter().filter_map(|item| match item {
        Item::Enum(item) => Some(item),
        _ => None,
    });
    let iport_enum = enums.next().expect("Expected input port enum").clone();
    let oport_enum = enums.next().expect("Expected output port enum").clone();

    let abstract_iport_enum_name = iport_enum.ident.clone();
    let abstract_oport_enum_name = oport_enum.ident.clone();

    let concrete_iport_enum_name = new_id(format!("Concrete{}", iport_enum.ident));
    let concrete_oport_enum_name = new_id(format!("Concrete{}", oport_enum.ident));

    let on_event_name = get_attr_val("on_event", &attr);
    let on_start_name = get_attr_val("on_start", &attr);

    let iport_name = iport_enum
        .variants
        .iter()
        .map(|variant| variant.ident.clone())
        .collect::<Vec<_>>();
    let oport_name = oport_enum
        .variants
        .iter()
        .map(|variant| variant.ident.clone())
        .collect();
    let iport_name_var = iport_name
        .iter()
        .map(|ident| new_id(&format!("_{}", ident)))
        .collect();

    let iport_type = iport_enum
        .variants
        .iter()
        .map(|variant| variant.fields.iter().next().unwrap().ty.clone())
        .collect();
    let oport_type = oport_enum
        .variants
        .iter()
        .map(|variant| variant.fields.iter().next().unwrap().ty.clone())
        .collect();

    TaskComponents {
        mod_name,
        task_name,
        iport_enum,
        oport_enum,
        abstract_iport_enum_name,
        abstract_oport_enum_name,
        concrete_iport_enum_name,
        concrete_oport_enum_name,
        iport_name_var,
        iport_name,
        oport_name,
        iport_type,
        oport_type,
        param_name,
        param_type,
        state_name,
        state_type,
        on_event_name,
        on_start_name,
    }
}

fn get_attr_val(name: &str, attr: &[NestedMeta]) -> Ident {
    attr.iter()
        .find_map(|arg| match arg {
            NestedMeta::Meta(meta) => match meta {
                Meta::NameValue(nv) if nv.path.is_ident(name) => match &nv.lit {
                    Lit::Str(x) => {
                        Some(x.parse().expect("Expected attr value to be an identifier"))
                    }
                    _ => None,
                },
                _ => None,
            },
            NestedMeta::Lit(lit) => None,
        })
        .unwrap_or_else(|| panic!("`{} = <id>` missing from identifiers", name))
}

fn has_attr_key(name: &str, attr: &[Attribute]) -> bool {
    attr.iter()
        .any(|a| matches!(a.parse_meta(), Ok(Meta::Path(x)) if x.is_ident(name)))
}

fn get_name_type(field: &[Field]) -> (Vec<syn::Ident>, Vec<syn::Type>) {
    field
        .iter()
        .map(|f| (f.ident.clone().expect("Expected named field"), f.ty.clone()))
        .unzip()
}

pub(crate) fn rewrite(attr: syn::AttributeArgs, item: syn::ItemMod) -> pm::TokenStream {
    let TaskComponents {
        mod_name,
        task_name,
        iport_enum,
        oport_enum,
        abstract_iport_enum_name,
        abstract_oport_enum_name,
        concrete_iport_enum_name,
        concrete_oport_enum_name,
        iport_name_var,
        iport_name,
        oport_name,
        iport_type,
        oport_type,
        param_name,
        param_type,
        state_name,
        state_type,
        on_event_name,
        on_start_name,
    } = extract(attr, item);

    let mut poll_outer = Vec::new();
    let mut poll_inner = Vec::new();

    for (i, port) in iport_name.iter().enumerate() {
        let skip = i;
        let on_event = quote! {
            match event {
                StreamEvent::Data(time, data) => self.handle_data(time, #concrete_iport_enum_name::#port(data.convert()).convert()),
                StreamEvent::Watermark(time) => todo!(),
                StreamEvent::End => todo!(),
            }
        };
        poll_outer.push(poll_port(&on_event, quote!(#skip), quote!(#skip), port));
        poll_inner.push(poll_port(
            &on_event,
            quote!(#skip),
            quote!(max_events),
            port,
        ));
    }
    //     for (i, port) in oport_name.iter().enumerate() {
    //         let skip = i + iport_name.len();
    //         on_events_outer.push(handle_port(quote!(handle_oport(event)), quote!(#skip), quote!(#skip), port));
    //         on_events_inner.push(handle_port(quote!(handle_oport(event)), quote!(#skip), quote!(max_events), port));
    //     }

    quote!(
        use #mod_name::*;
        use #mod_name::#concrete_iport_enum_name::*;
        use #mod_name::#concrete_oport_enum_name::*;
        #[allow(clippy::all)]
        #[allow(non_snake_case)]
        mod #mod_name {
            use arc_script::codegen::*;
            use arc_script::codegen::backend::prelude::*;
            use super::*;

            pub struct #task_name {
                pub ctx: ComponentContext<Self>,
                pub event_time: DateTime,
                #(pub #iport_name: ProvidedPort<StreamPort<<#iport_type as Convert>::T>>,)*
                #(pub #oport_name: RequiredPort<StreamPort<<#oport_type as Convert>::T>>,)*
                #(pub #param_name: #param_type,)*
                #(pub #state_name: State<#state_type>,)*
            }

            // Allows the component to be sent across threads even if it contains fields which are
            // not safely sendable. The code generator is required to generate thread-safe code.
            unsafe impl Send for #task_name {}

            #iport_enum
            #oport_enum

            type Input = (#(Stream<<#iport_type as Convert>::T>,)*);
            type Output = (#(Stream<<#oport_type as Convert>::T>),*);
            type TaskFn = Box<dyn ValueFn<Input, Output = Output>>;

            pub fn #task_name(#(#param_name: #param_type,)*) -> TaskFn {
                Box::new(move |#(#iport_name_var: Stream<<#iport_type as Convert>::T>,)*| {
                    #task_name::new(#(#param_name,)*).connect((#(#iport_name_var,)*))
                }) as TaskFn
            }

            impl #task_name {
                #[allow(deprecated)] // NOTE: DateTime::unix_epoch is deprecated
                fn new(#(#param_name: #param_type,)*) -> #task_name {
                    #task_name {
                        ctx: ComponentContext::uninitialised(),
                        event_time: DateTime::unix_epoch(),
                        #(#iport_name: ProvidedPort::uninitialised(),)*
                        #(#oport_name: RequiredPort::uninitialised(),)*
                        #(#param_name,)*
                        #(#state_name: State::Uninitialised,)*
                    }
                }

                fn connect(self, (#(#iport_name_var,)*): Input) -> (#(Stream<<#oport_type as Convert>::T>),*) {
                    // Step 1. Initialise the task
                    // TODO: We currently assume all tasks at least take one stream as input
                    let (stream, ..) = (#(&#iport_name_var,)*);
                    let task = stream.client.system().create(|| self);
                    // Step 2. Connect the input streams to each of the task's input ports
                    task.on_definition(|producer| {
                        #((#iport_name_var.connector)(&mut producer.#iport_name);)*
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
                    // Step 4. Return a stream for each of the task's output ports
                    (#({
                        let producer = task.clone();
                        let connector: Arc<ConnectFn<_>> = Arc::new(move |iport| {
                            producer.on_definition(|producer| {
                                iport.connect(producer.#oport_name.share());
                                producer.#oport_name.connect(iport.share());
                            });
                        });
                        let client = stream.client.clone();
                        let start_fns = stream.start_fns.clone();
                        Stream { client, connector, start_fns }
                    }),*)
                }

                #[inline(always)]
                fn handle_data(&mut self, time: DateTime, data: #abstract_iport_enum_name) -> Handled {
                    self.event_time = time;
                    self.#on_event_name(data);
                    Handled::Ok
                }

                // TODO: Handle events which flow in reverse direction
                #[inline(always)]
                fn handle_contraflow<T>(&mut self, data: T) -> Handled {
                    todo!()
                }

                pub fn emit(&mut self, data: #abstract_oport_enum_name) {
                    match data.concrete.as_ref() {
                        #(
                            #concrete_oport_enum_name::#oport_name(data) =>
                                self.#oport_name.trigger(StreamEvent::Data(self.event_time, data.clone().convert()))
                        ,)*
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
                    self.#on_start_name();
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
                fn get_provided_port_as_any(&mut self, _: TypeId) -> Option<&mut dyn Any> {
                    unreachable!();
                }

                fn get_required_port_as_any(&mut self, _: TypeId) -> Option<&mut dyn Any> {
                    unreachable!();
                }
            }
        }
    )
    .into()
}

fn poll_port(
    on_event: &pm2::TokenStream,
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
            let res = #on_event;
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
